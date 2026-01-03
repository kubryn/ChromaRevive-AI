import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, UnidentifiedImageError
import numpy as np
from skimage.color import rgb2lab
from tqdm import tqdm
import sys

# --- CONFIGURATION ---
TRAIN_FOLDER = "train_images"
MODEL_SAVE_NAME = "fast_color_model.pth"
IMAGE_SIZE = 224      
BATCH_SIZE = 16       
EPOCHS = 30            
LEARNING_RATE = 0.0002 

# ==========================================
# 0. INTERACTIVE SETUP
# ==========================================
def get_device():
    print("--- System Check ---")
    gpu_available = torch.cuda.is_available()
    print(f"GPU Available: {'YES' if gpu_available else 'NO'}")
    
    while True:
        choice = input("Do you want to use GPU or CPU? (Enter 'gpu' or 'cpu'): ").strip().lower()
        if choice == 'gpu':
            if gpu_available:
                return 'cuda'
            else:
                print("Error: You chose GPU, but no NVIDIA GPU was detected. Falling back to CPU.")
                return 'cpu'
        elif choice == 'cpu':
            return 'cpu'
        else:
            print("Invalid input. Please type 'gpu' or 'cpu'.")

DEVICE = get_device()
print(f"Training selected on: {DEVICE.upper()}\n")

# ==========================================
# 1. CUSTOM LOSS FUNCTION
# ==========================================
class ColorWeightedLoss(nn.Module):
    def __init__(self):
        super(ColorWeightedLoss, self).__init__()
        self.base_loss = nn.L1Loss(reduction='none')

    def forward(self, pred, target):
        loss = self.base_loss(pred, target)
        color_weight = 1.0 + torch.mean(torch.abs(target), dim=1, keepdim=True) * 2.0
        weighted_loss = loss * color_weight
        return torch.mean(weighted_loss)

# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
class FastColorizer(nn.Module):
    def __init__(self):
        super(FastColorizer, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        original_first_layer = resnet.conv1
        self.encoder_first = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.encoder_first.weight.copy_(original_first_layer.weight.mean(dim=1, keepdim=True))
        
        self.enc1 = nn.Sequential(self.encoder_first, resnet.bn1, resnet.relu, resnet.maxpool)
        self.enc2 = resnet.layer1
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3
        self.enc5 = resnet.layer4
        
        self.dec4 = self._make_dec_block(512 + 256, 256)
        self.dec3 = self._make_dec_block(256 + 128, 128)
        self.dec2 = self._make_dec_block(128 + 64, 64)
        self.dec1 = self._make_dec_block(64 + 64, 64)
        
        self.final = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=3, padding=1),
            nn.Tanh() 
        )

    def _make_dec_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.enc1(x) 
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        
        x5_up = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=True)
        d4 = self.dec4(torch.cat([x5_up, x4], dim=1))
        
        d4_up = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True)
        d3 = self.dec3(torch.cat([d4_up, x3], dim=1))
        
        d3_up = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)
        d2 = self.dec2(torch.cat([d3_up, x2], dim=1))

        d2_up = F.interpolate(d2, size=(x1.shape[2], x1.shape[3]), mode='bilinear', align_corners=True)
        d1 = self.dec1(torch.cat([d2_up, x1], dim=1))
        
        out = F.interpolate(d1, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)
        return self.final(out)

# ==========================================
# 3. UNIVERSAL DATASET (ANY IMAGE FORMAT)
# ==========================================
class ColorDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        
        # Support literally any common image extension
        self.valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', 
                           '.webp', '.gif', '.ppm', '.pgm')
        
        self.files = [f for f in os.listdir(root_dir) if f.lower().endswith(self.valid_exts)]
        
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)), 
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15), 
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            path = os.path.join(self.root_dir, self.files[idx])
            
            # .convert("RGB") is crucial!
            # It handles PNG transparency, GIF palettes, and Greyscale inputs automatically.
            img = Image.open(path).convert("RGB")
            
            img = self.transform(img)
            img_np = np.array(img)
            
            img_lab = rgb2lab(img_np).astype("float32")
            img_lab[:, :, 0] = (img_lab[:, :, 0] / 50.0) - 1.0
            img_lab[:, :, 1:] = (img_lab[:, :, 1:] / 128.0)
            
            img_t = torch.from_numpy(img_lab.transpose((2, 0, 1)))
            return img_t[[0], ...], img_t[1:, ...]
            
        except (UnidentifiedImageError, OSError, Exception) as e:
            # If an image is corrupt, don't crash. Return a blank black image.
            # This allows training to continue even if one file is bad.
            # print(f"Skipping corrupt file: {self.files[idx]}")
            return torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE), torch.zeros(2, IMAGE_SIZE, IMAGE_SIZE)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train():
    if not os.path.exists(TRAIN_FOLDER):
        print(f"ERROR: Folder '{TRAIN_FOLDER}' missing.")
        print("Please create the folder and put your images inside.")
        input("Press Enter to exit...")
        return

    dataset = ColorDataset(TRAIN_FOLDER)
    if len(dataset) == 0:
        print(f"ERROR: No images found in {TRAIN_FOLDER}")
        print(f"Supported formats: JPG, PNG, WEBP, BMP, TIFF, GIF, etc.")
        return

    use_pin = (DEVICE == 'cuda')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=use_pin)
    
    print(f"Initializing Model...")
    model = FastColorizer().to(DEVICE)
    
    criterion = ColorWeightedLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    scaler = torch.amp.GradScaler('cuda') if DEVICE == 'cuda' else None

    print(f"Starting training on {len(dataset)} images for {EPOCHS} epochs.")
    
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(loader, leave=True)
        running_loss = 0.0
        
        for L, ab in loop:
            L, ab = L.to(DEVICE), ab.to(DEVICE)
            
            optimizer.zero_grad()
            
            if DEVICE == 'cuda':
                with torch.amp.autocast('cuda'):
                    pred_ab = model(L)
                    loss = criterion(pred_ab, ab)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_ab = model(L)
                loss = criterion(pred_ab, ab)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        scheduler.step()
        
        avg_loss = running_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_SAVE_NAME)

    print("\n-------------------------------------------")
    print(f"Training Complete!")
    print(f"Best Loss: {best_loss:.5f}")
    print(f"Model saved to: {MODEL_SAVE_NAME}")
    print("-------------------------------------------")

if __name__ == "__main__":
    train()