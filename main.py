import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
import os
import time
from torchvision import models
from skimage.color import rgb2lab, lab2rgb
from collections import deque

# --- CONFIGURATION ---
MODEL_PATH = "fast_color_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- EXTREME QUALITY SETTINGS ---
# These settings effectively disable "speed optimizations" in favor of quality
SATURATION_BOOST = 1.4       # Higher saturation for more vibrant look
USE_TTA = True               # Analyze every frame twice (Normal + Flipped)
USE_LANCZOS_RESIZE = True    # Use high-quality math to resize colors (Slower)
USE_HEAVY_FILTERING = True   # The "Slow" part: Edge-preserving smoothing
SMOOTHING_BUFFER = 8         # Average last 8 frames for buttery smooth video

# ==========================================
# 1. MODEL DEFINITION
# ==========================================
class FastColorizer(nn.Module):
    def __init__(self):
        super(FastColorizer, self).__init__()
        resnet = models.resnet18(weights=None)
        original_first_layer = resnet.conv1
        self.encoder_first = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
# 2. SLOW & PRECISE INFERENCE LOGIC
# ==========================================
def get_ab_prediction(model, L_tensor):
    with torch.no_grad():
        ab_pred = model(L_tensor)
        ab_pred = ab_pred.cpu().squeeze(0).numpy().transpose(1, 2, 0)
        return ab_pred * 128.0

def process_frame_ultimate(model, frame_bgr):
    # Prepare Input
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_small = cv2.resize(img_rgb, (224, 224))
    
    lab_small = rgb2lab(img_small).astype("float32")
    L_norm = (lab_small[:, :, 0] / 50.0) - 1.0
    L_tensor = torch.from_numpy(L_norm).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
    
    # PASS 1: Standard Prediction
    ab_final = get_ab_prediction(model, L_tensor)
    
    # PASS 2: Test-Time Augmentation (Flip)
    # This doubles the inference time but fixes directional bias
    if USE_TTA:
        L_flipped = torch.flip(L_tensor, [3]) # Flip Horizontal
        ab_flipped = get_ab_prediction(model, L_flipped)
        ab_flipped = np.fliplr(ab_flipped)    # Flip back
        ab_final = (ab_final + ab_flipped) / 2.0

    return ab_final, img_rgb

# ==========================================
# 3. INTERACTIVE RUNNER
# ==========================================
def run():
    print(f"\n=== ULTIMATE QUALITY COLORIZER ===")
    print(f"Device: {DEVICE}")
    print("Optimization: Disabled (Max Quality Mode)")
    
    # 1. Load Model
    model = FastColorizer().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
    except FileNotFoundError:
        print(f"\n[ERROR] Model file '{MODEL_PATH}' not found!")
        print("Please run train.py first.")
        return

    # 2. Select Mode
    while True:
        print("\nWhat file type?")
        print("[1] Image")
        print("[2] Video")
        choice = input("Choice: ").strip()
        if choice in ['1', '2']: break

    is_video_mode = (choice == '2')

    # 3. Select File
    while True:
        file_path = input("\nEnter filename (drag & drop): ").strip().replace('"', '').replace("'", "")
        if os.path.exists(file_path): break
        print("File not found.")

    filename, ext = os.path.splitext(file_path)
    output_path = f"{filename}_ultimate{ext}"

    # --- VIDEO PROCESSING ---
    if is_video_mode:
        cap = cv2.VideoCapture(file_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        ab_buffer = deque(maxlen=SMOOTHING_BUFFER)
        
        print(f"\nProcessing {total_frames} frames.")
        print("NOTE: This will be slow. We are analyzing every pixel.")
        
        frame_idx = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # 1. AI Analysis (Dual Pass)
            ab_pred, img_rgb = process_frame_ultimate(model, frame)
            
            # 2. Temporal Smoothing
            ab_buffer.append(ab_pred)
            avg_ab = np.mean(ab_buffer, axis=0)
            
            # 3. Color Boosting
            avg_ab = avg_ab * SATURATION_BOOST
            
            # 4. High-Quality Upscaling (Lanczos-4)
            # Standard Resize is fast but blurry. Lanczos is slow but sharp.
            resize_method = cv2.INTER_LANCZOS4 if USE_LANCZOS_RESIZE else cv2.INTER_LINEAR
            ab_full = cv2.resize(avg_ab, (width, height), interpolation=resize_method)
            
            # 5. Heavy Bilateral Filtering
            # This is the heavy computation. It smooths color noise but preserves edges.
            if USE_HEAVY_FILTERING:
                a = ab_full[:, :, 0].astype('float32')
                b = ab_full[:, :, 1].astype('float32')
                # d=5, sigma=50 is a good balance for video. Higher d makes it very slow.
                ab_full[:, :, 0] = cv2.bilateralFilter(a, 5, 50, 50) 
                ab_full[:, :, 1] = cv2.bilateralFilter(b, 5, 50, 50)

            # 6. Reconstruct Final Image
            lab_orig = rgb2lab(img_rgb).astype("float32")
            lab_final = np.zeros((height, width, 3), dtype="float32")
            lab_final[:, :, 0] = lab_orig[:, :, 0] # Keep original Lightness
            lab_final[:, :, 1:] = ab_full          # Add new Color
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rgb_final = lab2rgb(lab_final)
            
            bgr_final = cv2.cvtColor((rgb_final * 255).astype("uint8"), cv2.COLOR_RGB2BGR)
            writer.write(bgr_final)
            
            # Stats
            frame_idx += 1
            elapsed = time.time() - start_time
            fps_calc = frame_idx / elapsed if elapsed > 0 else 0
            
            print(f"Frame {frame_idx}/{total_frames} | {fps_calc:.2f} FPS (High Quality Mode)", end='\r')
            
            cv2.imshow('Ultimate Quality Preview', cv2.resize(bgr_final, (0,0), fx=0.5, fy=0.5))
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        cap.release()
        writer.release()
        print(f"\n\nProcessing Complete. File saved to: {output_path}")

    # --- IMAGE PROCESSING ---
    else:
        frame = cv2.imread(file_path)
        print("\nProcessing Image...")
        height, width = frame.shape[:2]
        
        # AI Analysis
        ab_pred, img_rgb = process_frame_ultimate(model, frame)
        ab_pred = ab_pred * SATURATION_BOOST
        
        # High Quality Resize
        print("Applying Lanczos-4 Upscaling...")
        ab_full = cv2.resize(ab_pred, (width, height), interpolation=cv2.INTER_LANCZOS4)
        
        # Heavy Filtering (Stronger settings for single images)
        if USE_HEAVY_FILTERING:
            print("Applying Heavy Bilateral Filter (Edge Preserving)...")
            a = ab_full[:, :, 0].astype('float32')
            b = ab_full[:, :, 1].astype('float32')
            # Higher 'd' (9) for images because we don't care about fps
            ab_full[:, :, 0] = cv2.bilateralFilter(a, 9, 80, 80)
            ab_full[:, :, 1] = cv2.bilateralFilter(b, 9, 80, 80)
            
        # Reconstruct
        lab_orig = rgb2lab(img_rgb).astype("float32")
        lab_final = np.zeros((height, width, 3), dtype="float32")
        lab_final[:, :, 0] = lab_orig[:, :, 0]
        lab_final[:, :, 1:] = ab_full
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rgb_final = lab2rgb(lab_final)
            
        bgr_final = cv2.cvtColor((rgb_final * 255).astype("uint8"), cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(output_path, bgr_final)
        print(f"Saved to: {output_path}")
        
        cv2.imshow('Result', bgr_final)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()