# 🎨 ChromaRevive-AI

**ChromaRevive-AI** is a high-fidelity Deep Learning tool designed to restore color to black-and-white historical footage and photography. Built on PyTorch and ResNet18, it utilizes Transfer Learning to achieve vibrant results with minimal training data.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Alpha-orange)

> 🚧 **Alpha Testing Notice**
> 
> This project is currently in the **Alpha** stage of development. 
> *   You may encounter bugs or performance issues.
> *   Colorization accuracy depends heavily on the training data provided. 
> *   Results may vary: some images/videos will turn out great, while others may look washed out or have artifacts. 
> 
> We are constantly working to improve the model architecture and stability.

## ✨ Features

*   **Universal Processing:** Seamlessly colorizes both single Images (`.jpg`, `.png`) and Videos (`.mp4`).
*   **Smart Architecture:** Uses a pre-trained ResNet-18 U-Net backbone for robust feature recognition.
*   **Temporal Smoothing:** Video processing includes a frame buffer to prevent color flickering between frames.
*   **Ultimate Quality Mode:**
    *   **Test-Time Augmentation (TTA):** Dual-pass prediction (Normal + Mirrored) for maximum stability.
    *   **Lanczos Upscaling:** Mathematical resampling to preserve image sharpness.
    *   **Bilateral Filtering:** Heavy edge-preserving smoothing for a professional, "painted" look.
*   **Interactive CLI:** Auto-detects GPU/CPU and supports drag-and-drop file selection.

## 🤖 AI Development Disclosure

> **Transparency Note:** This project was conceptualized, architected, and coded with the assistance of Artificial Intelligence. While the core logic utilizes standard Deep Learning practices (Lab Color Space, Convolutional Neural Networks), the implementation and optimization of these scripts were generated through human-AI collaboration.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kubryn/ChromaRevive-AI.git
    cd ChromaRevive-AI
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Hardware:**
    *   **GPU:** Highly recommended (NVIDIA CUDA) for faster training and video processing.
    *   **CPU:** Supported, but processing high-quality video will be slow.

## 🚀 Usage

### Step 1: Train the Model
The AI needs to learn what colors look like.

1.  Create a folder named `train_images` inside the project directory.
2.  Add **50-100 colorful images** (JPG/PNG) to that folder. (Tip: Use images similar to your target video content).
3.  Run the trainer:
    ```bash
    python train.py
    ```
4.  Follow the prompts to select GPU or CPU.
5.  The script will save the best performing model as `fast_color_model.pth`.

### Step 2: Colorize!
1.  Have your black-and-white file ready.
2.  Run the main script:
    ```bash
    python main.py
    ```
3.  Select whether you are processing an Image or Video.
4.  Drag and drop your file into the terminal window when asked.
5.  The result will be saved in the same folder with `_ultimate` added to the name.

## ⚙️ Advanced Configuration

You can open `main.py` and modify the **Extreme Quality Settings** at the top of the file to balance speed vs. quality:

```python
SATURATION_BOOST = 1.4       # Increase for more vibrant colors (1.0 = Original)
USE_TTA = True               # Analyze every frame twice (Slower, fixes glitches)
USE_HEAVY_FILTERING = True   # Enable edge-preserving smoothing (Slowest, best look)
USE_LANCZOS_RESIZE = True    # High-quality upscaling
