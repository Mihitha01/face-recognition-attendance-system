# Installation Guide - Advanced Face Recognition System v2.0

## 📋 System Requirements

- **OS**: Windows 10/11, Linux (Ubuntu 20.04+), macOS 10.15+
- **Python**: 3.8 or higher (3.9-3.11 recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 2GB free space
- **Camera**: Webcam for real-time features
- **GPU**: Optional (NVIDIA GPU with CUDA for faster processing)

## 🔧 Installation Steps

### Step 1: Install Python (if not already installed)

**Windows:**
Download from [python.org](https://www.python.org/downloads/) and install with "Add to PATH" checked.

**Linux:**
```bash
sudo apt update
sudo apt install python3.9 python3-pip python3-dev
```

**macOS:**
```bash
brew install python@3.9
```

### Step 2: Install Build Tools

**Windows:**
Install Visual Studio Build Tools:
- Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Install "Desktop development with C++"

Or install CMake:
```bash
# Using Chocolatey
choco install cmake
```

**Linux:**
```bash
sudo apt update
sudo apt install build-essential cmake
sudo apt install libgtk-3-dev libboost-all-dev
sudo apt install python3-dev libdlib-dev
```

**macOS:**
```bash
xcode-select --install
brew install cmake
```

### Step 3: Clone or Download Project

```bash
git clone https://github.com/yourusername/face-recognition.git
cd face-recognition
```

Or download and extract ZIP file.

### Step 4: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 5: Install Core Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- face-recognition
- opencv-python
- customtkinter
- pillow
- numpy
- dlib (may take 5-10 minutes)

### Step 6: Install Advanced Features (Optional but Recommended)

**Option A: Install All Features**
```bash
pip install mtcnn retina-face mediapipe fer tensorflow matplotlib win10toast
```

**Option B: Install Selectively**

For multi-model detection:
```bash
pip install mtcnn retina-face mediapipe
```

For emotion recognition:
```bash
pip install fer tensorflow
```

For analytics:
```bash
pip install matplotlib
```

For notifications (Windows):
```bash
pip install win10toast
```

### Step 7: Verify Installation

Run the demo script:
```bash
python demo_advanced_features.py
```

If successful, you'll see all available features listed.

## 🎮 Running the Application

### Original GUI (v1.0)
```bash
python ui_app.py
```

### Advanced GUI (v2.0)
```bash
python advanced_ui_app.py
```

## 🔍 Troubleshooting

### Issue: dlib installation fails

**Solution 1 - Use pre-built wheel (Windows):**
```bash
pip install https://github.com/jloh02/dlib/releases/download/v19.24.1/dlib-19.24.1-cp39-cp39-win_amd64.whl
```
(Replace cp39 with your Python version: cp38, cp310, cp311)

**Solution 2 - Install via conda:**
```bash
conda install -c conda-forge dlib
```

**Solution 3 - Use dlib-bin:**
```bash
pip install dlib-bin
```

### Issue: CMake not found

**Windows:**
```bash
# Download from https://cmake.org/download/
# Or use Chocolatey:
choco install cmake
```

**Linux:**
```bash
sudo apt install cmake
```

### Issue: face_recognition import error

```bash
pip uninstall face-recognition
pip install face-recognition --no-cache-dir
```

### Issue: TensorFlow installation problems

For CPU-only version:
```bash
pip install tensorflow-cpu
```

For specific version:
```bash
pip install tensorflow==2.13.0
```

### Issue: Camera not detected

1. Check camera permissions in OS settings
2. Try different camera index (0, 1, 2) in Settings
3. Test camera:
```python
import cv2
cap = cv2.VideoCapture(0)
print(cap.isOpened())
```

### Issue: MTCNN/RetinaFace not working

```bash
pip uninstall mtcnn retina-face
pip install mtcnn==0.1.1
pip install retina-face==0.0.13
```

### Issue: win10toast not working (Windows)

```bash
pip uninstall win10toast
pip install win10toast-click
```

## 🚀 GPU Acceleration (Optional)

For NVIDIA GPU support:

### Step 1: Install CUDA Toolkit
Download from: https://developer.nvidia.com/cuda-downloads
Recommended: CUDA 11.8

### Step 2: Install cuDNN
Download from: https://developer.nvidia.com/cudnn
Extract to CUDA installation directory

### Step 3: Install TensorFlow GPU
```bash
pip uninstall tensorflow
pip install tensorflow-gpu==2.13.0
```

### Step 4: Verify GPU
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

## 📦 Package Versions (Tested)

```
Python: 3.9.13
face-recognition: 1.3.0
opencv-python: 4.12.0
customtkinter: 5.2.2
numpy: 2.2.6
pillow: 12.0.0
dlib: 19.24.1
mtcnn: 0.1.1
retina-face: 0.0.13
mediapipe: 0.10.9
fer: 22.5.0
tensorflow: 2.13.0
matplotlib: 3.7.1
```

## 🐳 Docker Installation (Alternative)

Coming soon: Docker image for easy deployment.

```bash
# Pull image
docker pull username/face-recognition:v2.0

# Run container
docker run -p 8000:8000 --device=/dev/video0 username/face-recognition:v2.0
```

## 🔄 Updating

### Update core packages:
```bash
pip install --upgrade face-recognition opencv-python customtkinter
```

### Update all packages:
```bash
pip install --upgrade -r requirements.txt
```

### Update advanced features:
```bash
pip install --upgrade mtcnn retina-face mediapipe fer tensorflow matplotlib
```

## 🧪 Testing Installation

Run all tests:
```bash
python -m pytest tests/
```

Quick test:
```bash
python demo_advanced_features.py
```

## 📞 Support

If you encounter issues:

1. Check this guide for solutions
2. Check `README_ADVANCED.md` for feature documentation
3. Check `QUICKSTART.md` for usage guide
4. Search existing GitHub issues
5. Create a new issue with:
   - OS and Python version
   - Full error message
   - Steps to reproduce

## ✅ Installation Checklist

- [ ] Python 3.8+ installed
- [ ] Build tools installed (CMake, Visual Studio, etc.)
- [ ] Virtual environment created and activated
- [ ] Core dependencies installed (requirements.txt)
- [ ] Advanced features installed (optional)
- [ ] Demo script runs successfully
- [ ] Application launches without errors
- [ ] Camera detected and working
- [ ] Face detection working

## 🎉 Success!

If all steps completed successfully:
```bash
python advanced_ui_app.py
```

Enjoy your advanced face recognition system! 🚀

---

**Need help?** Check [README_ADVANCED.md](README_ADVANCED.md) for detailed documentation.
