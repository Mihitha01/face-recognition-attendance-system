# Face Recognition System

A complete face recognition system built with Python, featuring real-time face detection and recognition, face registration, and an attendance tracking system.

## Features

- **Face Detection**: Detect faces in images and video streams
- **Face Registration**: Register new faces from webcam or image files
- **Real-time Recognition**: Recognize faces in real-time using webcam
- **Batch Registration**: Register multiple faces from a folder structure
- **Attendance System**: Track attendance with timestamps and reports

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam (for real-time features)

### Step 1: Install CMake and Build Tools

**Windows:**
```bash
# Install Visual Studio Build Tools or download CMake from https://cmake.org/download/
```

**Linux:**
```bash
sudo apt-get update
sudo apt-get install build-essential cmake
sudo apt-get install libgtk-3-dev libboost-all-dev
```

**macOS:**
```bash
brew install cmake
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

> **Note**: Installing `dlib` might take a few minutes as it needs to compile.

## Usage

### 1. Basic Face Recognition System

```bash
python face_recognition_system.py
```

This launches an interactive menu with options to:
- Register faces from webcam or images
- Run real-time face recognition
- Recognize faces in image files
- Manage registered faces

### 2. Batch Face Registration

Organize your images in folders:
```
known_faces/
├── John/
│   ├── photo1.jpg
│   └── photo2.jpg
├── Jane/
│   └── photo1.jpg
└── Bob/
    └── photo1.jpg
```

Then run:
```bash
python register_faces_from_folder.py known_faces
```

### 3. Attendance System

```bash
python attendance_system.py
```

Features:
- Automatic attendance marking when faces are recognized
- Late attendance tracking
- Attendance reports by date
- CSV export of attendance records

## API Usage

### Using as a Library

```python
from face_recognition_system import FaceRecognitionSystem

# Initialize the system
system = FaceRecognitionSystem()

# Register a face from an image
system.register_face_from_image("path/to/image.jpg", "Person Name")

# Recognize faces in an image
results = system.recognize_face_in_image("path/to/test_image.jpg")
for name, location in results:
    print(f"Found: {name}")

# Run real-time recognition
system.run_webcam_recognition()
```

### Attendance System API

```python
from attendance_system import AttendanceSystem

# Initialize
attendance = AttendanceSystem()

# Start attendance camera
attendance.run_attendance_camera(
    late_time="09:00",  # Mark as late after 9 AM
    end_time="10:00"    # Stop at 10 AM
)

# Get report
records = attendance.get_attendance_report("2024-01-15")
```

## Project Structure

```
face-recognition/
├── face_recognition_system.py    # Main face recognition module
├── attendance_system.py          # Attendance tracking system
├── register_faces_from_folder.py # Batch registration utility
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── known_faces/                  # Folder for batch registration
├── tests/                        # Unit tests
│   └── test_face_recognition.py  # Test suite
├── face_encodings.pkl            # Face database (auto-generated)
└── attendance.csv                # Attendance records (auto-generated)
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Configuration

### Recognition Tolerance

Adjust the `tolerance` parameter (default: 0.6):
- Lower values (0.4) = stricter matching, fewer false positives
- Higher values (0.7) = more lenient matching, may have false positives

```python
system.run_webcam_recognition(tolerance=0.5)
```

### Performance Tuning

Adjust the `scale` parameter for webcam recognition:
- Lower values (0.25) = faster processing, may miss small faces
- Higher values (0.5) = slower processing, better for distant faces

```python
system.run_webcam_recognition(scale=0.25)
```

## Keyboard Shortcuts

- **c**: Capture face (during registration)
- **q**: Quit webcam view

## Troubleshooting

### "No face detected" error
- Ensure good lighting
- Face the camera directly
- Remove obstructions (glasses, masks)

### Slow performance
- Reduce the `scale` parameter
- Close other applications using the webcam

### Installation issues with dlib
- Ensure CMake is installed
- On Windows, install Visual Studio Build Tools
- Try: `pip install cmake` first

## License

MIT License - feel free to use and modify for your projects.

## Acknowledgments

- [face_recognition](https://github.com/ageitgey/face_recognition) library
- [dlib](http://dlib.net/) for face detection models
- [OpenCV](https://opencv.org/) for image processing
