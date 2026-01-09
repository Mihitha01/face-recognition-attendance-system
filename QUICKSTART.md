# Quick Start Guide - Advanced Face Recognition System v2.0

## 🚀 Quick Installation (5 Minutes)

### Step 1: Install Core Dependencies
```bash
pip install face-recognition opencv-python customtkinter pillow numpy
```

### Step 2: Install Advanced Features
```bash
pip install mtcnn mediapipe fer tensorflow matplotlib
```

### Step 3: Run the Application
```bash
python advanced_ui_app.py
```

## 📝 Quick Usage Guide

### Register Your First Face
1. Click **"📝 Register"** in the sidebar
2. Enter a name in the text field
3. Click **"📷 Start Camera"**
4. Position your face in the camera view
5. Click **"📸 Capture from Camera"**
6. Wait for quality assessment
7. If quality is good, face will be registered automatically

### Recognize Faces
1. Click **"🔍 Recognize"** in the sidebar
2. Click **"📷 Start Recognition"**
3. Show your face to the camera
4. Your name will appear above your face
5. Emotion will be displayed with emoji

### Mark Attendance
1. Click **"📋 Attendance"** in the sidebar
2. Click **"▶ Start"**
3. Look at the camera
4. Your attendance will be marked automatically
5. Check the log on the right side

### View Analytics
1. Click **"📊 Analytics"** in the sidebar
2. Click any visualization button:
   - **Daily Trend**: See attendance over 30 days
   - **Hourly Distribution**: Peak attendance times
   - **Top Attendees**: Most frequent visitors
3. Save charts as PNG images

### Backup Your Database
1. Click **"💾 Backup"** in the sidebar
2. Click **"💾 Create Backup"**
3. Backup is saved in `backups/` folder
4. Export to JSON or SQLite for sharing

## ⚙️ Essential Settings

### Change Detection Backend
1. Go to **"⚙️ Settings"**
2. Select detection backend:
   - **MediaPipe**: Fast & balanced (recommended)
   - **MTCNN**: Most accurate
   - **OpenCV**: Fastest

### Enable/Disable Features
Edit `config.py`:
```python
LIVENESS['enabled'] = True      # Enable liveness detection
QUALITY['enabled'] = True       # Enable quality check
EMOTION['enabled'] = True       # Enable emotion recognition
NOTIFICATIONS['toast_enabled'] = True  # Enable notifications
```

### Adjust Quality Thresholds
```python
QUALITY['min_quality_score'] = 0.5  # 0.0 to 1.0
QUALITY['blur_threshold'] = 50       # Higher = stricter
QUALITY['brightness_min'] = 80       # 0 to 255
```

## 🎯 Common Tasks

### Batch Register Multiple People
1. Organize folder structure:
   ```
   people/
   ├── John_Doe/
   │   ├── photo1.jpg
   │   ├── photo2.jpg
   ├── Jane_Smith/
   │   ├── photo1.jpg
   ```
2. Click **"📝 Register"** → **"📂 Batch Register"**
3. Select the `people/` folder

### Export Database
1. Go to **"💾 Backup"**
2. Choose export format:
   - **JSON**: Human-readable
   - **SQLite**: Database format
3. Save to desired location

### Generate Daily Report
```python
from analytics import AnalyticsDashboard

analytics = AnalyticsDashboard()
stats = analytics.get_daily_statistics()
analytics.export_report("daily_report.json")
```

### Send Email Notifications
1. Edit `config.py`:
   ```python
   NOTIFICATIONS['email_enabled'] = True
   NOTIFICATIONS['email']['sender_email'] = 'your@gmail.com'
   NOTIFICATIONS['email']['sender_password'] = 'your-app-password'
   ```
2. Restart application

## 🐛 Quick Troubleshooting

### Camera Not Working
- Check camera index in Settings (try 0, 1, or 2)
- Ensure no other app is using the camera
- Run: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`

### Face Not Detected
- Ensure good lighting
- Move closer to camera
- Try different detection backend
- Check camera is not blocked

### Low Recognition Accuracy
- Re-register with better quality images
- Use multiple angles (3-5 photos per person)
- Ensure consistent lighting
- Enable quality check during registration

### Slow Performance
- Reduce `CAMERA['width']` and `CAMERA['height']`
- Increase `FACE_DETECTION['frame_skip']` to 6 or 8
- Use 'opencv' backend instead of 'mtcnn'
- Disable emotion recognition if not needed

## 📊 Performance Tips

### For Speed (30+ FPS)
```python
FACE_DETECTION['backend'] = 'mediapipe'
FACE_DETECTION['frame_skip'] = 4
EMOTION['enabled'] = False
CAMERA['width'] = 640
CAMERA['height'] = 480
```

### For Accuracy (<20 FPS but precise)
```python
FACE_DETECTION['backend'] = 'mtcnn'
FACE_DETECTION['frame_skip'] = 2
QUALITY['enabled'] = True
LIVENESS['enabled'] = True
```

### Balanced (25-30 FPS)
```python
FACE_DETECTION['backend'] = 'mediapipe'
FACE_DETECTION['frame_skip'] = 4
QUALITY['enabled'] = True
EMOTION['enabled'] = True
```

## 🎨 Customization

### Change UI Colors
Edit in `advanced_ui_app.py`:
```python
ctk.set_appearance_mode("dark")  # or "light"
ctk.set_default_color_theme("blue")  # or "green", "dark-blue"
```

### Custom Notification Sounds
Edit in `notifications.py`:
```python
sound_map = {
    'success': (1200, 150),  # (frequency_Hz, duration_ms)
    'error': (400, 200),
}
```

## 📚 Next Steps

1. **Explore Analytics**: Generate visualizations and reports
2. **Setup Email Alerts**: Configure email notifications for attendance
3. **Fine-tune Settings**: Adjust thresholds for your environment
4. **Batch Import**: Register multiple people at once
5. **Regular Backups**: Schedule automatic backups

## 💡 Pro Tips

1. **Register in good lighting**: Natural daylight is best
2. **Multiple angles**: 3-5 photos per person improves accuracy
3. **Quality check**: Always enable for registration
4. **Regular backups**: Use auto-backup feature
5. **Monitor analytics**: Check attendance patterns weekly
6. **Update regularly**: Keep dependencies up to date

## 🆘 Getting Help

- Check `README_ADVANCED.md` for detailed documentation
- Review `config.py` for all configuration options
- Check logs in `face_recognition.log`
- Open an issue on GitHub

---

**Ready to start?** Run `python advanced_ui_app.py` and enjoy! 🎉
