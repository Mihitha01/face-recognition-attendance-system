"""
Configuration file for Advanced Face Recognition System
========================================================
Centralized configuration for all advanced features.
"""

# ==================== FACE DETECTION ====================
FACE_DETECTION = {
    # Detection backend: 'mediapipe', 'mtcnn', 'retinaface', 'opencv'
    'backend': 'mediapipe',
    
    # Minimum detection confidence (0.0 - 1.0)
    'min_confidence': 0.5,
    
    # Face recognition tolerance (lower = more strict)
    'recognition_tolerance': 0.6,
    
    # Process every Nth frame (higher = faster but less accurate)
    'frame_skip': 4,
}

# ==================== LIVENESS DETECTION ====================
LIVENESS = {
    # Enable liveness detection
    'enabled': True,
    
    # Duration of liveness check in seconds
    'check_duration': 3.0,
    
    # Minimum number of blinks required
    'min_blinks': 1,
    
    # Eye aspect ratio threshold for blink detection
    'ear_threshold': 0.21,
    
    # Consecutive frames below threshold to count as blink
    'blink_frames': 2,
    
    # Minimum movement score (0.0 - 1.0)
    'movement_threshold': 0.5,
}

# ==================== QUALITY ASSESSMENT ====================
QUALITY = {
    # Enable quality check before registration
    'enabled': True,
    
    # Minimum overall quality score (0.0 - 1.0)
    'min_quality_score': 0.5,
    
    # Brightness range (0-255)
    'brightness_min': 80,
    'brightness_max': 180,
    
    # Blur threshold (Laplacian variance, higher = sharper)
    'blur_threshold': 50,
    
    # Minimum face size in pixels (width * height)
    'min_face_pixels': 6400,  # ~80x80
}

# ==================== EMOTION RECOGNITION ====================
EMOTION = {
    # Enable emotion recognition
    'enabled': True,
    
    # Model type: 'fer', 'keras', 'opencv'
    'model_type': 'fer',
    
    # Minimum confidence for emotion detection
    'min_confidence': 0.3,
    
    # Number of frames to average for stable emotion
    'smoothing_frames': 5,
}

# ==================== DATABASE & BACKUP ====================
DATABASE = {
    # Encodings file path
    'encodings_file': 'face_encodings.pkl',
    
    # Attendance file path
    'attendance_file': 'attendance.csv',
    
    # Enable auto-backup on startup/shutdown
    'auto_backup': True,
    
    # Maximum number of backups to keep
    'max_backups': 10,
    
    # Backup directory
    'backup_dir': 'backups',
}

# ==================== NOTIFICATIONS ====================
NOTIFICATIONS = {
    # Enable toast notifications
    'toast_enabled': True,
    
    # Enable sound notifications
    'sound_enabled': True,
    
    # Enable email notifications
    'email_enabled': False,
    
    # Email configuration (only used if email_enabled = True)
    'email': {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'sender_email': '',  # Fill in your email
        'sender_password': '',  # Fill in your app password
        'recipient_email': '',  # Fill in recipient email
    },
}

# ==================== ANALYTICS ====================
ANALYTICS = {
    # Default number of days for trend analysis
    'trend_days': 30,
    
    # Default number of top attendees to show
    'top_attendees_count': 10,
    
    # Chart DPI for saved images
    'chart_dpi': 300,
    
    # Chart style
    'chart_style': 'dark_background',  # matplotlib style
}

# ==================== CAMERA ====================
CAMERA = {
    # Default camera index
    'default_index': 0,
    
    # Camera resolution
    'width': 640,
    'height': 480,
    
    # Frame rate
    'fps': 30,
    
    # Enable multi-camera support
    'multi_camera': False,
    
    # Available camera indices
    'available_cameras': [0, 1, 2],
}

# ==================== UI SETTINGS ====================
UI = {
    # Appearance mode: 'dark', 'light', 'system'
    'appearance_mode': 'dark',
    
    # Color theme: 'blue', 'green', 'dark-blue'
    'color_theme': 'blue',
    
    # Window size
    'window_width': 1600,
    'window_height': 1000,
    
    # Minimum window size
    'min_width': 1400,
    'min_height': 800,
    
    # Update interval (milliseconds)
    'update_interval': 33,  # ~30 FPS
}

# ==================== PERFORMANCE ====================
PERFORMANCE = {
    # Enable GPU acceleration (requires tensorflow-gpu)
    'use_gpu': False,
    
    # Number of threads for parallel processing
    'num_threads': 4,
    
    # Maximum queue size for frame processing
    'max_queue_size': 2,
    
    # Enable face tracking (faster but less accurate)
    'face_tracking': True,
}

# ==================== SECURITY ====================
SECURITY = {
    # Require liveness check for attendance
    'require_liveness_for_attendance': False,
    
    # Require liveness check for registration
    'require_liveness_for_registration': True,
    
    # Minimum quality for registration
    'require_quality_check': True,
    
    # Enable logging of recognition events
    'log_recognition_events': True,
    
    # Log file path
    'log_file': 'face_recognition.log',
}

# ==================== ADVANCED FEATURES ====================
ADVANCED = {
    # Enable face clustering for unknown faces
    'face_clustering': False,
    
    # Enable age/gender estimation
    'age_gender_estimation': False,
    
    # Enable face attribute analysis
    'face_attributes': False,
    
    # Enable mask detection
    'mask_detection': False,
}
