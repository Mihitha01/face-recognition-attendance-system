"""
Liveness Detection Module
==========================
Implements anti-spoofing techniques to detect real faces vs photos/videos.
Uses eye blink detection and head movement analysis.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import logging
from collections import deque
import time

logger = logging.getLogger(__name__)


class LivenessDetector:
    """Detect if a face is live (not a photo or video)."""
    
    def __init__(self):
        """Initialize liveness detector."""
        self.eye_cascade = None
        self.blink_counter = 0
        self.total_blinks = 0
        self.blink_threshold = 0.21  # EAR threshold
        self.consec_frames = 0
        self.blink_frames_threshold = 2
        
        # Head pose tracking
        self.pose_history = deque(maxlen=30)  # Track last 30 frames
        self.start_time = None
        
        self._load_cascades()
    
    def _load_cascades(self):
        """Load Haar cascades for eye detection."""
        try:
            eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
            logger.info("Eye cascade loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load eye cascade: {e}")
    
    def reset(self):
        """Reset liveness detection state."""
        self.blink_counter = 0
        self.total_blinks = 0
        self.consec_frames = 0
        self.pose_history.clear()
        self.start_time = time.time()
    
    def calculate_ear(self, eye: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection.
        
        Args:
            eye: Array of eye landmark points
            
        Returns:
            Eye aspect ratio value
        """
        # Simplified EAR calculation using eye region
        if eye.size == 0:
            return 1.0
        
        h, w = eye.shape[:2]
        if h == 0 or w == 0:
            return 1.0
        
        # Simple ratio: height / width
        ear = h / (w + 1e-6)
        return ear
    
    def detect_blinks(self, frame: np.ndarray, face_location: Tuple[int, int, int, int]) -> dict:
        """
        Detect eye blinks in a face region.
        
        Args:
            frame: Full image frame
            face_location: (top, right, bottom, left) of face
            
        Returns:
            Dictionary with blink detection results
        """
        top, right, bottom, left = face_location
        face_roi = frame[top:bottom, left:right]
        
        if face_roi.size == 0:
            return {'blinks': 0, 'is_blinking': False, 'ear': 1.0}
        
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Detect eyes
        eyes = []
        if self.eye_cascade is not None:
            eyes = self.eye_cascade.detectMultiScale(
                gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
            )
        
        # Calculate average EAR
        ear_values = []
        for (ex, ey, ew, eh) in eyes:
            eye_roi = gray_face[ey:ey+eh, ex:ex+ew]
            ear = self.calculate_ear(eye_roi)
            ear_values.append(ear)
        
        avg_ear = np.mean(ear_values) if ear_values else 1.0
        
        # Blink detection logic
        is_blinking = False
        if avg_ear < self.blink_threshold:
            self.consec_frames += 1
        else:
            if self.consec_frames >= self.blink_frames_threshold:
                self.total_blinks += 1
                is_blinking = True
            self.consec_frames = 0
        
        return {
            'blinks': self.total_blinks,
            'is_blinking': is_blinking,
            'ear': avg_ear,
            'eyes_detected': len(eyes)
        }
    
    def detect_head_movement(self, frame: np.ndarray, 
                            face_location: Tuple[int, int, int, int]) -> dict:
        """
        Detect head movement to verify liveness.
        
        Args:
            frame: Full image frame
            face_location: (top, right, bottom, left) of face
            
        Returns:
            Dictionary with movement detection results
        """
        top, right, bottom, left = face_location
        
        # Calculate face center
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        face_width = right - left
        face_height = bottom - top
        
        # Store current pose
        self.pose_history.append({
            'center': (center_x, center_y),
            'width': face_width,
            'height': face_height,
            'timestamp': time.time()
        })
        
        # Need at least 10 frames to detect movement
        if len(self.pose_history) < 10:
            return {
                'has_movement': False,
                'movement_score': 0.0,
                'message': 'Collecting data...'
            }
        
        # Calculate movement variance
        centers = np.array([p['center'] for p in self.pose_history])
        x_variance = np.var(centers[:, 0])
        y_variance = np.var(centers[:, 1])
        
        # Total movement score
        movement_score = (x_variance + y_variance) / 1000  # Normalize
        
        has_movement = movement_score > 0.5
        
        message = "Natural movement detected" if has_movement else "Please move your head slightly"
        
        return {
            'has_movement': has_movement,
            'movement_score': movement_score,
            'x_variance': x_variance,
            'y_variance': y_variance,
            'message': message
        }
    
    def check_liveness(self, frame: np.ndarray, 
                      face_location: Tuple[int, int, int, int],
                      duration: float = 3.0) -> dict:
        """
        Comprehensive liveness check.
        
        Args:
            frame: Full image frame
            face_location: (top, right, bottom, left) of face
            duration: How long to collect data (seconds)
            
        Returns:
            Dictionary with liveness assessment
        """
        if self.start_time is None:
            self.reset()
        
        # Detect blinks
        blink_result = self.detect_blinks(frame, face_location)
        
        # Detect movement
        movement_result = self.detect_head_movement(frame, face_location)
        
        # Calculate elapsed time
        elapsed = time.time() - self.start_time
        
        # Liveness criteria
        has_blinked = blink_result['blinks'] >= 1
        has_moved = movement_result['has_movement']
        
        # Overall liveness score
        liveness_score = 0.0
        if has_blinked:
            liveness_score += 0.5
        if has_moved:
            liveness_score += 0.5
        
        is_live = liveness_score >= 0.5 and elapsed >= duration
        
        status_message = "Analyzing..."
        if elapsed < duration:
            progress = (elapsed / duration) * 100
            status_message = f"Liveness check: {progress:.0f}%"
        elif is_live:
            status_message = "✓ Live person detected"
        else:
            if not has_blinked:
                status_message = "⚠ Please blink"
            elif not has_moved:
                status_message = "⚠ Please move your head"
        
        return {
            'is_live': is_live,
            'liveness_score': liveness_score,
            'has_blinked': has_blinked,
            'has_moved': has_moved,
            'blinks': blink_result['blinks'],
            'movement_score': movement_result['movement_score'],
            'elapsed_time': elapsed,
            'status': status_message,
            'progress': min((elapsed / duration) * 100, 100)
        }


class TextureAnalyzer:
    """Analyze face texture to detect printed photos."""
    
    @staticmethod
    def analyze_texture(face_image: np.ndarray) -> dict:
        """
        Analyze face texture for print detection.
        
        Args:
            face_image: Cropped face image
            
        Returns:
            Dictionary with texture analysis results
        """
        if face_image.size == 0:
            return {'is_real': False, 'confidence': 0.0}
        
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Calculate local binary patterns (simplified)
        lbp_var = TextureAnalyzer._calculate_lbp_variance(gray)
        
        # Calculate edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Real faces have more texture variation
        # Printed photos tend to have uniform texture
        is_real_texture = lbp_var > 100  # Threshold for texture variance
        is_real_edges = edge_density > 0.05  # Threshold for edge density
        
        confidence = 0.0
        if is_real_texture:
            confidence += 0.5
        if is_real_edges:
            confidence += 0.5
        
        return {
            'is_real': confidence >= 0.5,
            'confidence': confidence,
            'texture_variance': lbp_var,
            'edge_density': edge_density
        }
    
    @staticmethod
    def _calculate_lbp_variance(gray_image: np.ndarray) -> float:
        """Calculate simplified Local Binary Pattern variance."""
        # Simple approximation using gradient variance
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        variance = np.var(magnitude)
        
        return variance
