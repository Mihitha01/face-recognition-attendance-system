"""
Advanced Face Detection Module
===============================
Provides multiple face detection backends including MTCNN, RetinaFace,
and MediaPipe for improved accuracy and performance.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AdvancedFaceDetector:
    """Advanced face detection with multiple backends."""
    
    def __init__(self, backend: str = "mtcnn"):
        """
        Initialize advanced face detector.
        
        Args:
            backend: Detection backend ('mtcnn', 'retinaface', 'mediapipe', 'opencv')
        """
        self.backend = backend.lower()
        self.detector = None
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize the selected detection backend."""
        try:
            if self.backend == "mtcnn":
                self._init_mtcnn()
            elif self.backend == "retinaface":
                self._init_retinaface()
            elif self.backend == "mediapipe":
                self._init_mediapipe()
            elif self.backend == "opencv":
                self._init_opencv()
            else:
                logger.warning(f"Unknown backend '{self.backend}', using OpenCV")
                self._init_opencv()
        except Exception as e:
            logger.error(f"Failed to initialize {self.backend}: {e}")
            logger.info("Falling back to OpenCV Haar Cascade")
            self._init_opencv()
    
    def _init_mtcnn(self):
        """Initialize MTCNN detector."""
        try:
            from mtcnn import MTCNN
            self.detector = MTCNN()
            logger.info("MTCNN detector initialized")
        except ImportError:
            logger.error("MTCNN not installed. Install with: pip install mtcnn")
            raise
    
    def _init_retinaface(self):
        """Initialize RetinaFace detector."""
        try:
            from retinaface import RetinaFace
            self.detector = RetinaFace
            logger.info("RetinaFace detector initialized")
        except ImportError:
            logger.error("RetinaFace not installed. Install with: pip install retina-face")
            raise
    
    def _init_mediapipe(self):
        """Initialize MediaPipe Face Detection."""
        try:
            import mediapipe as mp
            self.detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,  # 0 for short-range, 1 for full-range
                min_detection_confidence=0.5
            )
            logger.info("MediaPipe detector initialized")
        except ImportError:
            logger.error("MediaPipe not installed. Install with: pip install mediapipe")
            raise
    
    def _init_opencv(self):
        """Initialize OpenCV Haar Cascade detector."""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        logger.info("OpenCV Haar Cascade detector initialized")
    
    def detect_faces(self, image: np.ndarray, 
                     min_confidence: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image (BGR format)
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of face locations as (top, right, bottom, left) tuples
        """
        if self.detector is None:
            return []
        
        try:
            if self.backend == "mtcnn":
                return self._detect_mtcnn(image, min_confidence)
            elif self.backend == "retinaface":
                return self._detect_retinaface(image, min_confidence)
            elif self.backend == "mediapipe":
                return self._detect_mediapipe(image, min_confidence)
            elif self.backend == "opencv":
                return self._detect_opencv(image)
            return []
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return []
    
    def _detect_mtcnn(self, image: np.ndarray, 
                      min_confidence: float) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MTCNN."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(rgb_image)
        
        faces = []
        for detection in detections:
            if detection['confidence'] >= min_confidence:
                x, y, w, h = detection['box']
                # Convert to (top, right, bottom, left)
                top, right, bottom, left = y, x + w, y + h, x
                faces.append((top, right, bottom, left))
        
        return faces
    
    def _detect_retinaface(self, image: np.ndarray, 
                           min_confidence: float) -> List[Tuple[int, int, int, int]]:
        """Detect faces using RetinaFace."""
        detections = self.detector.detect_faces(image)
        
        faces = []
        for key in detections.keys():
            detection = detections[key]
            if detection.get('score', 0) >= min_confidence:
                facial_area = detection['facial_area']
                x1, y1, x2, y2 = facial_area
                # Convert to (top, right, bottom, left)
                top, right, bottom, left = y1, x2, y2, x1
                faces.append((top, right, bottom, left))
        
        return faces
    
    def _detect_mediapipe(self, image: np.ndarray, 
                          min_confidence: float) -> List[Tuple[int, int, int, int]]:
        """Detect faces using MediaPipe."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_image)
        
        faces = []
        if results.detections:
            h, w = image.shape[:2]
            for detection in results.detections:
                if detection.score[0] >= min_confidence:
                    bbox = detection.location_data.relative_bounding_box
                    # Convert relative coordinates to absolute
                    left = int(bbox.xmin * w)
                    top = int(bbox.ymin * h)
                    right = int((bbox.xmin + bbox.width) * w)
                    bottom = int((bbox.ymin + bbox.height) * h)
                    faces.append((top, right, bottom, left))
        
        return faces
    
    def _detect_opencv(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using OpenCV Haar Cascade."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        faces = []
        for (x, y, w, h) in detections:
            # Convert to (top, right, bottom, left)
            top, right, bottom, left = y, x + w, y + h, x
            faces.append((top, right, bottom, left))
        
        return faces
    
    def detect_with_landmarks(self, image: np.ndarray) -> List[dict]:
        """
        Detect faces with facial landmarks.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of dictionaries containing face location and landmarks
        """
        if self.backend == "mtcnn":
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections = self.detector.detect_faces(rgb_image)
            results = []
            for detection in detections:
                x, y, w, h = detection['box']
                results.append({
                    'location': (y, x + w, y + h, x),
                    'confidence': detection['confidence'],
                    'landmarks': detection['keypoints']
                })
            return results
        elif self.backend == "retinaface":
            detections = self.detector.detect_faces(image)
            results = []
            for key in detections.keys():
                detection = detections[key]
                facial_area = detection['facial_area']
                x1, y1, x2, y2 = facial_area
                results.append({
                    'location': (y1, x2, y2, x1),
                    'confidence': detection.get('score', 0),
                    'landmarks': detection.get('landmarks', {})
                })
            return results
        else:
            # Fallback: basic detection without landmarks
            faces = self.detect_faces(image)
            return [{'location': loc, 'confidence': 1.0, 'landmarks': {}} for loc in faces]


class FaceQualityAssessor:
    """Assess the quality of detected faces for registration."""
    
    @staticmethod
    def assess_quality(image: np.ndarray, face_location: Tuple[int, int, int, int]) -> dict:
        """
        Assess the quality of a face for registration.
        
        Args:
            image: Full image
            face_location: (top, right, bottom, left) of face
            
        Returns:
            Dictionary with quality metrics and overall score
        """
        top, right, bottom, left = face_location
        face = image[top:bottom, left:right]
        
        if face.size == 0:
            return {'overall_score': 0, 'message': 'Invalid face region'}
        
        # Check brightness
        brightness = FaceQualityAssessor._check_brightness(face)
        
        # Check blur
        blur_score = FaceQualityAssessor._check_blur(face)
        
        # Check size
        size_score = FaceQualityAssessor._check_size(face)
        
        # Calculate overall score
        overall = (brightness['score'] + blur_score['score'] + size_score['score']) / 3
        
        message = "Good quality"
        if overall < 0.4:
            message = "Poor quality - improve lighting, focus, and face size"
        elif overall < 0.7:
            message = "Fair quality - could be improved"
        
        return {
            'overall_score': overall,
            'brightness': brightness,
            'blur': blur_score,
            'size': size_score,
            'message': message,
            'acceptable': overall >= 0.5
        }
    
    @staticmethod
    def _check_brightness(face: np.ndarray) -> dict:
        """Check if face is well-lit."""
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if len(face.shape) == 3 else face
        mean_brightness = np.mean(gray)
        
        # Ideal brightness is around 100-150
        if 80 <= mean_brightness <= 180:
            score = 1.0
            message = "Good lighting"
        elif 50 <= mean_brightness < 80 or 180 < mean_brightness <= 200:
            score = 0.6
            message = "Lighting could be better"
        else:
            score = 0.3
            message = "Poor lighting (too dark or too bright)"
        
        return {'score': score, 'value': mean_brightness, 'message': message}
    
    @staticmethod
    def _check_blur(face: np.ndarray) -> dict:
        """Check if face is in focus."""
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if len(face.shape) == 3 else face
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Higher variance = sharper image
        if laplacian_var > 100:
            score = 1.0
            message = "Sharp and clear"
        elif laplacian_var > 50:
            score = 0.7
            message = "Slightly blurry"
        else:
            score = 0.3
            message = "Too blurry - improve focus"
        
        return {'score': score, 'value': laplacian_var, 'message': message}
    
    @staticmethod
    def _check_size(face: np.ndarray) -> dict:
        """Check if face is large enough."""
        h, w = face.shape[:2]
        pixels = h * w
        
        # Ideal face size is at least 80x80 pixels
        if pixels >= 10000:  # ~100x100
            score = 1.0
            message = "Good size"
        elif pixels >= 6400:  # ~80x80
            score = 0.8
            message = "Acceptable size"
        else:
            score = 0.4
            message = "Face too small - move closer"
        
        return {'score': score, 'value': pixels, 'message': message}
