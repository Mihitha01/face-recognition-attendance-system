"""
Emotion Recognition Module
===========================
Detects facial emotions using deep learning models.
Supports multiple emotions: happy, sad, angry, neutral, surprised, fearful, disgusted.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class EmotionRecognizer:
    """Recognize facial emotions using deep learning."""
    
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    def __init__(self, model_type: str = "opencv"):
        """
        Initialize emotion recognizer.
        
        Args:
            model_type: Type of model ('opencv', 'keras', 'fer')
        """
        self.model_type = model_type.lower()
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the emotion recognition model."""
        try:
            if self.model_type == "fer":
                self._init_fer()
            elif self.model_type == "keras":
                self._init_keras()
            else:
                self._init_opencv()
        except Exception as e:
            logger.error(f"Failed to initialize {self.model_type} model: {e}")
            logger.info("Emotion recognition will not be available")
    
    def _init_fer(self):
        """Initialize FER (Facial Expression Recognition) library."""
        try:
            from fer import FER
            self.model = FER(mtcnn=False)
            logger.info("FER emotion detector initialized")
        except ImportError:
            logger.error("FER not installed. Install with: pip install fer")
            raise
    
    def _init_keras(self):
        """Initialize Keras-based emotion model."""
        try:
            from tensorflow import keras
            # You would load a pre-trained model here
            # For now, we'll note this as a placeholder
            logger.warning("Keras emotion model not configured. Using FER fallback.")
            self._init_fer()
        except ImportError:
            logger.error("TensorFlow/Keras not installed")
            raise
    
    def _init_opencv(self):
        """Initialize OpenCV-based emotion detection (if available)."""
        logger.info("Using basic emotion detection")
        # Placeholder for OpenCV-based emotion detection
        self.model = "opencv_basic"
    
    def recognize_emotion(self, frame: np.ndarray, 
                         face_location: Optional[Tuple[int, int, int, int]] = None) -> Dict:
        """
        Recognize emotion in a face.
        
        Args:
            frame: Full image frame
            face_location: Optional (top, right, bottom, left) of face
            
        Returns:
            Dictionary with emotion probabilities and dominant emotion
        """
        if self.model is None:
            return self._default_emotion_result()
        
        try:
            if self.model_type == "fer":
                return self._recognize_fer(frame, face_location)
            else:
                return self._recognize_basic(frame, face_location)
        except Exception as e:
            logger.error(f"Emotion recognition error: {e}")
            return self._default_emotion_result()
    
    def _recognize_fer(self, frame: np.ndarray, 
                       face_location: Optional[Tuple[int, int, int, int]] = None) -> Dict:
        """Recognize emotion using FER library."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # If face location provided, crop to that region
        if face_location:
            top, right, bottom, left = face_location
            rgb_frame = rgb_frame[top:bottom, left:right]
        
        # Detect emotions
        result = self.model.detect_emotions(rgb_frame)
        
        if result and len(result) > 0:
            emotions = result[0]['emotions']
            dominant_emotion = max(emotions, key=emotions.get)
            
            return {
                'emotions': emotions,
                'dominant': dominant_emotion,
                'confidence': emotions[dominant_emotion],
                'detected': True
            }
        
        return self._default_emotion_result()
    
    def _recognize_basic(self, frame: np.ndarray, 
                        face_location: Optional[Tuple[int, int, int, int]] = None) -> Dict:
        """Basic emotion recognition (placeholder)."""
        # This is a simplified version - in production you'd use a real model
        # For now, we return neutral as default
        return {
            'emotions': {emotion: 0.14 for emotion in self.EMOTIONS},
            'dominant': 'neutral',
            'confidence': 0.14,
            'detected': False
        }
    
    def _default_emotion_result(self) -> Dict:
        """Return default emotion result when detection fails."""
        return {
            'emotions': {emotion: 0.0 for emotion in self.EMOTIONS},
            'dominant': 'unknown',
            'confidence': 0.0,
            'detected': False
        }
    
    def get_emotion_emoji(self, emotion: str) -> str:
        """
        Get emoji representation of an emotion.
        
        Args:
            emotion: Emotion name
            
        Returns:
            Emoji string
        """
        emoji_map = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'surprise': '😮',
            'fear': '😨',
            'disgust': '🤢',
            'neutral': '😐',
            'unknown': '❓'
        }
        return emoji_map.get(emotion.lower(), '❓')
    
    def get_emotion_color(self, emotion: str) -> Tuple[int, int, int]:
        """
        Get BGR color for visualizing an emotion.
        
        Args:
            emotion: Emotion name
            
        Returns:
            BGR color tuple
        """
        color_map = {
            'happy': (0, 255, 0),      # Green
            'sad': (255, 0, 0),        # Blue
            'angry': (0, 0, 255),      # Red
            'surprise': (0, 255, 255), # Yellow
            'fear': (255, 0, 255),     # Magenta
            'disgust': (128, 0, 128),  # Purple
            'neutral': (200, 200, 200) # Gray
        }
        return color_map.get(emotion.lower(), (255, 255, 255))


class EmotionTracker:
    """Track emotions over time for analytics."""
    
    def __init__(self, history_size: int = 100):
        """
        Initialize emotion tracker.
        
        Args:
            history_size: Number of emotion records to keep
        """
        self.history_size = history_size
        self.emotion_history = []
        self.person_emotions = {}  # Track emotions per person
    
    def add_emotion(self, person_name: str, emotion: str, confidence: float, timestamp: float):
        """
        Add an emotion record.
        
        Args:
            person_name: Name of the person
            emotion: Detected emotion
            confidence: Confidence score
            timestamp: Time of detection
        """
        record = {
            'person': person_name,
            'emotion': emotion,
            'confidence': confidence,
            'timestamp': timestamp
        }
        
        self.emotion_history.append(record)
        
        # Keep only recent history
        if len(self.emotion_history) > self.history_size:
            self.emotion_history.pop(0)
        
        # Update person-specific emotions
        if person_name not in self.person_emotions:
            self.person_emotions[person_name] = []
        
        self.person_emotions[person_name].append(record)
        
        # Keep only recent emotions per person
        if len(self.person_emotions[person_name]) > self.history_size:
            self.person_emotions[person_name].pop(0)
    
    def get_dominant_emotion(self, person_name: str, recent_count: int = 10) -> Optional[str]:
        """
        Get the dominant emotion for a person based on recent history.
        
        Args:
            person_name: Name of the person
            recent_count: Number of recent records to consider
            
        Returns:
            Dominant emotion or None
        """
        if person_name not in self.person_emotions:
            return None
        
        recent = self.person_emotions[person_name][-recent_count:]
        
        if not recent:
            return None
        
        # Count emotion occurrences
        emotion_counts = {}
        for record in recent:
            emotion = record['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Return most common emotion
        return max(emotion_counts, key=emotion_counts.get)
    
    def get_emotion_statistics(self) -> Dict:
        """
        Get overall emotion statistics.
        
        Returns:
            Dictionary with emotion statistics
        """
        if not self.emotion_history:
            return {'total': 0, 'distribution': {}}
        
        emotion_counts = {}
        for record in self.emotion_history:
            emotion = record['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        total = len(self.emotion_history)
        distribution = {
            emotion: (count / total) * 100 
            for emotion, count in emotion_counts.items()
        }
        
        return {
            'total': total,
            'distribution': distribution,
            'counts': emotion_counts
        }
    
    def get_person_emotion_stats(self, person_name: str) -> Dict:
        """
        Get emotion statistics for a specific person.
        
        Args:
            person_name: Name of the person
            
        Returns:
            Dictionary with person's emotion statistics
        """
        if person_name not in self.person_emotions:
            return {'total': 0, 'distribution': {}}
        
        records = self.person_emotions[person_name]
        emotion_counts = {}
        
        for record in records:
            emotion = record['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        total = len(records)
        distribution = {
            emotion: (count / total) * 100 
            for emotion, count in emotion_counts.items()
        }
        
        return {
            'total': total,
            'distribution': distribution,
            'counts': emotion_counts,
            'dominant': max(emotion_counts, key=emotion_counts.get) if emotion_counts else None
        }
