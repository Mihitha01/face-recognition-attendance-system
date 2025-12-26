"""
Unit Tests for Face Recognition System
======================================
Tests for the face recognition and attendance system.
"""

import os
import sys
import csv
import pickle
import tempfile
import shutil
import unittest
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from face_recognition_system import FaceRecognitionSystem
from attendance_system import AttendanceSystem


class TestFaceRecognitionSystem(unittest.TestCase):
    """Test cases for FaceRecognitionSystem class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.encodings_file = os.path.join(self.temp_dir, "test_encodings.pkl")
        self.system = FaceRecognitionSystem(encodings_file=self.encodings_file)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_creates_empty_lists(self):
        """Test that initialization creates empty encoding lists."""
        self.assertEqual(self.system.known_face_encodings, [])
        self.assertEqual(self.system.known_face_names, [])
    
    def test_save_and_load_encodings(self):
        """Test saving and loading face encodings."""
        # Add some mock encodings
        mock_encoding = np.random.rand(128)
        self.system.known_face_encodings.append(mock_encoding)
        self.system.known_face_names.append("Test Person")
        
        # Save encodings
        result = self.system.save_encodings()
        self.assertTrue(result)
        self.assertTrue(os.path.exists(self.encodings_file))
        
        # Create new system and load
        new_system = FaceRecognitionSystem(encodings_file=self.encodings_file)
        self.assertEqual(len(new_system.known_face_names), 1)
        self.assertEqual(new_system.known_face_names[0], "Test Person")
        np.testing.assert_array_almost_equal(
            new_system.known_face_encodings[0], mock_encoding
        )
    
    def test_register_face_with_empty_name(self):
        """Test that registering with empty name fails."""
        result = self.system.register_face_from_image("fake_path.jpg", "")
        self.assertFalse(result)
        
        result = self.system.register_face_from_image("fake_path.jpg", "   ")
        self.assertFalse(result)
    
    def test_register_face_with_nonexistent_file(self):
        """Test that registering from nonexistent file fails."""
        result = self.system.register_face_from_image(
            "nonexistent_image.jpg", "Test Person"
        )
        self.assertFalse(result)
    
    def test_match_face_with_no_known_faces(self):
        """Test face matching when no faces are registered."""
        mock_encoding = np.random.rand(128)
        name = self.system._match_face(mock_encoding)
        self.assertEqual(name, "Unknown")
    
    def test_list_registered_faces_empty(self):
        """Test listing faces when none registered."""
        result = self.system.list_registered_faces()
        self.assertEqual(result, [])
    
    def test_list_registered_faces_with_data(self):
        """Test listing registered faces."""
        self.system.known_face_names = ["Alice", "Bob", "Alice"]
        self.system.known_face_encodings = [
            np.random.rand(128) for _ in range(3)
        ]
        
        result = self.system.list_registered_faces()
        self.assertIn("Alice", result)
        self.assertIn("Bob", result)
        self.assertEqual(len(result), 2)  # Unique names
    
    def test_remove_face(self):
        """Test removing a registered face."""
        self.system.known_face_names = ["Alice", "Bob", "Alice"]
        self.system.known_face_encodings = [
            np.random.rand(128) for _ in range(3)
        ]
        
        removed = self.system.remove_face("Alice")
        self.assertEqual(removed, 2)
        self.assertEqual(len(self.system.known_face_names), 1)
        self.assertEqual(self.system.known_face_names[0], "Bob")
    
    def test_remove_nonexistent_face(self):
        """Test removing a face that doesn't exist."""
        removed = self.system.remove_face("Nonexistent")
        self.assertEqual(removed, 0)
    
    def test_supported_image_formats(self):
        """Test that supported image formats are defined."""
        self.assertIn('.jpg', self.system.SUPPORTED_IMAGE_FORMATS)
        self.assertIn('.png', self.system.SUPPORTED_IMAGE_FORMATS)
        self.assertIn('.jpeg', self.system.SUPPORTED_IMAGE_FORMATS)


class TestAttendanceSystem(unittest.TestCase):
    """Test cases for AttendanceSystem class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.encodings_file = os.path.join(self.temp_dir, "test_encodings.pkl")
        self.attendance_file = os.path.join(self.temp_dir, "test_attendance.csv")
        self.system = AttendanceSystem(
            encodings_file=self.encodings_file,
            attendance_file=self.attendance_file
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init_creates_empty_attendance(self):
        """Test that initialization creates empty attendance set."""
        self.assertEqual(len(self.system.today_attendance), 0)
    
    def test_mark_attendance_success(self):
        """Test marking attendance for a person."""
        result = self.system.mark_attendance("Test Person", "Present")
        self.assertTrue(result)
        self.assertIn("Test Person", self.system.today_attendance)
        
        # Check CSV file was created
        self.assertTrue(os.path.exists(self.attendance_file))
    
    def test_mark_attendance_duplicate(self):
        """Test that duplicate attendance is not marked."""
        self.system.mark_attendance("Test Person", "Present")
        result = self.system.mark_attendance("Test Person", "Present")
        self.assertFalse(result)
    
    def test_mark_attendance_unknown(self):
        """Test that 'Unknown' is not marked."""
        result = self.system.mark_attendance("Unknown", "Present")
        self.assertFalse(result)
    
    def test_attendance_csv_format(self):
        """Test that attendance CSV has correct format."""
        self.system.mark_attendance("Test Person", "Present")
        
        with open(self.attendance_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # Check header
        self.assertEqual(rows[0], ['Name', 'Date', 'Time', 'Status'])
        
        # Check data row
        self.assertEqual(rows[1][0], "Test Person")
        self.assertEqual(rows[1][1], datetime.now().strftime("%Y-%m-%d"))
        self.assertEqual(rows[1][3], "Present")
    
    def test_get_attendance_report(self):
        """Test getting attendance report."""
        self.system.mark_attendance("Alice", "Present")
        self.system.mark_attendance("Bob", "Late")
        
        today = datetime.now().strftime("%Y-%m-%d")
        records = self.system.get_attendance_report(today)
        
        self.assertEqual(len(records), 2)
        names = [r['name'] for r in records]
        self.assertIn("Alice", names)
        self.assertIn("Bob", names)
    
    def test_get_attendance_report_empty(self):
        """Test getting report when no records exist."""
        records = self.system.get_attendance_report("2000-01-01")
        self.assertEqual(records, [])
    
    def test_load_today_attendance(self):
        """Test loading today's attendance from existing file."""
        # Create a CSV file with today's attendance
        today = datetime.now().strftime("%Y-%m-%d")
        with open(self.attendance_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Date', 'Time', 'Status'])
            writer.writerow(['Pre-existing', today, '09:00:00', 'Present'])
        
        # Create new system that loads the file
        new_system = AttendanceSystem(
            encodings_file=self.encodings_file,
            attendance_file=self.attendance_file
        )
        
        self.assertIn("Pre-existing", new_system.today_attendance)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.encodings_file = os.path.join(self.temp_dir, "test_encodings.pkl")
        self.attendance_file = os.path.join(self.temp_dir, "test_attendance.csv")
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_inheritance_chain(self):
        """Test that AttendanceSystem properly inherits from FaceRecognitionSystem."""
        system = AttendanceSystem(
            encodings_file=self.encodings_file,
            attendance_file=self.attendance_file
        )
        
        # Should have all parent methods
        self.assertTrue(hasattr(system, 'register_face_from_image'))
        self.assertTrue(hasattr(system, 'run_webcam_recognition'))
        self.assertTrue(hasattr(system, 'list_registered_faces'))
        
        # Should have its own methods
        self.assertTrue(hasattr(system, 'mark_attendance'))
        self.assertTrue(hasattr(system, 'run_attendance_camera'))
    
    def test_encodings_persist_across_instances(self):
        """Test that encodings persist when loading from file."""
        # First instance adds encoding
        system1 = FaceRecognitionSystem(encodings_file=self.encodings_file)
        mock_encoding = np.random.rand(128)
        system1.known_face_encodings.append(mock_encoding)
        system1.known_face_names.append("Persistent Person")
        system1.save_encodings()
        
        # Second instance should load it
        system2 = FaceRecognitionSystem(encodings_file=self.encodings_file)
        self.assertEqual(len(system2.known_face_names), 1)
        self.assertEqual(system2.known_face_names[0], "Persistent Person")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
