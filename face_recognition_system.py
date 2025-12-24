"""
Face Recognition System
=======================
A complete face recognition system with the following features:
- Face detection
- Face encoding and registration
- Real-time face recognition via webcam
- Image-based face recognition
"""

import os
import pickle
import cv2
import numpy as np
import face_recognition
from datetime import datetime


class FaceRecognitionSystem:
    """Main class for face recognition operations."""
    
    def __init__(self, encodings_file="face_encodings.pkl"):
        """
        Initialize the face recognition system.
        
        Args:
            encodings_file: Path to save/load face encodings
        """
        self.encodings_file = encodings_file
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_encodings()
    
    def load_encodings(self):
        """Load saved face encodings from file."""
        if os.path.exists(self.encodings_file):
            with open(self.encodings_file, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data.get('encodings', [])
                self.known_face_names = data.get('names', [])
            print(f"Loaded {len(self.known_face_names)} face(s) from database.")
        else:
            print("No existing face database found. Starting fresh.")
    
    def save_encodings(self):
        """Save face encodings to file."""
        data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names
        }
        with open(self.encodings_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {len(self.known_face_names)} face(s) to database.")
    
    def register_face_from_image(self, image_path, name):
        """
        Register a new face from an image file.
        
        Args:
            image_path: Path to the image file
            name: Name of the person
            
        Returns:
            bool: True if registration successful, False otherwise
        """
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            return False
        
        # Load and process the image
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        
        if len(face_encodings) == 0:
            print(f"Error: No face detected in {image_path}")
            return False
        
        if len(face_encodings) > 1:
            print(f"Warning: Multiple faces detected. Using the first face.")
        
        # Add the encoding
        self.known_face_encodings.append(face_encodings[0])
        self.known_face_names.append(name)
        self.save_encodings()
        
        print(f"Successfully registered face for: {name}")
        return True
    
    def register_face_from_webcam(self, name):
        """
        Register a new face using the webcam.
        
        Args:
            name: Name of the person
            
        Returns:
            bool: True if registration successful, False otherwise
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return False
        
        print("Press 'c' to capture face, 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Display the frame
            display_frame = frame.copy()
            
            # Detect faces
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            # Draw rectangles around faces
            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Add instructions
            cv2.putText(display_frame, "Press 'c' to capture, 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Registering: {name}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Register Face', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                if len(face_locations) == 1:
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    if face_encodings:
                        self.known_face_encodings.append(face_encodings[0])
                        self.known_face_names.append(name)
                        self.save_encodings()
                        print(f"Successfully registered face for: {name}")
                        cap.release()
                        cv2.destroyAllWindows()
                        return True
                elif len(face_locations) == 0:
                    print("No face detected. Please position your face in front of the camera.")
                else:
                    print("Multiple faces detected. Please ensure only one face is visible.")
            
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return False
    
    def recognize_face_in_image(self, image_path, tolerance=0.6):
        """
        Recognize faces in an image file.
        
        Args:
            image_path: Path to the image file
            tolerance: How strict the face comparison is (lower = stricter)
            
        Returns:
            list: List of tuples (name, location) for each detected face
        """
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            return []
        
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        results = []
        
        for encoding, location in zip(face_encodings, face_locations):
            name = self._match_face(encoding, tolerance)
            results.append((name, location))
        
        return results
    
    def _match_face(self, face_encoding, tolerance=0.6):
        """
        Match a face encoding against known faces.
        
        Args:
            face_encoding: The face encoding to match
            tolerance: How strict the comparison is
            
        Returns:
            str: Name of the matched person or "Unknown"
        """
        if len(self.known_face_encodings) == 0:
            return "Unknown"
        
        # Compare with known faces
        matches = face_recognition.compare_faces(
            self.known_face_encodings, face_encoding, tolerance=tolerance
        )
        face_distances = face_recognition.face_distance(
            self.known_face_encodings, face_encoding
        )
        
        if True in matches:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                return self.known_face_names[best_match_index]
        
        return "Unknown"
    
    def run_webcam_recognition(self, tolerance=0.6, scale=0.25):
        """
        Run real-time face recognition using webcam.
        
        Args:
            tolerance: How strict the face comparison is
            scale: Scale factor for processing (smaller = faster)
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Starting webcam recognition. Press 'q' to quit.")
        
        # Variables for frame skip (process every nth frame)
        process_this_frame = True
        face_locations = []
        face_names = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Only process every other frame for better performance
            if process_this_frame:
                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                face_names = []
                for encoding in face_encodings:
                    name = self._match_face(encoding, tolerance)
                    face_names.append(name)
            
            process_this_frame = not process_this_frame
            
            # Draw results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations
                top = int(top / scale)
                right = int(right / scale)
                bottom = int(bottom / scale)
                left = int(left / scale)
                
                # Choose color based on recognition
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                
                # Draw rectangle
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                
                # Draw label background
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                
                # Draw label text
                cv2.putText(frame, name, (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display
            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def annotate_image(self, image_path, output_path=None, tolerance=0.6):
        """
        Recognize faces in an image and save annotated version.
        
        Args:
            image_path: Path to input image
            output_path: Path to save annotated image (optional)
            tolerance: Face matching tolerance
            
        Returns:
            numpy.ndarray: Annotated image
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image: {image_path}")
            return None
        
        results = self.recognize_face_in_image(image_path, tolerance)
        
        for name, (top, right, bottom, left) in results:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            cv2.rectangle(image, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(image, name, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Saved annotated image to: {output_path}")
        
        return image
    
    def list_registered_faces(self):
        """List all registered faces."""
        if not self.known_face_names:
            print("No faces registered.")
            return []
        
        print(f"\nRegistered faces ({len(self.known_face_names)}):")
        unique_names = list(set(self.known_face_names))
        for name in unique_names:
            count = self.known_face_names.count(name)
            print(f"  - {name} ({count} encoding(s))")
        
        return unique_names
    
    def remove_face(self, name):
        """
        Remove a registered face by name.
        
        Args:
            name: Name of the person to remove
            
        Returns:
            int: Number of encodings removed
        """
        indices_to_remove = [i for i, n in enumerate(self.known_face_names) if n == name]
        
        if not indices_to_remove:
            print(f"No face found with name: {name}")
            return 0
        
        # Remove in reverse order to maintain correct indices
        for i in reversed(indices_to_remove):
            del self.known_face_encodings[i]
            del self.known_face_names[i]
        
        self.save_encodings()
        print(f"Removed {len(indices_to_remove)} encoding(s) for: {name}")
        return len(indices_to_remove)


def main():
    """Main function with interactive menu."""
    system = FaceRecognitionSystem()
    
    while True:
        print("\n" + "="*50)
        print("       FACE RECOGNITION SYSTEM")
        print("="*50)
        print("1. Register face from webcam")
        print("2. Register face from image")
        print("3. Run webcam recognition")
        print("4. Recognize faces in image")
        print("5. List registered faces")
        print("6. Remove a registered face")
        print("7. Exit")
        print("="*50)
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            name = input("Enter the person's name: ").strip()
            if name:
                system.register_face_from_webcam(name)
            else:
                print("Name cannot be empty.")
        
        elif choice == '2':
            image_path = input("Enter the image path: ").strip()
            name = input("Enter the person's name: ").strip()
            if name and image_path:
                system.register_face_from_image(image_path, name)
            else:
                print("Both name and image path are required.")
        
        elif choice == '3':
            system.run_webcam_recognition()
        
        elif choice == '4':
            image_path = input("Enter the image path: ").strip()
            if image_path:
                output_path = input("Enter output path (or press Enter to skip): ").strip()
                output_path = output_path if output_path else None
                
                results = system.recognize_face_in_image(image_path)
                print(f"\nDetected {len(results)} face(s):")
                for name, location in results:
                    print(f"  - {name}")
                
                if output_path:
                    system.annotate_image(image_path, output_path)
                else:
                    # Display the annotated image
                    annotated = system.annotate_image(image_path)
                    if annotated is not None:
                        cv2.imshow('Recognized Faces', annotated)
                        print("Press any key to close the image window...")
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
        
        elif choice == '5':
            system.list_registered_faces()
        
        elif choice == '6':
            name = input("Enter the name to remove: ").strip()
            if name:
                system.remove_face(name)
            else:
                print("Name cannot be empty.")
        
        elif choice == '7':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
