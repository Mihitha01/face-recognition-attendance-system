"""
Face Recognition Attendance System
===================================
An attendance system that uses face recognition to track attendance.
Records are saved to a CSV file with timestamps.
"""

import os
import csv
from datetime import datetime
import cv2
import face_recognition
import numpy as np
from face_recognition_system import FaceRecognitionSystem


class AttendanceSystem(FaceRecognitionSystem):
    """Attendance system using face recognition."""
    
    def __init__(self, encodings_file="face_encodings.pkl", 
                 attendance_file="attendance.csv"):
        """
        Initialize the attendance system.
        
        Args:
            encodings_file: Path to face encodings database
            attendance_file: Path to attendance CSV file
        """
        super().__init__(encodings_file)
        self.attendance_file = attendance_file
        self.today_attendance = set()
        self._load_today_attendance()
    
    def _load_today_attendance(self):
        """Load today's attendance from file."""
        if not os.path.exists(self.attendance_file):
            return
        
        today = datetime.now().strftime("%Y-%m-%d")
        
        with open(self.attendance_file, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            
            for row in reader:
                if len(row) >= 2 and row[1] == today:
                    self.today_attendance.add(row[0])
    
    def _initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.attendance_file):
            with open(self.attendance_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Name', 'Date', 'Time', 'Status'])
    
    def mark_attendance(self, name, status="Present"):
        """
        Mark attendance for a person.
        
        Args:
            name: Name of the person
            status: Attendance status (Present, Late, etc.)
            
        Returns:
            bool: True if attendance was marked, False if already marked today
        """
        if name == "Unknown":
            return False
        
        if name in self.today_attendance:
            return False
        
        self._initialize_csv()
        
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")
        
        with open(self.attendance_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, date, time, status])
        
        self.today_attendance.add(name)
        print(f"✓ Attendance marked for {name} at {time}")
        return True
    
    def run_attendance_camera(self, tolerance=0.6, scale=0.25, 
                               late_time=None, end_time=None):
        """
        Run attendance system with webcam.
        
        Args:
            tolerance: Face matching tolerance
            scale: Scale factor for processing
            late_time: Time after which attendance is marked as "Late" (HH:MM format)
            end_time: Time to automatically stop (HH:MM format)
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Starting attendance system. Press 'q' to quit.")
        print("Faces will be automatically recognized and attendance marked.")
        
        process_this_frame = True
        face_locations = []
        face_names = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check end time
            if end_time:
                current_time = datetime.now().strftime("%H:%M")
                if current_time >= end_time:
                    print(f"\nEnd time reached ({end_time}). Stopping attendance.")
                    break
            
            if process_this_frame:
                # Resize and convert
                small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                face_names = []
                for encoding in face_encodings:
                    name = self._match_face(encoding, tolerance)
                    face_names.append(name)
                    
                    # Mark attendance
                    if name != "Unknown":
                        # Determine status
                        status = "Present"
                        if late_time:
                            current_time = datetime.now().strftime("%H:%M")
                            if current_time > late_time:
                                status = "Late"
                        
                        self.mark_attendance(name, status)
            
            process_this_frame = not process_this_frame
            
            # Draw results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back
                top = int(top / scale)
                right = int(right / scale)
                bottom = int(bottom / scale)
                left = int(left / scale)
                
                # Color based on attendance status
                if name == "Unknown":
                    color = (0, 0, 255)  # Red
                    status_text = name
                elif name in self.today_attendance:
                    color = (0, 255, 0)  # Green
                    status_text = f"{name} ✓"
                else:
                    color = (255, 165, 0)  # Orange
                    status_text = name
                
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, status_text, (left + 6, bottom - 6),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display info
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, f"Time: {timestamp}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Attendance count: {len(self.today_attendance)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Attendance System', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        self._print_summary()
    
    def _print_summary(self):
        """Print attendance summary."""
        print("\n" + "="*50)
        print("ATTENDANCE SUMMARY")
        print("="*50)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"Total present: {len(self.today_attendance)}")
        print("\nAttendees:")
        for name in sorted(self.today_attendance):
            print(f"  ✓ {name}")
    
    def get_attendance_report(self, date=None):
        """
        Get attendance report for a specific date.
        
        Args:
            date: Date in YYYY-MM-DD format (default: today)
            
        Returns:
            list: List of attendance records
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        if not os.path.exists(self.attendance_file):
            print("No attendance records found.")
            return []
        
        records = []
        
        with open(self.attendance_file, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            
            for row in reader:
                if len(row) >= 2 and row[1] == date:
                    records.append({
                        'name': row[0],
                        'date': row[1],
                        'time': row[2] if len(row) > 2 else '',
                        'status': row[3] if len(row) > 3 else 'Present'
                    })
        
        return records
    
    def print_report(self, date=None):
        """Print formatted attendance report."""
        records = self.get_attendance_report(date)
        
        if not records:
            print(f"No attendance records for {date or 'today'}")
            return
        
        print("\n" + "="*60)
        print(f"ATTENDANCE REPORT - {records[0]['date']}")
        print("="*60)
        print(f"{'Name':<20} {'Time':<12} {'Status':<10}")
        print("-"*60)
        
        for record in records:
            print(f"{record['name']:<20} {record['time']:<12} {record['status']:<10}")
        
        print("-"*60)
        print(f"Total: {len(records)} person(s)")


def main():
    """Main function with interactive menu."""
    system = AttendanceSystem()
    
    while True:
        print("\n" + "="*50)
        print("    FACE RECOGNITION ATTENDANCE SYSTEM")
        print("="*50)
        print("1. Start attendance camera")
        print("2. Register new face (webcam)")
        print("3. Register new face (image)")
        print("4. View today's attendance")
        print("5. View attendance by date")
        print("6. List registered faces")
        print("7. Exit")
        print("="*50)
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            late_time = input("Enter late time (HH:MM) or press Enter for none: ").strip()
            late_time = late_time if late_time else None
            
            end_time = input("Enter end time (HH:MM) or press Enter for none: ").strip()
            end_time = end_time if end_time else None
            
            system.run_attendance_camera(late_time=late_time, end_time=end_time)
        
        elif choice == '2':
            name = input("Enter the person's name: ").strip()
            if name:
                system.register_face_from_webcam(name)
        
        elif choice == '3':
            image_path = input("Enter the image path: ").strip()
            name = input("Enter the person's name: ").strip()
            if name and image_path:
                system.register_face_from_image(image_path, name)
        
        elif choice == '4':
            system.print_report()
        
        elif choice == '5':
            date = input("Enter date (YYYY-MM-DD): ").strip()
            system.print_report(date)
        
        elif choice == '6':
            system.list_registered_faces()
        
        elif choice == '7':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
