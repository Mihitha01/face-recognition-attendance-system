"""
Face Recognition System - GUI Application
==========================================
A modern graphical user interface for the face recognition system.
Built with CustomTkinter for a modern look and feel.
"""

import os
import csv
import threading
import queue
import time
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2

from face_recognition_system import FaceRecognitionSystem
from attendance_system import AttendanceSystem


# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class FaceRecognitionApp(ctk.CTk):
    """Main application window for Face Recognition System."""
    
    def __init__(self):
        super().__init__()
        # Window configuration
        self.title("Face Recognition System")
        self.geometry("1400x900")
        self.minsize(1200, 700)
        self.configure(bg="#181A20")
        # Initialize systems
        self.face_system = FaceRecognitionSystem()
        self.attendance_system = AttendanceSystem()
        # Video capture variables
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_camera_running = False
        self.current_mode = None
        self.register_name = ""
        self.camera_index = 0
        # Frame queue for thread-safe communication
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=10)
        # Current preview label reference
        self.current_preview_label = None
        # Create UI
        self._create_sidebar()
        self._create_main_content()
        self._create_status_bar()
        # Show home page by default
        self._show_home()
        # Start UI update loop
        self._schedule_ui_update()
        # Bind close event
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _create_sidebar(self):
        """Create the sidebar with navigation buttons."""
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=20, fg_color="#23272F")
        self.sidebar.pack(side="left", fill="y", padx=0, pady=0)
        self.sidebar.pack_propagate(False)
        # Logo/Title
        self.logo_label = ctk.CTkLabel(
            self.sidebar,
            text="🎭 Face\nRecognition",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#00BFFF"
        )
        self.logo_label.pack(pady=(40, 50))
        # Navigation buttons
        nav_buttons = [
            ("🏠 Home", self._show_home),
            ("📝 Register Face", self._show_register),
            ("🔍 Recognize", self._show_recognize),
            ("📋 Attendance", self._show_attendance),
            ("👥 Database", self._show_database),
            ("⚙️ Settings", self._show_settings),
        ]
        for text, cmd in nav_buttons:
            btn = ctk.CTkButton(
                self.sidebar,
                text=text,
                font=ctk.CTkFont(size=16, weight="bold"),
                height=48,
                corner_radius=12,
                fg_color="#00BFFF",
                text_color="#181A20",
                hover_color="#009ACD",
                command=cmd
            )
            btn.pack(pady=8, padx=24, fill="x")
        # Appearance mode toggle at bottom
        ctk.CTkLabel(
            self.sidebar,
            text="Appearance:",
            font=ctk.CTkFont(size=13),
            text_color="#B0B3B8"
        ).pack(side="bottom", pady=(0, 5))
        self.appearance_menu = ctk.CTkOptionMenu(
            self.sidebar,
            values=["Dark", "Light", "System"],
            command=self._change_appearance
        )
        self.appearance_menu.pack(side="bottom", pady=(0, 14), padx=24)
    
    def _create_main_content(self):
        """Create the main content area."""
        self.main_frame = ctk.CTkFrame(self, corner_radius=20, fg_color="#181A20")
        self.main_frame.pack(side="left", fill="both", expand=True, padx=0, pady=0)
    
    def _create_status_bar(self):
        """Create the status bar at the bottom."""
        self.status_bar = ctk.CTkFrame(self.main_frame, height=36, corner_radius=12, fg_color="#23272F")
        self.status_bar.pack(side="bottom", fill="x")
        self.status_label = ctk.CTkLabel(
            self.status_bar,
            text="Ready",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#00BFFF"
        )
        self.status_label.pack(side="left", padx=18)
        self.faces_count_label = ctk.CTkLabel(
            self.status_bar,
            text=f"Registered Faces: {len(self.face_system.known_face_names)}",
            font=ctk.CTkFont(size=14),
            text_color="#B0B3B8"
        )
        self.faces_count_label.pack(side="right", padx=18)
    
    def _clear_main_frame(self):
        """Clear all widgets from the main content area except status bar."""
        self._stop_camera()
        self.current_preview_label = None
        for widget in self.main_frame.winfo_children():
            if widget != self.status_bar:
                widget.destroy()
    
    def _update_status(self, message: str):
        """Update the status bar message."""
        self.status_label.configure(text=message)
        self.faces_count_label.configure(
            text=f"Registered Faces: {len(self.face_system.known_face_names)}"
        )
    
    def _schedule_ui_update(self):
        """Schedule the UI update loop."""
        self._update_ui()
        self.after(33, self._schedule_ui_update)  # ~30 FPS
    
    def _update_ui(self):
        """Periodically update UI with frames from camera thread."""
        try:
            # Process frame queue
            while not self.frame_queue.empty():
                try:
                    frame = self.frame_queue.get_nowait()
                    if frame is not None and self.current_preview_label is not None:
                        try:
                            if self.current_preview_label.winfo_exists():
                                self._display_image(frame, self.current_preview_label)
                        except Exception:
                            pass
                except queue.Empty:
                    break
            
            # Process result queue
            while not self.result_queue.empty():
                try:
                    result = self.result_queue.get_nowait()
                    if result:
                        action, data = result
                        self._handle_result(action, data)
                except queue.Empty:
                    break
                    
        except Exception as e:
            print(f"UI update error: {e}")
    
    def _handle_result(self, action: str, data):
        """Handle results from camera thread."""
        try:
            if action == "register_success":
                self._update_status(f"✅ Registered: {data}")
                messagebox.showinfo("Success", f"Successfully registered {data}!")
            elif action == "register_no_face":
                self._update_status("❌ No face detected")
            elif action == "register_multiple_faces":
                self._update_status("❌ Multiple faces - show only one")
            elif action == "attendance_marked":
                self._update_status(f"✅ Attendance marked: {data}")
        except Exception as e:
            print(f"Handle result error: {e}")
    
    # ==================== PAGE: HOME ====================
    def _show_home(self):
        """Show the home page."""
        self._clear_main_frame()
        self._update_status("Home")
        
        content = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=40, pady=40)
        
        # Welcome title
        ctk.CTkLabel(
            content,
            text="Welcome to Face Recognition System",
            font=ctk.CTkFont(size=32, weight="bold")
        ).pack(pady=(50, 20))
        
        ctk.CTkLabel(
            content,
            text="A complete solution for face detection, recognition, and attendance tracking",
            font=ctk.CTkFont(size=16)
        ).pack(pady=(0, 50))
        
        # Feature cards
        cards_frame = ctk.CTkFrame(content, fg_color="transparent")
        cards_frame.pack(fill="x", pady=20)
        
        features = [
            ("📝 Register", "Add new faces", self._show_register),
            ("🔍 Recognize", "Identify faces", self._show_recognize),
            ("📋 Attendance", "Track attendance", self._show_attendance),
            ("👥 Database", "Manage faces", self._show_database),
        ]
        
        for i, (icon, desc, cmd) in enumerate(features):
            card = ctk.CTkFrame(cards_frame)
            card.grid(row=0, column=i, padx=15, pady=10, sticky="nsew")
            cards_frame.grid_columnconfigure(i, weight=1)
            
            ctk.CTkLabel(card, text=icon, font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(20, 10))
            ctk.CTkLabel(card, text=desc, font=ctk.CTkFont(size=12)).pack(pady=(0, 10))
            ctk.CTkButton(card, text="Open", width=100, command=cmd).pack(pady=(10, 20))
        
        # Statistics
        stats_frame = ctk.CTkFrame(content)
        stats_frame.pack(fill="x", pady=30, padx=50)
        
        stats = [
            ("Registered Faces", len(self.face_system.known_face_names)),
            ("Unique Persons", len(set(self.face_system.known_face_names))),
            ("Today's Attendance", len(self.attendance_system.today_attendance)),
        ]
        
        for i, (label, value) in enumerate(stats):
            stats_frame.grid_columnconfigure(i, weight=1)
            ctk.CTkLabel(stats_frame, text=str(value), font=ctk.CTkFont(size=36, weight="bold")).grid(row=0, column=i, pady=(20, 5))
            ctk.CTkLabel(stats_frame, text=label, font=ctk.CTkFont(size=14)).grid(row=1, column=i, pady=(0, 20))
    
    # ==================== PAGE: REGISTER ====================
    def _show_register(self):
        """Show the face registration page."""
        self._clear_main_frame()
        self._update_status("Face Registration")
        
        content = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(content, text="Register New Face", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(10, 20))
        
        # Two-column layout
        columns = ctk.CTkFrame(content, fg_color="transparent")
        columns.pack(fill="both", expand=True)
        columns.grid_columnconfigure(0, weight=1)
        columns.grid_columnconfigure(1, weight=1)
        
        # Left: Camera preview
        preview_frame = ctk.CTkFrame(columns)
        preview_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(preview_frame, text="Preview", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        self.register_preview = ctk.CTkLabel(preview_frame, text="Click 'Start Camera' to begin", width=800, height=600)
        self.register_preview.pack(padx=20, pady=10)
        
        cam_controls = ctk.CTkFrame(preview_frame, fg_color="transparent")
        cam_controls.pack(pady=10)
        
        self.btn_start_register = ctk.CTkButton(cam_controls, text="📷 Start Camera", command=lambda: self._start_camera('register'))
        self.btn_start_register.pack(side="left", padx=5)
        
        self.btn_stop_register = ctk.CTkButton(cam_controls, text="⏹ Stop", command=self._stop_camera, state="disabled")
        self.btn_stop_register.pack(side="left", padx=5)
        
        # Right: Registration form
        form_frame = ctk.CTkFrame(columns)
        form_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(form_frame, text="Registration Details", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        ctk.CTkLabel(form_frame, text="Person's Name:").pack(pady=(20, 5))
        self.name_entry = ctk.CTkEntry(form_frame, width=250, placeholder_text="Enter name")
        self.name_entry.pack(pady=5)
        
        ctk.CTkLabel(form_frame, text="Registration Method:", font=ctk.CTkFont(size=14)).pack(pady=(30, 10))
        
        ctk.CTkButton(form_frame, text="📸 Capture from Camera", width=250, command=self._capture_face).pack(pady=10)
        ctk.CTkButton(form_frame, text="📁 Upload Image", width=250, command=self._upload_image_for_registration).pack(pady=10)
        ctk.CTkButton(form_frame, text="📂 Batch Register", width=250, command=self._batch_register).pack(pady=10)
    
    def _capture_face(self):
        """Capture face from camera for registration."""
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Please enter a name first!")
            return
        if not self.is_camera_running:
            messagebox.showwarning("Warning", "Please start the camera first!")
            return
        self.register_name = name
        self._update_status(f"Capturing face for: {name}")
    
    def _upload_image_for_registration(self):
        """Upload an image for face registration."""
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Please enter a name first!")
            return
        
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.webp"), ("All files", "*.*")]
        )
        
        if file_path:
            if self.face_system.register_face_from_image(file_path, name):
                messagebox.showinfo("Success", f"Successfully registered {name}!")
                self._update_status(f"Registered: {name}")
            else:
                messagebox.showerror("Error", "Failed to register. No face detected.")
    
    def _batch_register(self):
        """Batch register faces from a folder."""
        folder_path = filedialog.askdirectory(title="Select Folder with Person Subfolders")
        
        if folder_path:
            from register_faces_from_folder import register_faces_from_folder
            stats = register_faces_from_folder(folder_path, self.face_system)
            
            if stats:
                messagebox.showinfo("Complete", f"Processed: {stats['total_images']}\nSuccessful: {stats['successful']}\nFailed: {stats['failed']}")
                self._update_status(f"Batch registered {stats['successful']} faces")
    
    # ==================== PAGE: RECOGNIZE ====================
    def _show_recognize(self):
        """Show the face recognition page."""
        self._clear_main_frame()
        self._update_status("Face Recognition")
        
        content = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(content, text="Face Recognition", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(10, 20))
        
        self.recognize_preview = ctk.CTkLabel(content, text="Click 'Start Recognition' to begin", width=800, height=600)
        self.recognize_preview.pack(pady=10)
        
        controls = ctk.CTkFrame(content, fg_color="transparent")
        controls.pack(pady=10)
        
        self.btn_start_recognize = ctk.CTkButton(controls, text="📷 Start Recognition", command=lambda: self._start_camera('recognize'))
        self.btn_start_recognize.pack(side="left", padx=10)
        
        self.btn_stop_recognize = ctk.CTkButton(controls, text="⏹ Stop", command=self._stop_camera, state="disabled")
        self.btn_stop_recognize.pack(side="left", padx=10)
        
        ctk.CTkButton(controls, text="📁 From Image", command=self._recognize_from_image).pack(side="left", padx=10)
    
    def _recognize_from_image(self):
        """Recognize faces from an uploaded image."""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.webp"), ("All files", "*.*")]
        )
        
        if file_path:
            results = self.face_system.recognize_face_in_image(file_path)
            
            if results:
                names = [name for name, _ in results]
                self._update_status(f"Detected: {', '.join(names)}")
                
                image = cv2.imread(file_path)
                for name, (top, right, bottom, left) in results:
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(image, (left, top), (right, bottom), color, 2)
                    cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                if hasattr(self, 'recognize_preview'):
                    self._display_image(image, self.recognize_preview)
            else:
                self._update_status("No faces detected")
    
    # ==================== PAGE: ATTENDANCE ====================
    def _show_attendance(self):
        """Show the attendance tracking page."""
        self._clear_main_frame()
        self._update_status("Attendance System")
        
        content = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(content, text="Attendance Tracking", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(10, 20))
        
        columns = ctk.CTkFrame(content, fg_color="transparent")
        columns.pack(fill="both", expand=True)
        columns.grid_columnconfigure(0, weight=1)
        columns.grid_columnconfigure(1, weight=1)
        
        # Left: Camera
        cam_frame = ctk.CTkFrame(columns)
        cam_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.attendance_preview = ctk.CTkLabel(cam_frame, text="Click 'Start' to begin", width=800, height=600)
        self.attendance_preview.pack(padx=20, pady=20)
        
        cam_controls = ctk.CTkFrame(cam_frame, fg_color="transparent")
        cam_controls.pack(pady=10)
        
        self.btn_start_attendance = ctk.CTkButton(cam_controls, text="▶ Start", command=lambda: self._start_camera('attendance'))
        self.btn_start_attendance.pack(side="left", padx=5)
        
        self.btn_stop_attendance = ctk.CTkButton(cam_controls, text="⏹ Stop", command=self._stop_camera, state="disabled")
        self.btn_stop_attendance.pack(side="left", padx=5)
        
        # Right: Attendance log
        log_frame = ctk.CTkFrame(columns)
        log_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(log_frame, text="Today's Attendance", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        self.attendance_log = ctk.CTkTextbox(log_frame, width=350, height=300)
        self.attendance_log.pack(padx=20, pady=10)
        self._refresh_attendance_log()
        
        btn_frame = ctk.CTkFrame(log_frame, fg_color="transparent")
        btn_frame.pack(pady=10)
        
        ctk.CTkButton(btn_frame, text="🔄 Refresh", command=self._refresh_attendance_log).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="📁 Export", command=self._export_attendance).pack(side="left", padx=5)
    
    def _refresh_attendance_log(self):
        """Refresh the attendance log display."""
        try:
            if not hasattr(self, 'attendance_log'):
                return
            
            self.attendance_log.delete("1.0", "end")
            today = datetime.now().strftime("%Y-%m-%d")
            attendance_file = Path("attendance.csv")
            
            if attendance_file.exists():
                with open(attendance_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        if len(row) >= 4 and row[1] == today:
                            self.attendance_log.insert("end", f"{row[2]} - {row[0]} ({row[3]})\n")
            
            if self.attendance_log.get("1.0", "end").strip() == "":
                self.attendance_log.insert("end", "No attendance records for today")
        except Exception as e:
            print(f"Error: {e}")
    
    def _export_attendance(self):
        """Export attendance to CSV."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=f"attendance_{datetime.now().strftime('%Y%m%d')}.csv"
        )
        if file_path and Path("attendance.csv").exists():
            import shutil
            shutil.copy("attendance.csv", file_path)
            messagebox.showinfo("Export Complete", f"Exported to:\n{file_path}")
    
    # ==================== PAGE: DATABASE ====================
    def _show_database(self):
        """Show the database management page."""
        self._clear_main_frame()
        self._update_status("Database Management")
        
        content = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(content, text="Registered Faces Database", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(10, 20))
        
        # Stats
        unique_names = list(set(self.face_system.known_face_names))
        ctk.CTkLabel(
            content,
            text=f"Total: {len(self.face_system.known_face_names)} encodings | {len(unique_names)} unique persons",
            font=ctk.CTkFont(size=14)
        ).pack(pady=10)
        
        # List of persons
        scroll_frame = ctk.CTkScrollableFrame(content, width=600, height=400)
        scroll_frame.pack(padx=20, pady=10)
        
        from collections import Counter
        name_counts = Counter(self.face_system.known_face_names)
        
        for name, count in sorted(name_counts.items()):
            person_frame = ctk.CTkFrame(scroll_frame)
            person_frame.pack(fill="x", pady=2, padx=5)
            
            ctk.CTkLabel(person_frame, text=f"👤 {name} ({count})", font=ctk.CTkFont(size=14)).pack(side="left", padx=10, pady=5)
            ctk.CTkButton(person_frame, text="🗑", width=30, fg_color="red", hover_color="darkred",
                         command=lambda n=name: self._delete_person(n)).pack(side="right", padx=10, pady=5)
        
        if not name_counts:
            ctk.CTkLabel(scroll_frame, text="No faces registered yet", font=ctk.CTkFont(size=14)).pack(pady=20)
        
        # Buttons
        btn_frame = ctk.CTkFrame(content, fg_color="transparent")
        btn_frame.pack(pady=20)
        
        ctk.CTkButton(btn_frame, text="🗑 Clear All", fg_color="red", hover_color="darkred", command=self._clear_database).pack(side="left", padx=10)
        ctk.CTkButton(btn_frame, text="🔄 Refresh", command=self._show_database).pack(side="left", padx=10)
    
    def _delete_person(self, name: str):
        """Delete a person from the database."""
        if messagebox.askyesno("Confirm", f"Delete all encodings for '{name}'?"):
            indices = [i for i, n in enumerate(self.face_system.known_face_names) if n == name]
            for i in sorted(indices, reverse=True):
                del self.face_system.known_face_encodings[i]
                del self.face_system.known_face_names[i]
            self.face_system.save_encodings()
            self._show_database()
    
    def _clear_database(self):
        """Clear all face encodings."""
        if messagebox.askyesno("Confirm", "Delete ALL registered faces?"):
            self.face_system.known_face_encodings = []
            self.face_system.known_face_names = []
            self.face_system.save_encodings()
            self._show_database()
    
    # ==================== PAGE: SETTINGS ====================
    def _show_settings(self):
        """Show the settings page."""
        self._clear_main_frame()
        self._update_status("Settings")
        
        content = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=40, pady=40)
        
        ctk.CTkLabel(content, text="Settings", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(10, 30))
        
        settings_frame = ctk.CTkFrame(content)
        settings_frame.pack(fill="x", padx=100, pady=10)
        
        ctk.CTkLabel(settings_frame, text="Camera Index:", font=ctk.CTkFont(size=14)).pack(pady=(20, 5))
        self.camera_entry = ctk.CTkEntry(settings_frame, width=100)
        self.camera_entry.insert(0, str(self.camera_index))
        self.camera_entry.pack(pady=(5, 20))
        
        ctk.CTkButton(settings_frame, text="Save Settings", command=self._save_settings).pack(pady=20)
        
        # About
        about_frame = ctk.CTkFrame(content)
        about_frame.pack(fill="x", padx=100, pady=20)
        
        ctk.CTkLabel(about_frame, text="About", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        ctk.CTkLabel(about_frame, text="Face Recognition System v1.0\nBuilt with Python, OpenCV & face_recognition",
                    font=ctk.CTkFont(size=12), justify="center").pack(pady=(0, 20))
    
    def _save_settings(self):
        """Save settings."""
        try:
            self.camera_index = int(self.camera_entry.get())
            messagebox.showinfo("Settings", "Settings saved!")
        except ValueError:
            messagebox.showerror("Error", "Invalid camera index")
    
    # ==================== CAMERA FUNCTIONS ====================
    def _start_camera(self, mode: str):
        """Start the camera for a specific mode."""
        if self.is_camera_running:
            return
        
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.is_camera_running = True
        self.current_mode = mode
        
        # Set current preview label based on mode
        if mode == 'register' and hasattr(self, 'register_preview'):
            self.current_preview_label = self.register_preview
            self.btn_start_register.configure(state="disabled")
            self.btn_stop_register.configure(state="normal")
        elif mode == 'recognize' and hasattr(self, 'recognize_preview'):
            self.current_preview_label = self.recognize_preview
            self.btn_start_recognize.configure(state="disabled")
            self.btn_stop_recognize.configure(state="normal")
        elif mode == 'attendance' and hasattr(self, 'attendance_preview'):
            self.current_preview_label = self.attendance_preview
            self.btn_start_attendance.configure(state="disabled")
            self.btn_stop_attendance.configure(state="normal")
        
        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        # Start camera thread
        threading.Thread(target=self._camera_loop, daemon=True).start()
    
    def _stop_camera(self):
        """Stop the camera."""
        self.is_camera_running = False
        time.sleep(0.1)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.current_preview_label = None
        
        # Reset buttons (with error handling)
        for btn_name in ['btn_start_register', 'btn_stop_register', 'btn_start_recognize', 
                        'btn_stop_recognize', 'btn_start_attendance', 'btn_stop_attendance']:
            try:
                btn = getattr(self, btn_name, None)
                if btn and btn.winfo_exists():
                    if 'start' in btn_name:
                        btn.configure(state="normal")
                    else:
                        btn.configure(state="disabled")
            except:
                pass
    
    def _camera_loop(self):
        """Main camera loop running in a separate thread."""
        import face_recognition
        
        frame_count = 0
        face_locations = []
        face_names = []
        
        while self.is_camera_running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                
                display_frame = frame.copy()
                frame_count += 1
                
                if self.current_mode in ['recognize', 'attendance']:
                    # Process every 4th frame
                    if frame_count % 4 == 0:
                        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                        
                        face_locations = face_recognition.face_locations(rgb_small, model="hog")
                        
                        if face_locations:
                            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
                            face_names = []
                            
                            for encoding in face_encodings:
                                name = self.face_system._match_face(encoding, 0.6)
                                face_names.append(name)
                                
                                if self.current_mode == 'attendance' and name != "Unknown":
                                    if self.attendance_system.mark_attendance(name):
                                        try:
                                            self.result_queue.put_nowait(("attendance_marked", name))
                                        except queue.Full:
                                            pass
                        else:
                            face_names = []
                    
                    # Draw boxes
                    for (top, right, bottom, left), name in zip(face_locations, face_names):
                        top, right, bottom, left = top*4, right*4, bottom*4, left*4
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                        cv2.rectangle(display_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                        cv2.putText(display_frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                elif self.current_mode == 'register':
                    if frame_count % 5 == 0:
                        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                        face_locs = face_recognition.face_locations(rgb_small, model="hog")
                        face_locations = [(t*2, r*2, b*2, l*2) for t, r, b, l in face_locs]
                    
                    for (top, right, bottom, left) in face_locations:
                        cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    cv2.putText(display_frame, "Enter name & click Capture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Handle capture
                    if self.register_name:
                        name = self.register_name
                        self.register_name = ""
                        
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        locs = face_recognition.face_locations(rgb_frame, model="hog")
                        
                        if len(locs) == 1:
                            encodings = face_recognition.face_encodings(rgb_frame, locs)
                            if encodings:
                                self.face_system.known_face_encodings.append(encodings[0])
                                self.face_system.known_face_names.append(name)
                                self.face_system.save_encodings()
                                try:
                                    self.result_queue.put_nowait(("register_success", name))
                                except queue.Full:
                                    pass
                        elif len(locs) == 0:
                            try:
                                self.result_queue.put_nowait(("register_no_face", None))
                            except queue.Full:
                                pass
                        else:
                            try:
                                self.result_queue.put_nowait(("register_multiple_faces", None))
                            except queue.Full:
                                pass
                
                # Put frame in queue
                try:
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.frame_queue.put_nowait(display_frame)
                except queue.Full:
                    pass
                
                time.sleep(0.01)
                
            except Exception as e:
                print(f"Camera error: {e}")
                time.sleep(0.1)
    
    def _display_image(self, frame, label):
        """Display an OpenCV image on a CTk label."""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            max_w, max_h = 800, 600
            scale = min(max_w / w, max_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
            pil_image = Image.fromarray(frame_resized)
            ctk_image = ctk.CTkImage(pil_image, size=(new_w, new_h))
            label.configure(image=ctk_image, text="")
            label.image = ctk_image
        except:
            pass
    
    def _change_appearance(self, mode: str):
        """Change the appearance mode."""
        ctk.set_appearance_mode(mode.lower())
    
    def _on_closing(self):
        """Handle window close event."""
        self.is_camera_running = False
        time.sleep(0.2)
        if self.cap:
            self.cap.release()
        self.destroy()


def main():
    app = FaceRecognitionApp()
    app.mainloop()


if __name__ == "__main__":
    main()
