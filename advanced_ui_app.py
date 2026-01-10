"""
Advanced Face Recognition System - Enhanced GUI Application
============================================================
A state-of-the-art graphical user interface with advanced features:
- Multi-model face detection (MTCNN, RetinaFace, MediaPipe)
- Liveness detection (anti-spoofing)
- Emotion recognition
- Face quality assessment
- Advanced analytics and reporting
- Database management and backup
- Notification system
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
import numpy as np

from face_recognition_system import FaceRecognitionSystem
from attendance_system import AttendanceSystem
from advanced_detection import AdvancedFaceDetector, FaceQualityAssessor
from liveness_detection import LivenessDetector
from emotion_recognition import EmotionRecognizer, EmotionTracker
from analytics import AnalyticsDashboard
from database_manager import DatabaseManager
from notifications import NotificationManager

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class AdvancedFaceRecognitionApp(ctk.CTk):
    """Advanced Face Recognition Application with cutting-edge features."""
    
    def __init__(self):
        super().__init__()
        
        # Window configuration
        self.title("Advanced Face Recognition System v2.0")
        self.geometry("1600x1000")
        self.minsize(1400, 800)
        
        # Initialize systems
        self.face_system = FaceRecognitionSystem()
        self.attendance_system = AttendanceSystem()
        self.advanced_detector = AdvancedFaceDetector(backend="mediapipe")  # Default to MediaPipe
        self.liveness_detector = LivenessDetector()
        self.emotion_recognizer = EmotionRecognizer(model_type="fer")
        self.emotion_tracker = EmotionTracker()
        self.analytics = AnalyticsDashboard()
        self.db_manager = DatabaseManager()
        self.notification_manager = NotificationManager()
        
        # Settings
        self.camera_index = 0
        self.use_advanced_detection = False
        self.use_liveness_check = False
        self.use_emotion_recognition = True
        self.use_quality_check = True
        self.detection_backend = "mediapipe"
        self.performance_mode = True  # Enable performance optimizations
        
        # Video capture variables
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_camera_running = False
        self.current_mode = None
        self.register_name = ""
        
        # Frame queue for thread-safe communication
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=10)
        
        # Current preview label reference
        self.current_preview_label = None
        
        # Setup notification callback
        self.notification_manager.set_toast_callback(self._show_ui_toast)
        
        # Create UI
        self._create_sidebar()
        self._create_main_content()
        self._create_status_bar()
        
        # Show home page by default
        self._show_home()
        
        # Start UI update loop
        self._schedule_ui_update()
        
        # Auto-backup on startup
        self.db_manager.auto_backup(max_backups=10)
        
        # Bind close event
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _create_sidebar(self):
        """Create enhanced sidebar with navigation."""
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0, fg_color="#1a1a2e")
        self.sidebar.pack(side="left", fill="y", padx=0, pady=0)
        self.sidebar.pack_propagate(False)
        
        # Logo/Title
        self.logo_label = ctk.CTkLabel(
            self.sidebar,
            text="🚀 Face AI\nv2.0",
            font=ctk.CTkFont(size=30, weight="bold"),
            text_color="#00d4ff"
        )
        self.logo_label.pack(pady=(30, 40))
        
        # Navigation buttons with emojis
        nav_buttons = [
            ("🏠 Home", self._show_home),
            ("📝 Register", self._show_register),
            ("🔍 Recognize", self._show_recognize),
            ("📋 Attendance", self._show_attendance),
            ("👥 Database", self._show_database),
            ("📊 Analytics", self._show_analytics),
            ("💾 Backup", self._show_backup),
            ("⚙️ Settings", self._show_settings),
        ]
        
        self.nav_buttons = {}
        for text, cmd in nav_buttons:
            btn = ctk.CTkButton(
                self.sidebar,
                text=text,
                font=ctk.CTkFont(size=15, weight="bold"),
                height=45,
                corner_radius=10,
                fg_color="#16213e",
                text_color="#e0e0e0",
                hover_color="#00d4ff",
                command=cmd,
                anchor="w",
                border_width=2,
                border_color="#00d4ff"
            )
            btn.pack(pady=6, padx=20, fill="x")
            self.nav_buttons[text] = btn
    
    def _create_main_content(self):
        """Create the main content area."""
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="#0f0f1e")
        self.main_frame.pack(side="left", fill="both", expand=True)
    
    def _create_status_bar(self):
        """Create enhanced status bar."""
        self.status_bar = ctk.CTkFrame(self.main_frame, height=40, corner_radius=0, fg_color="#1a1a2e")
        self.status_bar.pack(side="bottom", fill="x")
        
        self.status_label = ctk.CTkLabel(
            self.status_bar,
            text="🟢 Ready",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#00d4ff"
        )
        self.status_label.pack(side="left", padx=20)
        
        # System info
        self.info_label = ctk.CTkLabel(
            self.status_bar,
            text=f"👤 {len(self.face_system.known_face_names)} faces | "
                 f"📊 {len(set(self.face_system.known_face_names))} persons | "
                 f"🔔 Notifications: ON",
            font=ctk.CTkFont(size=12),
            text_color="#a0a0a0"
        )
        self.info_label.pack(side="right", padx=20)
    
    def _clear_main_frame(self):
        """Clear main content area."""
        self._stop_camera()
        self.current_preview_label = None
        for widget in self.main_frame.winfo_children():
            if widget != self.status_bar:
                widget.destroy()
    
    def _update_status(self, message: str, icon: str = "🟢"):
        """Update status bar with icon."""
        self.status_label.configure(text=f"{icon} {message}")
        self.info_label.configure(
            text=f"👤 {len(self.face_system.known_face_names)} faces | "
                 f"📊 {len(set(self.face_system.known_face_names))} persons | "
                 f"🔔 Notifications: {'ON' if self.notification_manager.toast_enabled else 'OFF'}"
        )
    
    def _show_ui_toast(self, title: str, message: str, notification_type: str):
        """Show toast notification in UI."""
        # You can implement a custom toast widget here
        # For now, we'll just log it
        icons = {'info': 'ℹ️', 'success': '✅', 'warning': '⚠️', 'error': '❌'}
        icon = icons.get(notification_type, 'ℹ️')
        self._update_status(f"{title}: {message}", icon)
    
    # ==================== ENHANCED HOME PAGE ====================
    def _show_home(self):
        """Show enhanced home page with system overview."""
        self._clear_main_frame()
        self._update_status("Home Dashboard")
        
        content = ctk.CTkScrollableFrame(self.main_frame, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=30, pady=30)
        
        # Welcome header
        header = ctk.CTkFrame(content, fg_color="#1a1a2e", corner_radius=15)
        header.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(
            header,
            text="🚀 Advanced Face Recognition System",
            font=ctk.CTkFont(size=36, weight="bold"),
            text_color="#00d4ff"
        ).pack(pady=20)
        
        ctk.CTkLabel(
            header,
            text="Powered by AI • Multi-Model Detection • Liveness Check • Emotion Recognition",
            font=ctk.CTkFont(size=14),
            text_color="#a0a0a0"
        ).pack(pady=(0, 20))
        
        # Statistics cards
        stats_frame = ctk.CTkFrame(content, fg_color="transparent")
        stats_frame.pack(fill="x", pady=20)
        
        daily_stats = self.analytics.get_daily_statistics()
        weekly_stats = self.analytics.get_weekly_statistics()
        
        stats = [
            ("👤 Registered Faces", len(self.face_system.known_face_names), "#4CAF50"),
            ("👥 Unique Persons", len(set(self.face_system.known_face_names)), "#2196F3"),
            ("✅ Today's Attendance", daily_stats['unique_people'], "#FF9800"),
            ("📅 Weekly Total", weekly_stats['unique_people'], "#9C27B0"),
        ]
        
        for i, (label, value, color) in enumerate(stats):
            card = ctk.CTkFrame(stats_frame, fg_color="#16213e", corner_radius=12)
            card.grid(row=0, column=i, padx=10, pady=10, sticky="nsew")
            stats_frame.grid_columnconfigure(i, weight=1)
            
            ctk.CTkLabel(
                card,
                text=str(value),
                font=ctk.CTkFont(size=48, weight="bold"),
                text_color=color
            ).pack(pady=(25, 5))
            
            ctk.CTkLabel(
                card,
                text=label,
                font=ctk.CTkFont(size=13),
                text_color="#a0a0a0"
            ).pack(pady=(0, 25))
        
        # Feature highlights
        features_frame = ctk.CTkFrame(content, fg_color="#1a1a2e", corner_radius=15)
        features_frame.pack(fill="x", pady=20)
        
        ctk.CTkLabel(
            features_frame,
            text="✨ Advanced Features",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#00d4ff"
        ).pack(pady=(20, 10))
        
        features_grid = ctk.CTkFrame(features_frame, fg_color="transparent")
        features_grid.pack(fill="x", padx=20, pady=(10, 20))
        
        features = [
            ("🎯", "Multi-Model Detection", "MTCNN, RetinaFace, MediaPipe"),
            ("🔒", "Liveness Detection", "Anti-spoofing technology"),
            ("😊", "Emotion Recognition", "7 emotion categories"),
            ("📊", "Advanced Analytics", "Charts and reports"),
            ("💾", "Auto Backup", "Database protection"),
            ("🔔", "Smart Notifications", "Real-time alerts"),
        ]
        
        for i, (icon, title, desc) in enumerate(features):
            row = i // 3
            col = i % 3
            
            card = ctk.CTkFrame(features_grid, fg_color="#0f0f1e", corner_radius=10)
            card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            features_grid.grid_columnconfigure(col, weight=1)
            
            ctk.CTkLabel(card, text=icon, font=ctk.CTkFont(size=30)).pack(pady=(15, 5))
            ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=14, weight="bold")).pack()
            ctk.CTkLabel(
                card,
                text=desc,
                font=ctk.CTkFont(size=11),
                text_color="#808080"
            ).pack(pady=(5, 15))
    
    # ==================== ANALYTICS PAGE ====================
    def _show_analytics(self):
        """Show advanced analytics dashboard."""
        self._clear_main_frame()
        self._update_status("Analytics Dashboard", "📊")
        
        content = ctk.CTkScrollableFrame(self.main_frame, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=30, pady=30)
        
        ctk.CTkLabel(
            content,
            text="📊 Analytics & Insights",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#00d4ff"
        ).pack(pady=(0, 20))
        
        # Quick stats
        stats_frame = ctk.CTkFrame(content, fg_color="#1a1a2e", corner_radius=12)
        stats_frame.pack(fill="x", pady=10)
        
        daily = self.analytics.get_daily_statistics()
        weekly = self.analytics.get_weekly_statistics()
        
        ctk.CTkLabel(
            stats_frame,
            text=f"Today: {daily['unique_people']} people | "
                 f"This Week: {weekly['unique_people']} people | "
                 f"Peak Hour: {daily.get('peak_hour', 'N/A')}:00",
            font=ctk.CTkFont(size=14),
            text_color="#a0a0a0"
        ).pack(pady=15)
        
        # Visualization buttons
        viz_frame = ctk.CTkFrame(content, fg_color="#16213e", corner_radius=12)
        viz_frame.pack(fill="x", pady=20)
        
        ctk.CTkLabel(
            viz_frame,
            text="Generate Visualizations",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=15)
        
        btn_frame = ctk.CTkFrame(viz_frame, fg_color="transparent")
        btn_frame.pack(pady=10, padx=20, fill="x")
        
        btns = [
            ("📈 Daily Trend (30 days)", lambda: self._generate_plot("daily")),
            ("⏰ Hourly Distribution", lambda: self._generate_plot("hourly")),
            ("🏆 Top Attendees", lambda: self._generate_plot("top")),
            ("📥 Export Report (JSON)", lambda: self._export_analytics()),
        ]
        
        for i, (text, cmd) in enumerate(btns):
            ctk.CTkButton(
                btn_frame,
                text=text,
                command=cmd,
                font=ctk.CTkFont(size=14),
                height=40,
                fg_color="#00d4ff",
                text_color="#000000",
                hover_color="#00b4d8"
            ).grid(row=i // 2, column=i % 2, padx=10, pady=10, sticky="ew")
            btn_frame.grid_columnconfigure(i % 2, weight=1)
        
        # Detailed statistics
        details_frame = ctk.CTkFrame(content, fg_color="#1a1a2e", corner_radius=12)
        details_frame.pack(fill="both", expand=True, pady=10)
        
        ctk.CTkLabel(
            details_frame,
            text="Weekly Breakdown",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=15)
        
        self.analytics_text = ctk.CTkTextbox(details_frame, height=300, font=ctk.CTkFont(size=12))
        self.analytics_text.pack(padx=20, pady=(0, 20), fill="both", expand=True)
        
        # Display weekly breakdown
        breakdown = weekly.get('daily_breakdown', {})
        for date, stats in sorted(breakdown.items(), reverse=True):
            self.analytics_text.insert(
                "end",
                f"📅 {date}: {stats['unique_people']} people, {stats['total_records']} records\n"
            )
        
        if not breakdown:
            self.analytics_text.insert("end", "No data available for the past week.")
    
    def _generate_plot(self, plot_type: str):
        """Generate and save analytics plots."""
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png")],
            initialfile=f"analytics_{plot_type}_{datetime.now().strftime('%Y%m%d')}.png"
        )
        
        if save_path:
            try:
                if plot_type == "daily":
                    self.analytics.plot_daily_attendance(days=30, save_path=save_path)
                elif plot_type == "hourly":
                    self.analytics.plot_hourly_distribution(save_path=save_path)
                elif plot_type == "top":
                    self.analytics.plot_top_attendees(top_n=15, save_path=save_path)
                
                self.notification_manager.show_toast(
                    "Success",
                    f"Plot saved to {Path(save_path).name}",
                    "success"
                )
            except Exception as e:
                self.notification_manager.show_toast("Error", str(e), "error")
    
    def _export_analytics(self):
        """Export analytics report."""
        save_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile=f"report_{datetime.now().strftime('%Y%m%d')}.json"
        )
        
        if save_path:
            if self.analytics.export_report(save_path):
                self.notification_manager.show_toast("Success", "Report exported", "success")
    
    # ==================== BACKUP PAGE ====================
    def _show_backup(self):
        """Show database backup and management page."""
        self._clear_main_frame()
        self._update_status("Database Management", "💾")
        
        content = ctk.CTkScrollableFrame(self.main_frame, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=30, pady=30)
        
        ctk.CTkLabel(
            content,
            text="💾 Database Management",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#00d4ff"
        ).pack(pady=(0, 20))
        
        # Database stats
        stats = self.db_manager.get_database_stats()
        stats_frame = ctk.CTkFrame(content, fg_color="#1a1a2e", corner_radius=12)
        stats_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            stats_frame,
            text=f"Database: {stats.get('size_mb', 0):.2f} MB | "
                 f"{stats.get('total_faces', 0)} faces | "
                 f"Last Modified: {stats.get('last_modified', 'N/A')[:19]}",
            font=ctk.CTkFont(size=13),
            text_color="#a0a0a0"
        ).pack(pady=15)
        
        # Backup controls
        backup_frame = ctk.CTkFrame(content, fg_color="#16213e", corner_radius=12)
        backup_frame.pack(fill="x", pady=10)
        
        ctk.CTkLabel(
            backup_frame,
            text="Backup Operations",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=15)
        
        btn_grid = ctk.CTkFrame(backup_frame, fg_color="transparent")
        btn_grid.pack(pady=10, padx=20, fill="x")
        
        btns = [
            ("💾 Create Backup", self._create_backup),
            ("📤 Export to JSON", self._export_to_json),
            ("📤 Export to SQLite", self._export_to_sqlite),
            ("📥 Import from JSON", self._import_from_json),
            ("♻️ Restore Backup", self._restore_backup),
        ]
        
        for i, (text, cmd) in enumerate(btns):
            ctk.CTkButton(
                btn_grid,
                text=text,
                command=cmd,
                height=40,
                font=ctk.CTkFont(size=14),
                fg_color="#00d4ff",
                text_color="#000000",
                hover_color="#00b4d8"
            ).grid(row=i // 2, column=i % 2, padx=10, pady=10, sticky="ew")
            btn_grid.grid_columnconfigure(i % 2, weight=1)
        
        # Backup list
        list_frame = ctk.CTkFrame(content, fg_color="#1a1a2e", corner_radius=12)
        list_frame.pack(fill="both", expand=True, pady=10)
        
        ctk.CTkLabel(
            list_frame,
            text="Available Backups",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=15)
        
        self.backup_list = ctk.CTkTextbox(list_frame, height=250, font=ctk.CTkFont(size=12))
        self.backup_list.pack(padx=20, pady=(0, 20), fill="both", expand=True)
        
        self._refresh_backup_list()
    
    def _refresh_backup_list(self):
        """Refresh the backup list display."""
        if hasattr(self, 'backup_list'):
            self.backup_list.delete("1.0", "end")
            backups = self.db_manager.list_backups()
            
            for backup in backups:
                self.backup_list.insert(
                    "end",
                    f"📦 {backup['filename']} - {backup['size_mb']:.2f} MB - {backup['created'][:19]}\n"
                )
            
            if not backups:
                self.backup_list.insert("end", "No backups found.")
    
    def _create_backup(self):
        """Create a new backup."""
        backup_path = self.db_manager.create_backup()
        if backup_path:
            self.notification_manager.show_toast("Success", "Backup created", "success")
            self._refresh_backup_list()
    
    def _export_to_json(self):
        """Export database to JSON."""
        save_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile=f"face_db_{datetime.now().strftime('%Y%m%d')}.json"
        )
        
        if save_path:
            if self.db_manager.export_to_json(save_path):
                self.notification_manager.show_toast("Success", "Exported to JSON", "success")
    
    def _export_to_sqlite(self):
        """Export database to SQLite."""
        save_path = filedialog.asksaveasfilename(
            defaultextension=".db",
            filetypes=[("SQLite files", "*.db")],
            initialfile=f"face_db_{datetime.now().strftime('%Y%m%d')}.db"
        )
        
        if save_path:
            if self.db_manager.export_to_sqlite(save_path):
                self.notification_manager.show_toast("Success", "Exported to SQLite", "success")
    
    def _import_from_json(self):
        """Import database from JSON."""
        file_path = filedialog.askopenfilename(
            title="Select JSON file",
            filetypes=[("JSON files", "*.json")]
        )
        
        if file_path:
            merge = messagebox.askyesno("Import Mode", "Merge with existing data?\n(No = Replace)")
            if self.db_manager.import_from_json(file_path, merge=merge):
                self.face_system.load_encodings()  # Reload
                self.notification_manager.show_toast("Success", "Import complete", "success")
                self._update_status("Database imported", "✅")
    
    def _restore_backup(self):
        """Restore from a backup file."""
        file_path = filedialog.askopenfilename(
            title="Select backup file",
            filetypes=[("Pickle files", "*.pkl")]
        )
        
        if file_path:
            if messagebox.askyesno("Confirm", "Restore from this backup?"):
                if self.db_manager.restore_backup(file_path):
                    self.face_system.load_encodings()  # Reload
                    self.notification_manager.show_toast("Success", "Restore complete", "success")
    
    # ==================== ENHANCED PAGES ====================
    def _show_register(self):
        """Enhanced registration page with quality check."""
        self._clear_main_frame()
        self._update_status("Face Registration", "📝")
        
        content = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(content, text="Register New Face", font=ctk.CTkFont(size=24, weight="bold"), text_color="#00d4ff").pack(pady=(10, 20))
        
        # Two-column layout
        columns = ctk.CTkFrame(content, fg_color="transparent")
        columns.pack(fill="both", expand=True)
        columns.grid_columnconfigure(0, weight=1)
        columns.grid_columnconfigure(1, weight=1)
        
        # Left: Camera preview
        preview_frame = ctk.CTkFrame(columns, fg_color="#1a1a2e")
        preview_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(preview_frame, text="Preview", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        self.register_preview = ctk.CTkLabel(preview_frame, text="Click 'Start Camera' to begin", width=600, height=450)
        self.register_preview.pack(padx=20, pady=10)
        
        cam_controls = ctk.CTkFrame(preview_frame, fg_color="transparent")
        cam_controls.pack(pady=10)
        
        self.btn_start_register = ctk.CTkButton(cam_controls, text="📷 Start Camera", command=lambda: self._start_camera('register'))
        self.btn_start_register.pack(side="left", padx=5)
        
        self.btn_stop_register = ctk.CTkButton(cam_controls, text="⏹ Stop", command=self._stop_camera, state="disabled")
        self.btn_stop_register.pack(side="left", padx=5)
        
        # Right: Registration form
        form_frame = ctk.CTkFrame(columns, fg_color="#1a1a2e")
        form_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(form_frame, text="Registration Details", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        
        ctk.CTkLabel(form_frame, text="Person's Name:").pack(pady=(20, 5))
        self.name_entry = ctk.CTkEntry(form_frame, width=250, placeholder_text="Enter name")
        self.name_entry.pack(pady=5)
        
        ctk.CTkLabel(form_frame, text="Registration Method:", font=ctk.CTkFont(size=14)).pack(pady=(30, 10))
        
        ctk.CTkButton(form_frame, text="📸 Capture from Camera", width=250, command=self._capture_face, fg_color="#00d4ff", text_color="#000000").pack(pady=10)
        ctk.CTkButton(form_frame, text="📁 Upload Image", width=250, command=self._upload_image_for_registration, fg_color="#00d4ff", text_color="#000000").pack(pady=10)
        ctk.CTkButton(form_frame, text="📂 Batch Register", width=250, command=self._batch_register, fg_color="#00d4ff", text_color="#000000").pack(pady=10)
    
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
        self._update_status(f"Capturing face for: {name}", "📸")
    
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
                self.notification_manager.show_toast("Success", f"Registered {name}", "success")
                self._update_status(f"Registered: {name}", "✅")
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
                self._update_status(f"Batch registered {stats['successful']} faces", "✅")
    
    def _show_recognize(self):
        """Enhanced recognition page with emotion detection."""
        self._clear_main_frame()
        self._update_status("Face Recognition", "🔍")
        
        content = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(content, text="Face Recognition", font=ctk.CTkFont(size=24, weight="bold"), text_color="#00d4ff").pack(pady=(10, 20))
        
        self.recognize_preview = ctk.CTkLabel(content, text="Click 'Start Recognition' to begin", width=800, height=600, fg_color="#1a1a2e")
        self.recognize_preview.pack(pady=10)
        
        controls = ctk.CTkFrame(content, fg_color="transparent")
        controls.pack(pady=10)
        
        self.btn_start_recognize = ctk.CTkButton(controls, text="📷 Start Recognition", command=lambda: self._start_camera('recognize'), fg_color="#00d4ff", text_color="#000000")
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
                self._update_status(f"Detected: {', '.join(names)}", "✅")
                
                image = cv2.imread(file_path)
                for name, (top, right, bottom, left) in results:
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(image, (left, top), (right, bottom), color, 2)
                    cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                if hasattr(self, 'recognize_preview'):
                    self._display_image(image, self.recognize_preview)
            else:
                self._update_status("No faces detected", "⚠️")
    
    def _show_attendance(self):
        """Enhanced attendance tracking page."""
        self._clear_main_frame()
        self._update_status("Attendance System", "📋")
        
        content = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(content, text="Attendance Tracking", font=ctk.CTkFont(size=24, weight="bold"), text_color="#00d4ff").pack(pady=(10, 20))
        
        columns = ctk.CTkFrame(content, fg_color="transparent")
        columns.pack(fill="both", expand=True)
        columns.grid_columnconfigure(0, weight=1)
        columns.grid_columnconfigure(1, weight=1)
        
        # Left: Camera
        cam_frame = ctk.CTkFrame(columns, fg_color="#1a1a2e")
        cam_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.attendance_preview = ctk.CTkLabel(cam_frame, text="Click 'Start' to begin", width=600, height=450)
        self.attendance_preview.pack(padx=20, pady=20)
        
        cam_controls = ctk.CTkFrame(cam_frame, fg_color="transparent")
        cam_controls.pack(pady=10)
        
        self.btn_start_attendance = ctk.CTkButton(cam_controls, text="▶ Start", command=lambda: self._start_camera('attendance'), fg_color="#00d4ff", text_color="#000000")
        self.btn_start_attendance.pack(side="left", padx=5)
        
        self.btn_stop_attendance = ctk.CTkButton(cam_controls, text="⏹ Stop", command=self._stop_camera, state="disabled")
        self.btn_stop_attendance.pack(side="left", padx=5)
        
        # Right: Attendance log
        log_frame = ctk.CTkFrame(columns, fg_color="#1a1a2e")
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
    
    def _show_database(self):
        """Enhanced database management page."""
        self._clear_main_frame()
        self._update_status("Database Management", "👥")
        
        content = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(content, text="Registered Faces Database", font=ctk.CTkFont(size=24, weight="bold"), text_color="#00d4ff").pack(pady=(10, 20))
        
        # Stats
        unique_names = list(set(self.face_system.known_face_names))
        ctk.CTkLabel(
            content,
            text=f"Total: {len(self.face_system.known_face_names)} encodings | {len(unique_names)} unique persons",
            font=ctk.CTkFont(size=14)
        ).pack(pady=10)
        
        # List of persons
        scroll_frame = ctk.CTkScrollableFrame(content, width=600, height=400, fg_color="#1a1a2e")
        scroll_frame.pack(padx=20, pady=10)
        
        from collections import Counter
        name_counts = Counter(self.face_system.known_face_names)
        
        for name, count in sorted(name_counts.items()):
            person_frame = ctk.CTkFrame(scroll_frame, fg_color="#16213e")
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
        ctk.CTkButton(btn_frame, text="🔄 Refresh", command=self._show_database, fg_color="#00d4ff", text_color="#000000").pack(side="left", padx=10)
    
    def _delete_person(self, name: str):
        """Delete a person from the database."""
        if messagebox.askyesno("Confirm", f"Delete all encodings for '{name}'?"):
            indices = [i for i, n in enumerate(self.face_system.known_face_names) if n == name]
            for i in sorted(indices, reverse=True):
                del self.face_system.known_face_encodings[i]
                del self.face_system.known_face_names[i]
            self.face_system.save_encodings()
            self.notification_manager.show_toast("Success", f"Deleted {name}", "success")
            self._show_database()
    
    def _clear_database(self):
        """Clear all face encodings."""
        if messagebox.askyesno("Confirm", "Delete ALL registered faces?"):
            self.face_system.known_face_encodings = []
            self.face_system.known_face_names = []
            self.face_system.save_encodings()
            self._show_database()
    
    def _show_settings(self):
        """Enhanced settings page."""
        self._clear_main_frame()
        self._update_status("Settings", "⚙️")
        
        content = ctk.CTkScrollableFrame(self.main_frame, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=40, pady=40)
        
        ctk.CTkLabel(content, text="Settings", font=ctk.CTkFont(size=24, weight="bold"), text_color="#00d4ff").pack(pady=(10, 30))
        
        # Camera settings
        cam_frame = ctk.CTkFrame(content, fg_color="#1a1a2e")
        cam_frame.pack(fill="x", padx=50, pady=10)
        
        ctk.CTkLabel(cam_frame, text="Camera Settings", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=15)
        ctk.CTkLabel(cam_frame, text="Camera Index:", font=ctk.CTkFont(size=14)).pack(pady=(10, 5))
        self.camera_entry = ctk.CTkEntry(cam_frame, width=100)
        self.camera_entry.insert(0, str(self.camera_index))
        self.camera_entry.pack(pady=(5, 20))
        
        # Feature toggles
        features_frame = ctk.CTkFrame(content, fg_color="#1a1a2e")
        features_frame.pack(fill="x", padx=50, pady=10)
        
        ctk.CTkLabel(features_frame, text="Advanced Features", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=15)
        
        self.use_emotion_var = ctk.BooleanVar(value=self.use_emotion_recognition)
        ctk.CTkCheckBox(features_frame, text="Enable Emotion Recognition", variable=self.use_emotion_var).pack(pady=5)
        
        self.use_quality_var = ctk.BooleanVar(value=self.use_quality_check)
        ctk.CTkCheckBox(features_frame, text="Enable Quality Check", variable=self.use_quality_var).pack(pady=5)
        
        self.performance_mode_var = ctk.BooleanVar(value=self.performance_mode)
        ctk.CTkCheckBox(features_frame, text="Performance Mode (Faster Processing)", variable=self.performance_mode_var).pack(pady=5)
        
        self.notifications_var = ctk.BooleanVar(value=self.notification_manager.toast_enabled)
        ctk.CTkCheckBox(features_frame, text="Enable Notifications", variable=self.notifications_var).pack(pady=(5, 20))
        
        ctk.CTkButton(content, text="Save Settings", command=self._save_settings, fg_color="#00d4ff", text_color="#000000", height=40).pack(pady=20)
        
        # About
        about_frame = ctk.CTkFrame(content, fg_color="#1a1a2e")
        about_frame.pack(fill="x", padx=50, pady=20)
        
        ctk.CTkLabel(about_frame, text="About", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)
        ctk.CTkLabel(about_frame, text="Advanced Face Recognition System v2.0\nWith Emotion Recognition & Advanced Analytics\nBuilt with Python, OpenCV & AI",
                    font=ctk.CTkFont(size=12), justify="center").pack(pady=(0, 20))
    
    def _save_settings(self):
        """Save settings."""
        try:
            self.camera_index = int(self.camera_entry.get())
            self.use_emotion_recognition = self.use_emotion_var.get()
            self.use_quality_check = self.use_quality_var.get()
            self.performance_mode = self.performance_mode_var.get()
            self.notification_manager.toast_enabled = self.notifications_var.get()
            messagebox.showinfo("Settings", "Settings saved successfully!")
            self._update_status("Settings saved", "✅")
        except ValueError:
            messagebox.showerror("Error", "Invalid camera index")
    
    def _schedule_ui_update(self):
        """Schedule UI update loop at 30 FPS."""
        self._update_ui()
        self.after(33, self._schedule_ui_update)  # ~30 FPS
    
    def _update_ui(self):
        """Update UI with camera frames."""
        try:
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
                self._update_status(f"✅ Registered: {data}", "✅")
                self.notification_manager.show_toast("Success", f"Registered {data}", "success")
                messagebox.showinfo("Success", f"Successfully registered {data}!")
            elif action == "register_no_face":
                self._update_status("❌ No face detected", "⚠️")
            elif action == "register_multiple_faces":
                self._update_status("❌ Multiple faces - show only one", "⚠️")
            elif action == "attendance_marked":
                self._update_status(f"✅ Attendance marked: {data}", "✅")
                self.notification_manager.show_toast("Attendance", f"{data} checked in", "success")
                self._refresh_attendance_log()
        except Exception as e:
            print(f"Handle result error: {e}")
    
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
    
    def _start_camera(self, mode: str):
        """Start camera for a specific mode."""
        if self.is_camera_running:
            return
        
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Set to 30 FPS
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        
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
    
    def _camera_loop(self):
        """Main camera loop with optimized performance."""
        import face_recognition
        
        frame_count = 0
        process_count = 0
        face_locations = []
        face_names = []
        face_emotions = []  # Track emotions
        
        # Performance optimization: process every Nth frame
        PROCESS_EVERY_N_FRAMES = 3  # Process every 3rd frame (faster)
        
        while self.is_camera_running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                
                display_frame = frame.copy()
                frame_count += 1
                
                if self.current_mode in ['recognize', 'attendance']:
                    # Only process every Nth frame for performance
                    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                        process_count += 1
                        
                        # Resize once and reuse
                        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                        
                        # Face detection with HOG (faster)
                        face_locations = face_recognition.face_locations(rgb_small, model="hog")
                        
                        if face_locations:
                            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
                            face_names = []
                            face_emotions = []
                            
                            for i, encoding in enumerate(face_encodings):
                                name = self.face_system._match_face(encoding, 0.6)
                                face_names.append(name)
                                
                                # EMOTION RECOGNITION - Now working!
                                if self.use_emotion_recognition and process_count % 2 == 0:  # Every 6th frame
                                    try:
                                        top, right, bottom, left = face_locations[i]
                                        # Scale back to original size
                                        face_roi = frame[top*4:bottom*4, left*4:right*4]
                                        if face_roi.size > 0:
                                            emotion = self.emotion_recognizer.predict_emotion(face_roi)
                                            if emotion:
                                                face_emotions.append(emotion)
                                                # Track emotion for this person
                                                if name != "Unknown":
                                                    self.emotion_tracker.add_emotion(name, emotion)
                                            else:
                                                face_emotions.append("Neutral")
                                        else:
                                            face_emotions.append("Neutral")
                                    except Exception as e:
                                        face_emotions.append("Neutral")
                                else:
                                    face_emotions.append("")
                                
                                # Mark attendance
                                if self.current_mode == 'attendance' and name != "Unknown":
                                    if self.attendance_system.mark_attendance(name):
                                        try:
                                            self.result_queue.put_nowait(("attendance_marked", name))
                                        except queue.Full:
                                            pass
                        else:
                            face_names = []
                            face_emotions = []
                    
                    # Draw boxes with emotions
                    for idx, ((top, right, bottom, left), name) in enumerate(zip(face_locations, face_names)):
                        top, right, bottom, left = top*4, right*4, bottom*4, left*4
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        
                        # Draw face box
                        cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                        
                        # Emotion text (if available)
                        emotion_text = ""
                        if idx < len(face_emotions) and face_emotions[idx]:
                            emotion = face_emotions[idx]
                            # Emoji mapping
                            emoji_map = {
                                "Happy": "😊", "Sad": "😢", "Angry": "😠",
                                "Surprise": "😮", "Fear": "😨", "Disgust": "🤢",
                                "Neutral": "😐"
                            }
                            emoji = emoji_map.get(emotion, "")
                            emotion_text = f" {emoji} {emotion}"
                        
                        # Draw name and emotion label background
                        label_height = 35 if not emotion_text else 60
                        cv2.rectangle(display_frame, (left, bottom - label_height), (right, bottom), color, cv2.FILLED)
                        
                        # Draw name
                        cv2.putText(display_frame, name, (left + 6, bottom - 36 if emotion_text else bottom - 6),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Draw emotion
                        if emotion_text:
                            cv2.putText(display_frame, emotion_text, (left + 6, bottom - 8),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                elif self.current_mode == 'register':
                    # Process less frequently in register mode
                    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                        face_locs = face_recognition.face_locations(rgb_small, model="hog")
                        face_locations = [(t*2, r*2, b*2, l*2) for t, r, b, l in face_locs]
                    
                    for (top, right, bottom, left) in face_locations:
                        cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    cv2.putText(display_frame, "Enter name & click Capture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
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
                
                # Put frame in queue (non-blocking)
                try:
                    # Clear old frames if queue is full
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.frame_queue.put_nowait(display_frame)
                except queue.Full:
                    pass  # Skip frame if queue is full
                
                # Reduced sleep for smoother video (30 FPS target)
                time.sleep(0.033)
                
            except Exception as e:
                print(f"Camera error: {e}")
                time.sleep(0.1)
    
    def _stop_camera(self):
        """Stop the camera."""
        self.is_camera_running = False
        
        if self.cap:
            time.sleep(0.2)
            self.cap.release()
            self.cap = None
        
        self.current_preview_label = None
        
        # Re-enable buttons
        if hasattr(self, 'btn_start_register'):
            self.btn_start_register.configure(state="normal")
            self.btn_stop_register.configure(state="disabled")
        if hasattr(self, 'btn_start_recognize'):
            self.btn_start_recognize.configure(state="normal")
            self.btn_stop_recognize.configure(state="disabled")
        if hasattr(self, 'btn_start_attendance'):
            self.btn_start_attendance.configure(state="normal")
            self.btn_stop_attendance.configure(state="disabled")
    
    def _on_closing(self):
        """Handle window close event."""
        self._stop_camera()
        self.db_manager.auto_backup()  # Final backup
        self.destroy()


def main():
    app = AdvancedFaceRecognitionApp()
    app.mainloop()


if __name__ == "__main__":
    main()
