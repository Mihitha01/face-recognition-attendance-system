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
    
    # Implement remaining pages from original ui_app.py...
    # (Register, Recognize, Attendance, Database, Settings pages with enhancements)
    
    def _show_register(self):
        """Enhanced registration page."""
        # TODO: Implement with quality check and liveness detection
        pass
    
    def _show_recognize(self):
        """Enhanced recognition page."""
        # TODO: Implement with emotion recognition
        pass
    
    def _show_attendance(self):
        """Enhanced attendance page."""
        # TODO: Implement with notifications
        pass
    
    def _show_database(self):
        """Enhanced database page."""
        # TODO: Implement from original with export options
        pass
    
    def _show_settings(self):
        """Enhanced settings page."""
        # TODO: Implement with all new feature toggles
        pass
    
    def _schedule_ui_update(self):
        """Schedule UI update loop."""
        self._update_ui()
        self.after(33, self._schedule_ui_update)
    
    def _update_ui(self):
        """Update UI with camera frames."""
        # Similar to original implementation
        pass
    
    def _start_camera(self, mode: str):
        """Start camera with advanced features."""
        # Enhanced version with new detectors
        pass
    
    def _stop_camera(self):
        """Stop camera."""
        self.is_camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
    
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
