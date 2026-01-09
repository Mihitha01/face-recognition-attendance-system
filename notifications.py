"""
Notification System Module
===========================
Handles various notification types including toast notifications,
email alerts, and sound notifications.
"""

import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Optional, Callable
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


class NotificationManager:
    """Manage various types of notifications."""
    
    def __init__(self):
        """Initialize notification manager."""
        self.sound_enabled = True
        self.toast_enabled = True
        self.email_enabled = False
        self.email_config = {}
        
        # Toast notification callback (to be set by UI)
        self.toast_callback: Optional[Callable] = None
    
    def set_toast_callback(self, callback: Callable):
        """
        Set callback function for toast notifications.
        
        Args:
            callback: Function to call with (title, message, type)
        """
        self.toast_callback = callback
    
    def show_toast(self, title: str, message: str, notification_type: str = "info"):
        """
        Show a toast notification.
        
        Args:
            title: Notification title
            message: Notification message
            notification_type: Type ('info', 'success', 'warning', 'error')
        """
        if not self.toast_enabled:
            return
        
        logger.info(f"Toast: [{notification_type.upper()}] {title} - {message}")
        
        # Call UI callback if set
        if self.toast_callback:
            try:
                self.toast_callback(title, message, notification_type)
            except Exception as e:
                logger.error(f"Toast callback error: {e}")
    
    def play_sound(self, sound_type: str = "success"):
        """
        Play a notification sound.
        
        Args:
            sound_type: Type of sound ('success', 'error', 'warning', 'info')
        """
        if not self.sound_enabled:
            return
        
        try:
            # On Windows, use winsound
            import winsound
            
            sound_map = {
                'success': (1000, 100),  # Frequency, Duration
                'error': (500, 200),
                'warning': (750, 150),
                'info': (800, 100)
            }
            
            freq, duration = sound_map.get(sound_type, (800, 100))
            
            # Play in a separate thread to not block
            threading.Thread(
                target=lambda: winsound.Beep(freq, duration),
                daemon=True
            ).start()
        except ImportError:
            # Not on Windows or winsound not available
            logger.debug("Sound notification not available on this platform")
        except Exception as e:
            logger.error(f"Sound playback error: {e}")
    
    def configure_email(self, smtp_server: str, smtp_port: int,
                       sender_email: str, sender_password: str):
        """
        Configure email notification settings.
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            sender_email: Sender email address
            sender_password: Sender email password
        """
        self.email_config = {
            'smtp_server': smtp_server,
            'smtp_port': smtp_port,
            'sender_email': sender_email,
            'sender_password': sender_password
        }
        self.email_enabled = True
        logger.info("Email notifications configured")
    
    def send_email(self, recipient: str, subject: str, body: str,
                   html: bool = False) -> bool:
        """
        Send an email notification.
        
        Args:
            recipient: Recipient email address
            subject: Email subject
            body: Email body
            html: Whether body is HTML
            
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.email_enabled or not self.email_config:
            logger.warning("Email notifications not configured")
            return False
        
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = self.email_config['sender_email']
            msg['To'] = recipient
            msg['Subject'] = subject
            
            mime_type = 'html' if html else 'plain'
            msg.attach(MIMEText(body, mime_type))
            
            # Send email in a separate thread
            threading.Thread(
                target=self._send_email_async,
                args=(msg, recipient),
                daemon=True
            ).start()
            
            return True
        except Exception as e:
            logger.error(f"Email preparation error: {e}")
            return False
    
    def _send_email_async(self, msg: MIMEMultipart, recipient: str):
        """Send email asynchronously."""
        try:
            with smtplib.SMTP(
                self.email_config['smtp_server'],
                self.email_config['smtp_port']
            ) as server:
                server.starttls()
                server.login(
                    self.email_config['sender_email'],
                    self.email_config['sender_password']
                )
                server.send_message(msg)
            
            logger.info(f"Email sent to {recipient}")
        except Exception as e:
            logger.error(f"Email sending error: {e}")
    
    def notify_attendance_marked(self, person_name: str, time: str):
        """
        Send notification when attendance is marked.
        
        Args:
            person_name: Name of the person
            time: Time of attendance
        """
        self.show_toast(
            "Attendance Marked",
            f"{person_name} - {time}",
            "success"
        )
        self.play_sound("success")
    
    def notify_face_registered(self, person_name: str):
        """
        Send notification when a new face is registered.
        
        Args:
            person_name: Name of the person
        """
        self.show_toast(
            "Face Registered",
            f"Successfully registered {person_name}",
            "success"
        )
        self.play_sound("success")
    
    def notify_unknown_face(self):
        """Send notification when an unknown face is detected."""
        self.show_toast(
            "Unknown Face",
            "Unrecognized person detected",
            "warning"
        )
        self.play_sound("warning")
    
    def notify_error(self, error_message: str):
        """
        Send error notification.
        
        Args:
            error_message: Error description
        """
        self.show_toast(
            "Error",
            error_message,
            "error"
        )
        self.play_sound("error")
    
    def notify_liveness_failed(self):
        """Send notification when liveness check fails."""
        self.show_toast(
            "Liveness Check Failed",
            "Please ensure you are a real person",
            "warning"
        )
        self.play_sound("error")
    
    def notify_quality_warning(self, issue: str):
        """
        Send notification for image quality issues.
        
        Args:
            issue: Description of the quality issue
        """
        self.show_toast(
            "Quality Warning",
            issue,
            "warning"
        )
    
    def send_daily_report(self, recipient: str, stats: dict):
        """
        Send daily attendance report via email.
        
        Args:
            recipient: Email recipient
            stats: Dictionary with daily statistics
        """
        if not self.email_enabled:
            return
        
        subject = f"Daily Attendance Report - {datetime.now().strftime('%Y-%m-%d')}"
        
        body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .header {{ background-color: #4CAF50; color: white; padding: 10px; }}
                .content {{ padding: 20px; }}
                .stats {{ background-color: #f0f0f0; padding: 15px; margin: 10px 0; }}
                .footer {{ color: #888; font-size: 12px; padding: 10px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>Daily Attendance Report</h2>
            </div>
            <div class="content">
                <p><strong>Date:</strong> {stats.get('date', 'N/A')}</p>
                <div class="stats">
                    <p><strong>Total Records:</strong> {stats.get('total_records', 0)}</p>
                    <p><strong>Unique People:</strong> {stats.get('unique_people', 0)}</p>
                    <p><strong>Peak Hour:</strong> {stats.get('peak_hour', 'N/A')}</p>
                </div>
                <p><strong>People Present:</strong></p>
                <ul>
                    {''.join([f"<li>{name}</li>" for name in stats.get('people_list', [])])}
                </ul>
            </div>
            <div class="footer">
                <p>Generated by Face Recognition Attendance System</p>
            </div>
        </body>
        </html>
        """
        
        self.send_email(recipient, subject, body, html=True)


class ToastNotification:
    """Simple toast notification implementation for desktop."""
    
    @staticmethod
    def show(title: str, message: str, notification_type: str = "info"):
        """
        Show a desktop toast notification (Windows 10+).
        
        Args:
            title: Notification title
            message: Notification message
            notification_type: Type of notification
        """
        try:
            from win10toast import ToastNotifier
            
            toaster = ToastNotifier()
            
            # Map notification type to icon
            icon_map = {
                'success': None,  # Default icon
                'error': None,
                'warning': None,
                'info': None
            }
            
            # Show toast (runs in background)
            threading.Thread(
                target=lambda: toaster.show_toast(
                    title,
                    message,
                    duration=5,
                    threaded=True
                ),
                daemon=True
            ).start()
        except ImportError:
            logger.debug("win10toast not available - using fallback notification")
        except Exception as e:
            logger.error(f"Toast notification error: {e}")
