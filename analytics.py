"""
Analytics and Reporting Module
================================
Generates comprehensive analytics and visualizations for attendance and recognition data.
"""

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class AnalyticsDashboard:
    """Generate analytics and visualizations for the face recognition system."""
    
    def __init__(self, attendance_file: str = "attendance.csv"):
        """
        Initialize analytics dashboard.
        
        Args:
            attendance_file: Path to attendance CSV file
        """
        self.attendance_file = Path(attendance_file)
        self.attendance_data = []
        self.emotion_data = {}
        self._load_data()
    
    def _load_data(self):
        """Load attendance data from CSV."""
        if not self.attendance_file.exists():
            logger.warning("Attendance file not found")
            return
        
        try:
            with open(self.attendance_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.attendance_data = list(reader)
            logger.info(f"Loaded {len(self.attendance_data)} attendance records")
        except Exception as e:
            logger.error(f"Error loading attendance data: {e}")
    
    def get_daily_statistics(self, date: Optional[str] = None) -> Dict:
        """
        Get attendance statistics for a specific date.
        
        Args:
            date: Date string (YYYY-MM-DD). If None, uses today.
            
        Returns:
            Dictionary with daily statistics
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        daily_records = [r for r in self.attendance_data if r.get('Date') == date]
        
        unique_people = set(r['Name'] for r in daily_records)
        
        # Time distribution
        time_counts = defaultdict(int)
        for record in daily_records:
            time_str = record.get('Time', '')
            if time_str:
                hour = time_str.split(':')[0]
                time_counts[hour] += 1
        
        # Status distribution
        status_counts = Counter(r.get('Status', 'Present') for r in daily_records)
        
        return {
            'date': date,
            'total_records': len(daily_records),
            'unique_people': len(unique_people),
            'people_list': list(unique_people),
            'time_distribution': dict(time_counts),
            'status_distribution': dict(status_counts),
            'peak_hour': max(time_counts, key=time_counts.get) if time_counts else None
        }
    
    def get_weekly_statistics(self) -> Dict:
        """
        Get attendance statistics for the past week.
        
        Returns:
            Dictionary with weekly statistics
        """
        today = datetime.now()
        week_ago = today - timedelta(days=7)
        
        weekly_records = []
        for record in self.attendance_data:
            record_date = datetime.strptime(record.get('Date', ''), "%Y-%m-%d")
            if week_ago <= record_date <= today:
                weekly_records.append(record)
        
        # Daily breakdown
        daily_breakdown = defaultdict(lambda: {'count': 0, 'people': set()})
        for record in weekly_records:
            date = record.get('Date')
            daily_breakdown[date]['count'] += 1
            daily_breakdown[date]['people'].add(record['Name'])
        
        # Convert sets to counts
        daily_stats = {
            date: {
                'total_records': stats['count'],
                'unique_people': len(stats['people'])
            }
            for date, stats in daily_breakdown.items()
        }
        
        return {
            'total_records': len(weekly_records),
            'unique_people': len(set(r['Name'] for r in weekly_records)),
            'daily_breakdown': daily_stats,
            'average_daily': len(weekly_records) / 7
        }
    
    def get_person_statistics(self, person_name: str) -> Dict:
        """
        Get statistics for a specific person.
        
        Args:
            person_name: Name of the person
            
        Returns:
            Dictionary with person-specific statistics
        """
        person_records = [r for r in self.attendance_data if r['Name'] == person_name]
        
        if not person_records:
            return {
                'name': person_name,
                'total_attendance': 0,
                'dates': [],
                'average_time': None
            }
        
        dates = [r.get('Date') for r in person_records]
        times = [r.get('Time') for r in person_records]
        
        # Calculate average attendance time
        time_minutes = []
        for time_str in times:
            if time_str:
                parts = time_str.split(':')
                if len(parts) >= 2:
                    hours, minutes = int(parts[0]), int(parts[1])
                    time_minutes.append(hours * 60 + minutes)
        
        avg_minutes = int(np.mean(time_minutes)) if time_minutes else 0
        avg_time = f"{avg_minutes // 60:02d}:{avg_minutes % 60:02d}"
        
        # Attendance pattern (day of week)
        day_counts = defaultdict(int)
        for date_str in dates:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            day_name = date_obj.strftime("%A")
            day_counts[day_name] += 1
        
        return {
            'name': person_name,
            'total_attendance': len(person_records),
            'dates': dates,
            'average_time': avg_time,
            'day_distribution': dict(day_counts),
            'first_attendance': min(dates) if dates else None,
            'last_attendance': max(dates) if dates else None
        }
    
    def plot_daily_attendance(self, days: int = 30, save_path: Optional[str] = None):
        """
        Plot daily attendance trend.
        
        Args:
            days: Number of days to include
            save_path: Path to save the plot (optional)
        """
        today = datetime.now()
        start_date = today - timedelta(days=days)
        
        # Collect daily counts
        daily_counts = defaultdict(int)
        for record in self.attendance_data:
            record_date = datetime.strptime(record.get('Date', ''), "%Y-%m-%d")
            if start_date <= record_date <= today:
                daily_counts[record['Date']] += 1
        
        # Prepare data for plotting
        dates = sorted(daily_counts.keys())
        counts = [daily_counts[date] for date in dates]
        date_objects = [datetime.strptime(d, "%Y-%m-%d") for d in dates]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(date_objects, counts, marker='o', linewidth=2, markersize=6)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Attendance Count', fontsize=12)
        plt.title(f'Daily Attendance Trend (Last {days} Days)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=max(1, days // 10)))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_hourly_distribution(self, date: Optional[str] = None, save_path: Optional[str] = None):
        """
        Plot hourly attendance distribution.
        
        Args:
            date: Specific date (YYYY-MM-DD). If None, uses all data.
            save_path: Path to save the plot (optional)
        """
        if date:
            records = [r for r in self.attendance_data if r.get('Date') == date]
            title = f'Hourly Attendance Distribution - {date}'
        else:
            records = self.attendance_data
            title = 'Overall Hourly Attendance Distribution'
        
        # Count by hour
        hour_counts = defaultdict(int)
        for record in records:
            time_str = record.get('Time', '')
            if time_str:
                hour = int(time_str.split(':')[0])
                hour_counts[hour] += 1
        
        # Prepare data
        hours = list(range(24))
        counts = [hour_counts.get(h, 0) for h in hours]
        
        # Create plot
        plt.figure(figsize=(14, 6))
        plt.bar(hours, counts, color='skyblue', edgecolor='navy', alpha=0.7)
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Attendance Count', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(hours, [f'{h:02d}:00' for h in hours], rotation=45)
        plt.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_top_attendees(self, top_n: int = 10, save_path: Optional[str] = None):
        """
        Plot top attendees by frequency.
        
        Args:
            top_n: Number of top attendees to show
            save_path: Path to save the plot (optional)
        """
        # Count attendance per person
        person_counts = Counter(r['Name'] for r in self.attendance_data)
        
        # Get top N
        top_people = person_counts.most_common(top_n)
        names = [name for name, _ in top_people]
        counts = [count for _, count in top_people]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        plt.barh(names, counts, color='coral', edgecolor='darkred', alpha=0.7)
        plt.xlabel('Attendance Count', fontsize=12)
        plt.ylabel('Person', fontsize=12)
        plt.title(f'Top {top_n} Attendees', fontsize=14, fontweight='bold')
        plt.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def export_report(self, output_file: str = "attendance_report.json"):
        """
        Export comprehensive analytics report.
        
        Args:
            output_file: Path to output JSON file
        """
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_records': len(self.attendance_data),
            'unique_people': len(set(r['Name'] for r in self.attendance_data)),
            'daily_stats': self.get_daily_statistics(),
            'weekly_stats': self.get_weekly_statistics(),
            'date_range': {
                'start': min((r.get('Date', '') for r in self.attendance_data), default='N/A'),
                'end': max((r.get('Date', '') for r in self.attendance_data), default='N/A')
            }
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Report exported to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            return False
