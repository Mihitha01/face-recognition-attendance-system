"""
Database Management and Backup Module
======================================
Handles database export, import, backup, and synchronization.
"""

import os
import json
import sqlite3
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manage face recognition database with advanced backup and export features."""
    
    def __init__(self, encodings_file: str = "face_encodings.pkl"):
        """
        Initialize database manager.
        
        Args:
            encodings_file: Path to the pickle file storing face encodings
        """
        self.encodings_file = Path(encodings_file)
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)
    
    def create_backup(self, include_timestamp: bool = True) -> Optional[str]:
        """
        Create a backup of the face encodings database.
        
        Args:
            include_timestamp: Whether to include timestamp in backup filename
            
        Returns:
            Path to backup file or None if failed
        """
        if not self.encodings_file.exists():
            logger.warning("No database file to backup")
            return None
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"face_encodings_backup_{timestamp}.pkl" if include_timestamp else "face_encodings_backup.pkl"
            backup_path = self.backup_dir / backup_name
            
            shutil.copy2(self.encodings_file, backup_path)
            logger.info(f"Backup created: {backup_path}")
            return str(backup_path)
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return None
    
    def auto_backup(self, max_backups: int = 10):
        """
        Automatically manage backups, keeping only the most recent ones.
        
        Args:
            max_backups: Maximum number of backups to keep
        """
        # Create new backup
        self.create_backup()
        
        # List all backups
        backups = sorted(self.backup_dir.glob("face_encodings_backup_*.pkl"))
        
        # Remove old backups if exceeding max
        while len(backups) > max_backups:
            oldest = backups.pop(0)
            oldest.unlink()
            logger.info(f"Removed old backup: {oldest}")
    
    def restore_backup(self, backup_path: str) -> bool:
        """
        Restore database from a backup file.
        
        Args:
            backup_path: Path to the backup file
            
        Returns:
            True if restore successful, False otherwise
        """
        backup_file = Path(backup_path)
        
        if not backup_file.exists():
            logger.error(f"Backup file not found: {backup_path}")
            return False
        
        try:
            # Create backup of current database first
            if self.encodings_file.exists():
                current_backup = self.encodings_file.with_suffix('.pkl.before_restore')
                shutil.copy2(self.encodings_file, current_backup)
            
            # Restore from backup
            shutil.copy2(backup_file, self.encodings_file)
            logger.info(f"Database restored from: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False
    
    def export_to_json(self, output_file: str = "face_database.json") -> bool:
        """
        Export face database to JSON format.
        
        Args:
            output_file: Path to output JSON file
            
        Returns:
            True if export successful, False otherwise
        """
        if not self.encodings_file.exists():
            logger.error("No database file to export")
            return False
        
        try:
            with open(self.encodings_file, 'rb') as f:
                data = pickle.load(f)
            
            # Convert numpy arrays to lists for JSON serialization
            export_data = {
                'version': data.get('version', '1.0'),
                'saved_at': data.get('saved_at', datetime.now().isoformat()),
                'names': data.get('names', []),
                'encodings': [encoding.tolist() for encoding in data.get('encodings', [])],
                'total_faces': len(data.get('names', [])),
                'unique_persons': len(set(data.get('names', [])))
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Database exported to JSON: {output_file}")
            return True
        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            return False
    
    def import_from_json(self, json_file: str, merge: bool = False) -> bool:
        """
        Import face database from JSON format.
        
        Args:
            json_file: Path to JSON file
            merge: If True, merge with existing data; if False, replace
            
        Returns:
            True if import successful, False otherwise
        """
        json_path = Path(json_file)
        
        if not json_path.exists():
            logger.error(f"JSON file not found: {json_file}")
            return False
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Convert lists back to numpy arrays
            import numpy as np
            encodings = [np.array(enc) for enc in import_data.get('encodings', [])]
            names = import_data.get('names', [])
            
            if merge and self.encodings_file.exists():
                # Load existing data
                with open(self.encodings_file, 'rb') as f:
                    existing_data = pickle.load(f)
                
                # Merge
                encodings = existing_data.get('encodings', []) + encodings
                names = existing_data.get('names', []) + names
            
            # Save merged/new data
            data = {
                'encodings': encodings,
                'names': names,
                'version': import_data.get('version', '1.0'),
                'saved_at': datetime.now().isoformat()
            }
            
            # Create backup before import
            self.create_backup()
            
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Database imported from JSON: {json_file}")
            logger.info(f"Total faces: {len(names)}, Merge mode: {merge}")
            return True
        except Exception as e:
            logger.error(f"JSON import failed: {e}")
            return False
    
    def export_to_sqlite(self, db_file: str = "face_database.db") -> bool:
        """
        Export face database to SQLite format.
        
        Args:
            db_file: Path to SQLite database file
            
        Returns:
            True if export successful, False otherwise
        """
        if not self.encodings_file.exists():
            logger.error("No database file to export")
            return False
        
        try:
            with open(self.encodings_file, 'rb') as f:
                data = pickle.load(f)
            
            # Create SQLite connection
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    created_at TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_encodings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    encoding BLOB,
                    created_at TEXT,
                    FOREIGN KEY (person_id) REFERENCES persons (id)
                )
            ''')
            
            # Insert data
            names = data.get('names', [])
            encodings = data.get('encodings', [])
            
            person_ids = {}
            for i, (name, encoding) in enumerate(zip(names, encodings)):
                # Get or create person
                if name not in person_ids:
                    cursor.execute(
                        'INSERT INTO persons (name, created_at) VALUES (?, ?)',
                        (name, datetime.now().isoformat())
                    )
                    person_ids[name] = cursor.lastrowid
                
                # Insert encoding
                encoding_blob = pickle.dumps(encoding)
                cursor.execute(
                    'INSERT INTO face_encodings (person_id, encoding, created_at) VALUES (?, ?, ?)',
                    (person_ids[name], encoding_blob, datetime.now().isoformat())
                )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Database exported to SQLite: {db_file}")
            return True
        except Exception as e:
            logger.error(f"SQLite export failed: {e}")
            return False
    
    def list_backups(self) -> List[Dict]:
        """
        List all available backups.
        
        Returns:
            List of dictionaries with backup information
        """
        backups = []
        
        for backup_file in sorted(self.backup_dir.glob("face_encodings_backup_*.pkl"), reverse=True):
            stat = backup_file.stat()
            backups.append({
                'filename': backup_file.name,
                'path': str(backup_file),
                'size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'size_mb': stat.st_size / (1024 * 1024)
            })
        
        return backups
    
    def get_database_stats(self) -> Dict:
        """
        Get statistics about the current database.
        
        Returns:
            Dictionary with database statistics
        """
        if not self.encodings_file.exists():
            return {
                'exists': False,
                'size': 0,
                'total_faces': 0,
                'unique_persons': 0
            }
        
        try:
            with open(self.encodings_file, 'rb') as f:
                data = pickle.load(f)
            
            names = data.get('names', [])
            encodings = data.get('encodings', [])
            
            stat = self.encodings_file.stat()
            
            return {
                'exists': True,
                'path': str(self.encodings_file),
                'size': stat.st_size,
                'size_mb': stat.st_size / (1024 * 1024),
                'total_faces': len(names),
                'unique_persons': len(set(names)),
                'version': data.get('version', 'unknown'),
                'last_modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {
                'exists': True,
                'error': str(e)
            }
