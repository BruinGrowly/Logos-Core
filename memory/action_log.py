"""
Action Log - System Action Tracking
===================================

Tracks all system-initiated file actions for RLHF correction detection.
When a user manually moves a file that the system recently placed,
we can detect this as a correction.

Storage: workspace/action_log.json
"""

import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta


class ActionLog:
    """
    Logs system actions for RLHF correction detection.
    
    Tracks file moves with timestamps so we can detect when
    a user manually corrects a system classification.
    """
    
    def __init__(self, log_path: str = None):
        """
        Initialize the action log.
        
        Args:
            log_path: Path to log file. Defaults to workspace/action_log.json
        """
        if log_path is None:
            project_root = Path(__file__).parent.parent
            log_dir = project_root / "workspace"
            log_dir.mkdir(exist_ok=True)
            self.log_path = log_dir / "action_log.json"
        else:
            self.log_path = Path(log_path)
        
        self._entries = self._load_log()
    
    def _load_log(self) -> List[Dict]:
        """Load log from disk."""
        if self.log_path.exists():
            try:
                with open(self.log_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []
    
    def _save_log(self) -> None:
        """Save log to disk."""
        try:
            with open(self.log_path, 'w') as f:
                json.dump(self._entries, f, indent=2)
        except IOError as e:
            print(f"[ActionLog] Warning: Could not save log: {e}")
    
    def log_action(self, file_path: str, action_type: str, 
                   destination: str, concept: str = None) -> None:
        """
        Log a system action.
        
        Args:
            file_path: Original file path
            action_type: Type of action ('move', 'copy', etc.)
            destination: Where the file was moved to
            concept: The semantic concept that triggered the action
        """
        entry = {
            'file_path': str(file_path),
            'file_name': os.path.basename(file_path),
            'action_type': action_type,
            'destination': str(destination),
            'concept': concept,
            'timestamp': datetime.now().isoformat(),
            'epoch': time.time()
        }
        
        self._entries.append(entry)
        self._save_log()
        
        print(f"[ActionLog] Logged: {action_type} '{os.path.basename(file_path)}' -> {destination} (concept: {concept})")
    
    def get_recent_action(self, file_path: str = None, file_name: str = None,
                          within_minutes: int = 10) -> Optional[Dict]:
        """
        Get the most recent action for a file.
        
        Args:
            file_path: Full path to check (optional)
            file_name: Filename to check (optional, used if file was moved)
            within_minutes: Time window for considering actions "recent"
            
        Returns:
            Action entry dict if found within time window, None otherwise
        """
        if not file_path and not file_name:
            return None
        
        cutoff = time.time() - (within_minutes * 60)
        
        # Search in reverse (most recent first)
        for entry in reversed(self._entries):
            # Check if within time window
            if entry.get('epoch', 0) < cutoff:
                continue
            
            # Match by file path or filename
            if file_path and entry.get('file_path') == str(file_path):
                return entry
            if file_name and entry.get('file_name') == file_name:
                return entry
            
            # Also check destination path (file may have been moved there)
            dest = entry.get('destination', '')
            if file_path and dest and str(file_path).startswith(str(dest)):
                return entry
        
        return None
    
    def clear_old_entries(self, older_than_hours: int = 24) -> int:
        """
        Remove old log entries.
        
        Args:
            older_than_hours: Remove entries older than this
            
        Returns:
            Number of entries removed
        """
        cutoff = time.time() - (older_than_hours * 3600)
        original_count = len(self._entries)
        
        self._entries = [
            e for e in self._entries 
            if e.get('epoch', 0) >= cutoff
        ]
        
        removed = original_count - len(self._entries)
        if removed > 0:
            self._save_log()
            print(f"[ActionLog] Cleared {removed} old entries")
        
        return removed
    
    def get_all_entries(self) -> List[Dict]:
        """Get all log entries."""
        return self._entries.copy()
    
    def __len__(self) -> int:
        """Return number of entries."""
        return len(self._entries)


# Global instance for easy access
_action_log = None

def get_action_log() -> ActionLog:
    """Get the global action log instance."""
    global _action_log
    if _action_log is None:
        _action_log = ActionLog()
    return _action_log


if __name__ == "__main__":
    # Quick test
    log = ActionLog()
    print(f"[ActionLog] Loaded {len(log)} entries")
    print(f"[ActionLog] Log file: {log.log_path}")
