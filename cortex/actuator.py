"""
Actuator - File System Actions with RLHF Logging
=================================================

Executes actions on files (move, copy, log) and records all
actions for RLHF correction detection.
"""

import os
import shutil
import time
from memory.action_log import get_action_log


class Actuator:
    """
    Executes file system actions and logs them for RLHF.
    """
    
    def __init__(self):
        self.action_log = get_action_log()

    def execute(self, action, file_path, concept: str = None):
        """
        Executes a specific action on a file.
        
        Args:
            action: Action dict with Type and parameters
            file_path: Path to the file
            concept: The semantic concept that triggered this action
        """
        if not action:
            return

        action_type = action.get('Type')
        
        if action_type == 'Move_File':
            destination = action.get('Destination')
            # Extract concept from action if not provided
            if concept is None:
                concept = action.get('Concept', destination)
            self.move_file(file_path, destination, concept)
        elif action_type == 'Log_Event':
            message = action.get('Message')
            self.log_event(message, file_path)
        else:
            print(f"[Actuator] Unknown action type: {action_type}")

    def move_file(self, source, dest_folder_rel, concept: str = None):
        """
        Moves a file to a destination folder and logs for RLHF.
        
        Args:
            source: Source file path
            dest_folder_rel: Destination folder (relative to source directory)
            concept: The semantic concept that triggered this move
        """
        try:
            source_dir = os.path.dirname(source)
            filename = os.path.basename(source)
            
            # Resolve destination path relative to the source directory
            dest_dir = os.path.join(source_dir, dest_folder_rel)
            
            if not os.path.exists(dest_dir):
                print(f"[Actuator] Creating directory: {dest_dir}")
                os.makedirs(dest_dir)

            dest_path = os.path.join(dest_dir, filename)

            # Handle overwrite or naming collision if needed (simple overwrite for now)
            shutil.move(source, dest_path)
            print(f"[Actuator] MOVED: {filename} -> {dest_folder_rel}")
            
            # Log the action for RLHF correction detection
            self.action_log.log_action(
                file_path=source,
                action_type='move',
                destination=dest_path,
                concept=concept or dest_folder_rel
            )
            
        except Exception as e:
            print(f"[Actuator] Error moving file: {e}")

    def log_event(self, message, file_path):
        """
        Logs an event to the console.
        """
        filename = os.path.basename(file_path)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[Actuator] [{timestamp}] {message} (File: {filename})")
