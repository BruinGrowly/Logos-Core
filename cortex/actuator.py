import os
import shutil
import time

class Actuator:
    def __init__(self):
        pass

    def execute(self, action, file_path):
        """
        Executes a specific action on a file.
        """
        if not action:
            return

        action_type = action.get('Type')
        
        if action_type == 'Move_File':
            destination = action.get('Destination')
            self.move_file(file_path, destination)
        elif action_type == 'Log_Event':
            message = action.get('Message')
            self.log_event(message, file_path)
        else:
            print(f"[Actuator] Unknown action type: {action_type}")

    def move_file(self, source, dest_folder_rel):
        """
        Moves a file to a destination folder (relative to the file's current location).
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
            
        except Exception as e:
            print(f"[Actuator] Error moving file: {e}")

    def log_event(self, message, file_path):
        """
        Logs an event to the console.
        """
        filename = os.path.basename(file_path)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[Actuator] [{timestamp}] {message} (File: {filename})")
