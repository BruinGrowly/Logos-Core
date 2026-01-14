"""
Watcher - File System Monitor with RLHF Correction Detection
=============================================================

Monitors the workspace for file changes and triggers the Logos
processing pipeline. Now includes move event detection for
RLHF learning from user corrections.
"""

import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from cortex.interpreter import LogosInterpreter
from cortex.rules_engine import RulesEngine
from cortex.semantic_engine import SemanticEngine
from cortex.actuator import Actuator
from memory.action_log import get_action_log
from memory.hard_negatives import get_hard_negatives
from sensory.notifications import notify_correction


# Time window for considering a user move as a correction (minutes)
CORRECTION_WINDOW_MINUTES = 10


class WorkspaceHandler(FileSystemEventHandler):
    """
    Handles file system events including creations, modifications, and moves.
    Detects user corrections for RLHF learning.
    """
    
    def __init__(self, interpreter, rules_engine, semantic_engine, actuator):
        self.interpreter = interpreter
        self.rules_engine = rules_engine
        self.semantic_engine = semantic_engine
        self.actuator = actuator
        self.action_log = get_action_log()
        self.hard_negatives = get_hard_negatives()

    def on_created(self, event):
        """Trigger when a file is created."""
        if not event.is_directory:
            self.process_change(event.src_path)

    def on_modified(self, event):
        """Trigger when a file is modified."""
        if not event.is_directory:
            self.process_change(event.src_path)
    
    def on_moved(self, event):
        """
        Trigger when a file is moved.
        
        Detects user corrections when a file is moved out of a
        location where the system recently placed it.
        """
        if event.is_directory:
            return
        
        old_path = event.src_path
        new_path = event.dest_path
        filename = os.path.basename(new_path)
        
        print(f"[Watcher] Move detected: {filename}")
        print(f"[Watcher]   From: {old_path}")
        print(f"[Watcher]   To:   {new_path}")
        
        # Check if this was a system-moved file
        recent_action = self.action_log.get_recent_action(
            file_path=old_path,
            file_name=filename,
            within_minutes=CORRECTION_WINDOW_MINUTES
        )
        
        if recent_action:
            # Was the file moved to a DIFFERENT location than where system put it?
            system_dest = recent_action.get('destination', '')
            
            # Normalize paths for comparison
            if os.path.normpath(new_path) != os.path.normpath(system_dest):
                # USER CORRECTION DETECTED!
                self.handle_correction(
                    filename=filename,
                    old_path=old_path,
                    new_path=new_path,
                    recent_action=recent_action
                )
            else:
                print(f"[Watcher] File moved within same destination - not a correction")
        else:
            print(f"[Watcher] No recent system action for this file - normal user move")

    def handle_correction(self, filename: str, old_path: str, 
                          new_path: str, recent_action: dict):
        """
        Handle a user correction to a system classification.
        
        1. Store hard negative (file + wrong concept)
        2. Store positive (file + correct folder)
        3. Show notification
        """
        wrong_concept = recent_action.get('concept', 'Unknown')
        correct_folder = os.path.basename(os.path.dirname(new_path))
        
        print(f"[Watcher] *** USER CORRECTION DETECTED ***")
        print(f"[Watcher]   File: {filename}")
        print(f"[Watcher]   Wrong: {wrong_concept}")
        print(f"[Watcher]   Correct: {correct_folder}")
        
        # Get file vector for learning
        file_vector = self._get_file_vector(new_path)
        
        if file_vector is not None:
            # Store hard negative
            self.hard_negatives.add_negative(
                file_vector=file_vector,
                wrong_concept=wrong_concept,
                file_name=filename
            )
            
            # Store positive reinforcement
            self.hard_negatives.add_positive(
                file_vector=file_vector,
                correct_concept=correct_folder,
                file_name=filename
            )
            
            print(f"[Watcher] Learning stored successfully!")
        else:
            print(f"[Watcher] Warning: Could not vectorize file for learning")
        
        # Show notification to user
        notify_correction(
            file_name=filename,
            wrong_concept=wrong_concept,
            correct_folder=correct_folder
        )
    
    def _get_file_vector(self, file_path: str):
        """
        Get vector embedding for a file (using content or filename).
        """
        try:
            # Try to read content first
            from sensory.reader import read_header, is_supported
            
            if is_supported(file_path):
                content = read_header(file_path, limit=1000)
                if content and len(content.strip()) > 10:
                    return self.semantic_engine.memory.encode(content)
            
            # Fallback to filename
            filename = os.path.basename(file_path)
            clean_name = filename.replace('_', ' ').replace('-', ' ')
            name_without_ext = os.path.splitext(clean_name)[0]
            return self.semantic_engine.memory.encode(name_without_ext)
            
        except Exception as e:
            print(f"[Watcher] Error getting file vector: {e}")
            return None

    def process_change(self, file_path):
        """Process a file creation or modification."""
        filename = os.path.basename(file_path)
        directory = os.path.dirname(file_path)
        
        # Look for project.logos in the same directory
        logos_path = os.path.join(directory, 'project.logos')
        
        if not os.path.exists(file_path):
            return

        if filename == 'project.logos':
            print(f"[Watcher] project.logos modified. Reloading rules...")
            return

        if os.path.exists(logos_path):
            # 1. Parse the Context
            data = self.interpreter.parse(logos_path)
            
            # 2. Extract Rules
            rules = data.get('Rules', [])
            
            if rules:
                # 3a. Reflex Check (Rules Engine)
                action = self.rules_engine.evaluate(file_path, rules)
                
                if action:
                    # Get concept from the matching rule
                    concept = self._extract_concept(action, rules)
                    self.actuator.execute(action, file_path, concept)
                    return  # Reflex matched, stop processing

                # 3b. Semantic Check (Semantic Engine)
                print("[Watcher] No reflex match. Engaging Semantic Brain...")
                action = self.semantic_engine.evaluate(file_path, rules)
                
                if action:
                    concept = self._extract_concept(action, rules)
                    self.actuator.execute(action, file_path, concept)
                else:
                    print(f"[Watcher] No matching rules (Reflex or Semantic) for {filename}.")
            else:
                print(f"[Watcher] No rules defined in project.logos.")
                
        else:
            print(f"[Watcher] File detected: {filename}. No project.logos found in {directory}.")
    
    def _extract_concept(self, action: dict, rules: list) -> str:
        """Extract the concept from an action or its triggering rule."""
        # Check if action has concept
        if action.get('Concept'):
            return action['Concept']
        
        # Try to find matching rule and get concept from trigger
        for rule in rules:
            if rule.get('Action') == action:
                trigger = rule.get('Trigger', {})
                if trigger.get('Concept'):
                    return trigger['Concept']
        
        # Fallback to destination
        return action.get('Destination', 'Unknown')


class LogosWatcher:
    """
    Main watcher class that monitors a directory for file changes.
    """
    
    def __init__(self, watch_dir):
        self.watch_dir = watch_dir
        self.interpreter = LogosInterpreter()
        self.rules_engine = RulesEngine()
        self.semantic_engine = SemanticEngine()
        self.actuator = Actuator()
        self.observer = Observer()

    def start(self):
        """Start watching the directory."""
        event_handler = WorkspaceHandler(
            self.interpreter, 
            self.rules_engine, 
            self.semantic_engine, 
            self.actuator
        )
        self.observer.schedule(event_handler, self.watch_dir, recursive=True)
        self.observer.start()
        print(f"Logos Watcher started. Monitoring: {self.watch_dir}")
        print(f"  - RLHF correction detection: ENABLED")
        print(f"  - Correction window: {CORRECTION_WINDOW_MINUTES} minutes")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop watching."""
        self.observer.stop()
        self.observer.join()
        print("Logos Watcher stopped.")
