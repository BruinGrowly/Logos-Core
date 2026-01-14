import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from cortex.interpreter import LogosInterpreter
from cortex.rules_engine import RulesEngine
from cortex.semantic_engine import SemanticEngine
from cortex.actuator import Actuator

class WorkspaceHandler(FileSystemEventHandler):
    def __init__(self, interpreter, rules_engine, semantic_engine, actuator):
        self.interpreter = interpreter
        self.rules_engine = rules_engine
        self.semantic_engine = semantic_engine
        self.actuator = actuator

    def on_created(self, event):
        # Trigger when a file is created
        if not event.is_directory:
            self.process_change(event.src_path)

    def on_modified(self, event):
        # Trigger when a file is modified
        if not event.is_directory:
            self.process_change(event.src_path)

    def process_change(self, file_path):
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
                    self.actuator.execute(action, file_path)
                    return # Reflex matched, stop processing

                # 3b. Semantic Check (Semantic Engine)
                print("[Watcher] No reflex match. Engaging Semantic Brain...")
                action = self.semantic_engine.evaluate(file_path, rules)
                
                if action:
                    self.actuator.execute(action, file_path)
                else:
                    print(f"[Watcher] No matching rules (Reflex or Semantic) for {filename}.")
            else:
                print(f"[Watcher] No rules defined in project.logos.")
                
        else:
            print(f"[Watcher] File detected: {filename}. No project.logos found in {directory}.")

class LogosWatcher:
    def __init__(self, watch_dir):
        self.watch_dir = watch_dir
        self.interpreter = LogosInterpreter()
        self.rules_engine = RulesEngine()
        self.semantic_engine = SemanticEngine()
        self.actuator = Actuator()
        self.observer = Observer()

    def start(self):
        event_handler = WorkspaceHandler(self.interpreter, self.rules_engine, self.semantic_engine, self.actuator)
        self.observer.schedule(event_handler, self.watch_dir, recursive=False)
        self.observer.start()
        print(f"Logos Watcher started. Monitoring: {self.watch_dir}")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.observer.stop()
        self.observer.join()
        print("Logos Watcher stopped.")
