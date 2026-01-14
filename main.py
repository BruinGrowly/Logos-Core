import os
import sys
from sensory.watcher import LogosWatcher

def main():
    # Define the workspace directory relative to the script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.join(base_dir, 'workspace')

    if not os.path.exists(workspace_dir):
        print(f"Creating workspace directory: {workspace_dir}")
        os.makedirs(workspace_dir)

    watcher = LogosWatcher(workspace_dir)
    watcher.start()

if __name__ == "__main__":
    main()
