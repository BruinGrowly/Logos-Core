"""
Logos-Core Main Entry Point
============================

The central nervous system of the Logos project.

Usage:
    python main.py                  # Default: watch mode
    python main.py --watch          # Watch workspace for file changes
    python main.py --meditate       # Run dream cycle (metacognition)
    python main.py --dashboard      # Launch harmony dashboard
"""

import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Logos-Core - Semantic File Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  watch      Monitor workspace for file changes (default)
  meditate   Run dream cycle to consolidate learnings
  dashboard  Launch the harmony dashboard
        """
    )
    
    parser.add_argument(
        '--watch', '-w',
        action='store_true',
        help='Watch workspace for file changes (default mode)'
    )
    
    parser.add_argument(
        '--meditate', '-m',
        action='store_true',
        help='Run dream cycle to consolidate learnings'
    )
    
    parser.add_argument(
        '--dashboard', '-d',
        action='store_true',
        help='Launch the harmony dashboard'
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=5000,
        help='Port for dashboard (default: 5000)'
    )
    
    args = parser.parse_args()
    
    # Define the workspace directory relative to the script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.join(base_dir, 'workspace')

    if not os.path.exists(workspace_dir):
        print(f"Creating workspace directory: {workspace_dir}")
        os.makedirs(workspace_dir)
    
    # Determine mode
    if args.meditate:
        run_meditate()
    elif args.dashboard:
        run_dashboard(base_dir, args.port)
    else:
        # Default to watch mode
        run_watch(workspace_dir)


def run_watch(workspace_dir: str):
    """Run the file watcher."""
    from sensory.watcher import LogosWatcher
    
    print("\n" + "=" * 60)
    print("  🧠 LOGOS-CORE: Watch Mode")
    print("=" * 60)
    print(f"\n  Monitoring: {workspace_dir}")
    print("  Press Ctrl+C to stop\n")
    
    watcher = LogosWatcher(workspace_dir)
    watcher.start()


def run_meditate():
    """Run the dream cycle for metacognitive processing."""
    print("\n" + "=" * 60)
    print("  🧘 LOGOS-CORE: Meditate Mode")
    print("=" * 60 + "\n")
    
    from cortex.autopoiesis import dream
    
    print("Initiating Dream Cycle...\n")
    report = dream.start_cycle()
    
    print("\n" + "-" * 60)
    print("  Awake. See workspace/morning_brief.md for details.")
    print("-" * 60 + "\n")


def run_dashboard(base_dir: str, port: int):
    """Run the harmony dashboard."""
    print("\n" + "=" * 60)
    print("  📊 LOGOS-CORE: Dashboard Mode")
    print("=" * 60 + "\n")
    
    from cortex.autopoiesis.dashboard import run_dashboard as start_dashboard
    start_dashboard(base_dir, port)


if __name__ == "__main__":
    main()
