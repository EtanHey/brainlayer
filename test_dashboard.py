#!/usr/bin/env python3
"""Demo script to test the dashboard functionality."""

import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from zikaron.dashboard import DashboardApp

if __name__ == "__main__":
    print("Testing zikaron dashboard...")
    
    try:
        app = DashboardApp()
        print("✓ Dashboard app created successfully")
        
        # Test database connection
        app.setup_database()
        print(f"✓ Database connected: {app.stats.get('total_chunks', 0)} chunks")
        
        # Test view rendering
        home_panel = app.run_home_view()
        print("✓ Home view rendered")
        
        memory_panel = app.run_memory_view()
        print("✓ Memory view rendered")
        
        print("\nDashboard components working! Run 'zikaron dashboard' to use interactively.")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
