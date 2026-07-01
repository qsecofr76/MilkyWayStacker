import sys
import os
import multiprocessing

# Add current directory to python path to resolve core/ui modules correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ui.app import MilkyWayStackerApp

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = MilkyWayStackerApp()
    app.mainloop()
