import os
import sys
import subprocess
import shutil

print("Checking dependencies...")
try:
    import PyInstaller
except ImportError:
    print("Installing PyInstaller...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

import customtkinter
import PyInstaller.__main__

# Find path of customtkinter module to include its assets
ctk_path = os.path.dirname(customtkinter.__file__)
print(f"CustomTkinter located at: {ctk_path}")

# Detect platform separator for PyInstaller --add-data
# Windows uses ';', Linux/macOS uses ':'
sep = ';' if sys.platform.startswith('win') else ':'

add_data_ctk = f"{ctk_path}{sep}customtkinter"
add_data_logo = f"logo.png{sep}."

print(f"Using path separator '{sep}' for platform '{sys.platform}'")
print("Starting compilation of standalone executable...")

PyInstaller.__main__.run([
    'main.py',
    '--name=MilkyWayStacker',
    '--onefile',
    '--windowed',
    '--icon=logo.ico',
    f'--add-data={add_data_ctk}',
    f'--add-data={add_data_logo}',
    f'--add-data=core{sep}core',
    '--clean',
    '--noconfirm'
])

print("\n------------------------------------------------------------")
print("SUCCESS: Compilation finished successfully!")
print("You can find the single executable file at:")
if sys.platform.startswith('win'):
    print(f" -> {os.path.abspath('dist/MilkyWayStacker.exe')}")
else:
    print(f" -> {os.path.abspath('dist/MilkyWayStacker')}")
print("------------------------------------------------------------")
