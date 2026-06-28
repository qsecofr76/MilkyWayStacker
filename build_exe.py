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

# In Windows PyInstaller, add-data files are separated by a semicolon (;)
# Syntax: "source_path;destination_relative_path"
add_data_ctk = f"{ctk_path};customtkinter"
add_data_logo = "logo.png;."

print("Starting compilation of standalone Windows executable...")

PyInstaller.__main__.run([
    'main.py',
    '--name=MilkyWayStacker',
    '--onefile',
    '--windowed',
    f'--add-data={add_data_ctk}',
    f'--add-data={add_data_logo}',
    '--clean',
    '--noconfirm'
])

print("\n------------------------------------------------------------")
print("SUCCESS: Compilation finished successfully!")
print("You can find the single executable file at:")
print(f" -> {os.path.abspath('dist/MilkyWayStacker.exe')}")
print("------------------------------------------------------------")
