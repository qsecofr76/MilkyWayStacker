# MilkyWayStacker

An elegant, open-source astrophotography stacking tool built in Python. Features high-performance sky star alignment and rigid, deformation-free landscape contour registration designed as an alternative to Sequator.

---

### 📥 Download the App
* **[Download MilkyWayStacker for Windows (Executable)](https://github.com/qsecofr76/MilkyWayStacker/releases/latest)**

---

## Features
- **Sky Star Alignment**: Automatically detects stars, filters outliers, and samples stars homogeneously using a spatial grid for precise partial affine alignment.
- **Deformation-Free Landscape Alignment**: Uses Fourier Phase Correlation on Bilateral-filtered Canny edge contours to achieve sub-pixel accurate rigid landscape registration (translation + rotation) with zero scaling or skew warp distortions.
- **Dynamic Star Sensitivity**: Adjust contrast thresholds and blur factors dynamically to optimize stars detection.
- **Masking Canvas**: Paint your sky mask directly on the reference frame in the GUI.
- **Robust File Support**: Supports loading JPG, PNG, TIFF, DNG raw files, and FITS astronomical files (via `astropy` and `rawpy`). Saves in high-quality TIFF, PNG, or JPEG.
- **Association Branding**: Displays the custom association logo dynamically in the scrollable sidebar.

## Installation
Ensure you have Python 3.10+ installed. Install the dependencies:
```bash
pip install -r requirements.txt
```

*(Optional) For DNG and FITS astronomical formats support:*
```bash
pip install rawpy astropy
```

## Running the Application
To launch the GUI app:
```bash
python main.py
```

## License
MIT License
