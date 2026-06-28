# MilkyWayStacker

A standalone desktop application for astrophotography stacking. It aligns and averages sequences of starry skies while keeping the landscape sharp.

---

### 📥 Nightly Build
* **[Download MilkyWayStacker for Windows (Executable)](https://github.com/qsecofr76/MilkyWayStacker/releases/tag/nightly)**

---

## Technical Overview & Algorithms

MilkyWayStacker processes images by splitting them into two regions (sky and landscape) using a user-drawn mask, registering them independently, and blending them together.

### 1. Sky Alignment
* **Detection & Filtering**: Detects stars in the sky region using local peak extraction.
* **Homogeneous Grid Sampling**: Samples stars uniformly using a spatial grid to prevent clusters (e.g. Milky Way core) from biasing the alignment.
* **Transform Estimation**: Computes a coordinate transformation (Affine or Homography) using RANSAC outlier rejection to register the starry sky across frames.

### 2. Landscape Alignment (Deformation-free)
* **Contour Extraction**: Applies a Bilateral filter followed by high-sensitivity Canny edge tracing to capture faint silhouette lines (mountain profiles, buildings) under low light.
* **Fourier Phase Correlation**: Computes translation offsets in the frequency domain. This is translation-only (no scale or skew skewing) preventing blur/loss of high-frequency details due to interpolation warping.
* **Star Filtering**: Ignores short contours (stars/noise) using a length filter to prevent sky objects from biasing the landscape alignment.

---

## Installation & Requirements

Ensure you have Python 3.10+ installed.

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python main.py
   ```

---

## Compilation

To compile a standalone binary executable for Windows or Linux:
```bash
python build_app.py
```
The output file will be saved in the `dist/` directory.
