# MilkyWayStacker

A standalone desktop application for astrophotography stacking. It aligns and averages sequences of starry skies while keeping the landscape sharp.


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

### 3. Color Calibration & White Balance (Post-Stacking / Preview)
* **Sensor Quantum Efficiency (QE) Presets**: Calibrated channel multipliers for Sony IMX294/IMX492, IMX571, IMX533, IMX585, IMX462, and DSLRs.
* **Background Neutralization**: Aligns the black point across R, G, and B channels to produce a clean, neutral dark sky background.
* **Auto Photometric Stars WB**: Automatically calculates White Balance multipliers from non-saturated stellar flux in the sky region.
* **SCNR (Green Noise Suppression)**: Eliminates green Bayer color cast.
* **Color Saturation**: Boosts celestial colors (Milky Way dust lanes, Andromeda galaxy, stellar hues) in 16-bit high dynamic range.

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
