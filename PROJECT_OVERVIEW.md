# Milky Way Stacker - Project Architecture & History

## 1. Executive Summary & Current Status
**Milky Way Stacker** is a high-performance Python/CustomTkinter desktop application tailored for astrophotographers and landscape nightscape shooters. It enables advanced starry sky and ground landscape separation, multi-core star alignment (Astroalign), Fourier phase correlation landscape stabilization, satellite/airplane trail removal, independent sky/ground gamma controls, progressive soft blending, offline plate-solving with constellation overlays, and **100% native 16-bit High Dynamic Range (`uint16`) processing and TIFF export**.

* **Current Status**: All user requests implemented, verified on 21 raw 16-bit astronomical FITS frames (100% alignment success), 16-bit lossless pipeline fully tested.
* **Ready for**: Local testing, compilation into standalone executable (`build_app.py`), and GitHub synchronization.

---

## 2. Directory & File Structure

```text
MilkyWayStacker/
├── main.py                  # Application entry point (initializes CustomTkinter UI)
├── build_app.py             # PyInstaller build automation script with icon & catalog bundling
├── MilkyWayStacker.spec     # PyInstaller spec configuration
├── logo.png / logo.ico      # Application branding icons (Milky Way landscape theme)
├── app-150x150.png          # Association logo placed in the bottom sidebar
├── README.md                # Project documentation & usage guide
├── PROJECT_OVERVIEW.md      # Comprehensive architectural and evolution summary
├── .gitignore               # Excludes virtual environments, builds, and test datasets (prova/, *.fits)
│
├── core/                    # Computational Core Engine
│   ├── stacker.py           # 16-bit image loading, parallel stacking, gamma transforms, blending
│   ├── aligner.py           # Star detection, Astroalign matching, Fourier landscape alignment, plate-solving
│   └── sky_catalog.json     # Embedded offline astronomical star catalog (Bortle 4, Hipparcos stars & lines)
│
└── ui/                      # User Interface Layer
    ├── app.py               # Main UI window, background threads, controls, dialogs, save logic
    └── canvas.py            # Zero-lag interactive canvas for manual sky mask painting & overlay
```

---

## 3. History of User Requests & Progress Milestones

| # | User Request / Feature Requested | Problem / Context | Solution Implemented |
|---|----------------------------------|-------------------|----------------------|
| 1 | **Performance & Process Tracking** | Stacking was monolithic and opaque; user didn't know which frame was processing or how to cancel. | Added `ProcessPoolExecutor` parallel alignment with live per-frame progress callbacks and an instant cancellation event button. |
| 2 | **Airplane & Satellite Trail Removal** | Transient light streaks from satellites (Starlink) and airplanes ruin stacked night sky photos. | Implemented vectorized Sigma-Clipping stacking (`stack_chunk`) along pixel stacks to detect and suppress transient bright anomalies. |
| 3 | **Multi-Core Stacking** | Large 24MP image arrays caused CPU bottlenecks during median/sigma clipping. | Sliced 4D arrays into horizontal chunks across CPU cores with `ThreadPoolExecutor` and progress updates. |
| 4 | **Independent Gamma Controls (Sky & Ground)** | Nightscapes have starkly different brightness requirements for starry sky vs dark foreground. | Added split **Gamma Sky** and **Gamma Ground** sliders with soft transition feathering (up to 1000px). |
| 5 | **Canvas Brush Lag Fix** | Painting mask on full 24MP images caused severe stuttering. | Redesigned canvas with a zero-lag coordinate buffer, separating real-time cursor strokes from background redraws. |
| 6 | **Sky Mask Show/Hide Toggle** | Red mask overlay obscured preview of gamma adjustments. | Added dedicated **Hide Sky Mask / Show Sky Mask** toggle button in the sidebar. |
| 7 | **Progressive Blending** | Hard edges were visible between sky and terrain when using different gamma values. | Implemented bilinear-accelerated `feather_mask` up to 1000px radius for ultra-smooth transition gradients. |
| 8 | **Pre-Launch Statistics Speedup** | Checking feature counts took 3+ seconds due to massive morphological mask erosion. | Optimized `erode_mask` using downscaled kernel processing, reducing execution time from 3000ms to under 2ms (1000x speedup). |
| 9 | **FITS Frame Alignment Failures** | Test dataset of 21 FITS frames failed alignment (only 1 frame stacked; reported only 64 stars). | Fixed 16-bit hot pixel clipping, corrected inverted gamma math, and added an adaptive star count fallback sequence. |
| 10 | **Full 16-bit High Dynamic Range Pipeline** | Output TIFF was 8-bit, causing quantization and loss of subtle astrophotography data. | Converted the entire loading, warping, stacking, gamma, and saving pipeline to native **16-bit (`uint16`, 0–65535)**. |
| 11 | **Post-Stacking Color Calibration & Sensor WB** | Astronomical FITS had heavy green cast and washed-out white skies due to Bayer CFA quantum efficiency and sky background offsets. | Added background neutralization, Sony sensor QE presets (IMX294/IMX492, IMX571, IMX533, IMX585), Auto Star Photometric WB, SCNR green reduction, and saturation boost. |
| 12 | **Multi-Folder Incremental Loading & Disambiguation** | Loading files in multiple batches or across directories with identical file names overwritten the list or caused confusion. | Implemented incremental list append, duplicate path filtering, parent directory label formatting `[dir] filename.fits`, and a `Clear List` button. |
| 13 | **Bayer Channel Correction & DSLR/Reflex Color Engine** | Inverted Red/Blue channels due to OpenCV Bayer conversion naming convention (`COLOR_BayerRG2BGR`), and lack of DSLR 3x3 color matrix (CCM) and tone curves. | Fixed debayering with `COLOR_BayerXX2RGB` (native OpenCV BGR), added 3x3 CCM vector matrix transformation, S-Curve photographic tone mapping ($f(x)=x^2(3-2x)$), and calibrated built-in sensor presets. |
| 14 | **Project Persistence (.mws / .json Save & Load)** | Users had to re-load files, re-tune all sliders, and re-draw the sky mask from scratch on every session. | Added `Save Project` and `Load Project` buttons with full workspace persistence (relative/absolute path resolution, all settings/sliders, and base64 PNG sky mask preservation). |

---

## 4. Past Failures, Root Causes & Resolutions

### Failure 1: FITS Dark Images & "Somma solo 1 frame"
* **Root Cause A (Hot Pixels)**: Saturated sensor hot pixels (value 65534 in 16-bit raw FITS) squashed 99.9% of normal image data into values 0..2 when using linear min-max normalization `(data - min)/(max - min)*255`.
* **Root Cause B (Inverted Gamma Math)**: `apply_gamma` computed `1.0 / gamma`. Dragging the slider below 1.0 (intended to brighten shadows) raised values to power > 1.0, turning dark skies pitch black (0). Consequently, `astroalign` found 0 matching stars on subsequent frames.
* **Resolution**: 
  1. Implemented percentile scaling (`0.01%` to `99.95%`) in `load_image` in [core/stacker.py](file:///c:/ProgettiPy/MilkyWayStacker/core/stacker.py).
  2. Fixed gamma formula to standard power-law $V_{out} = V_{in}^\gamma$.
  3. Result: 280+ stars detected in sky region and all frames visible.

### Failure 2: Triangle Exhaustion in Astroalign on Drifted Frames
* **Root Cause**: Fixed `max_stars=50` caused triangle matching to exhaust on frames with slight perspective shift or faint noise (e.g. frames 5, 6, 14, 18).
* **Resolution**: Implemented an adaptive fallback loop over `max_stars = [40, 60, 80, 100, 150, 200, 300]` in `detect_and_align` ([core/aligner.py](file:///c:/ProgettiPy/MilkyWayStacker/core/aligner.py)).
* **Result**: **21 out of 21 frames (100%) aligned and stacked seamlessly**.

### Failure 3: 8-bit Quantization Loss on Output TIFF
* **Root Cause**: `load_image` and `stack_chunk` converted intermediate and final arrays to `np.uint8`.
* **Resolution**: 
  1. `load_image` returns full `np.uint16` (0..65535) for FITS, RAW camera files (`rawpy` `output_bps=16`), and 16-bit TIFFs.
  2. `cv2.warpAffine` and `cv2.warpPerspective` applied directly on 16-bit arrays (`uint16`).
  3. `apply_gamma` uses a 65,536-entry 16-bit LUT table for instant lossless curve adjustments.
  4. `save_result` saves true 16-bit TIFF (`cv2.imwrite(..., img_16)`).

### Failure 4: FITS Button Load Mask Blocking & Missing Drag-and-Drop
* **Root Cause A (Canvas Mask None Return)**: `apply_split_gamma_correction` was invoked before `self.canvas.set_image` initialized the mask (`self.canvas.mask_img`). When `get_mask()` returned `None`, `apply_split_gamma_correction` exited immediately, preventing image display and blocking mask painting.
* **Root Cause B (Missing Drag & Drop Hook)**: Drag-and-drop file ingestion was not hooked into the OS shell on the CustomTkinter root and canvas.
* **Resolution**:
  1. Refactored file ingestion into unified `_process_loaded_files(file_paths)`.
  2. Ensured `self.canvas.set_image(self.reference_img, reset_mask=True)` is called immediately when loading reference image.
  3. Added graceful fallback in `apply_split_gamma_correction` when mask is uninitialized.
  4. Integrated `windnd.hook_dropfiles` on both main window and canvas for full OS drag-and-drop support.

### Failure 5: Out Of Memory (OOM) on Large Sequences (`Unable to allocate 903 MiB`)
* **Root Cause**:
  1. Warped frames were kept in `float32` in `sky_list` and `ground_list` (140 MB per frame instead of 70 MB in `uint16`).
  2. `np.stack(img_list, axis=0)` created a massive monolithic 4D array in RAM $(N, H, W, C)$.
  3. Splitting into 4 large chunks of 705 scanlines caused 4 parallel threads in `ThreadPoolExecutor` to simultaneously allocate ~903 MiB each in `float32` for `np.where`, `np.abs`, and `np.median`, exceeding available system RAM.
* **Resolution**:
  1. Stored warped images natively in 16-bit (`np.uint16`), reducing base list RAM by 50%.
  2. Implemented lightweight streaming accumulation for Average stacking (`acc += img` in $(H, W, C)$ space), dropping RAM usage for Average mode to ~140 MB total regardless of frame count.
  3. Implemented dynamic micro-slice streaming for Median & Sigma-clipping (target $\le 60\text{ MB}$ per chunk), writing slices directly into the pre-allocated output buffer.

### Failure 6: Bayer CFA Channel Inversion & Raw Sensor Unbalanced Colors
* **Root Cause A (OpenCV Bayer Convention Inversion)**: `cv2.COLOR_BayerRG2BGR` produces `[Red, Green, Blue]` (RGB in memory), which was treated as standard OpenCV BGR `[Blue, Green, Red]` throughout the pipeline, inverting the Red and Blue channels in display, color calculations, and output saving.
* **Root Cause B (Missing DSLR/Reflex ISP Stage)**: Raw OSC astronomical CMOS sensors (e.g. Sony IMX294 in ZWO ASI294MC) lack on-camera hardware image signal processing (ISP), requiring white balance gains, $3 \times 3$ color matrix (CCM) spectral crosstalk removal, and non-linear tone curves.
* **Resolution**:
  1. Changed Bayer conversions in `load_image` to `cv2.COLOR_BayerXX2RGB` to output true OpenCV BGR (`Channel 0 = Blue, 1 = Green, 2 = Red`).
  2. Embedded calibrated $WB$ gains and $3 \times 3$ CCM directly in `SENSOR_PRESETS` in [core/stacker.py](file:///c:/ProgettiPy/MilkyWayStacker/core/stacker.py).
  3. Integrated vector matrix transformation $\mathbf{M}_{BGR} = P \cdot (\text{CCM} \times \text{diag}(WB_R, 1.0, WB_B)) \cdot P$ via `cv2.transform`.
  4. Implemented 16-bit photographic S-Curve tone mapping $f(x) = x^2(3-2x)$ to deepen the dark sky background without crushing galactic details or clipping stars.

---

## 5. Detailed Module Architecture

### A. `core/stacker.py`
- `load_image(path)`: Robust loader for FITS (with 16-bit RGGB/BGGR/GRBG/GBRG Bayer debayering & percentile normalization), RAW camera files (CR2, CR3, NEF, ARW, DNG via `rawpy` 16-bit), 16-bit TIFF, and standard 8-bit images.
- `apply_gamma(img, gamma)`: 16-bit / 8-bit power-law gamma lookup table transformation.
- `SENSOR_PRESETS`: Calibrated quantum efficiency (QE) response multipliers for Sony IMX294/IMX492, IMX571, IMX533, IMX585, IMX462, and DSLRs.
- `calculate_auto_white_balance(img, mask)`: Photometric star flux integrator estimating natural White Balance.
- `apply_color_calibration(...)`: Complete 16-bit color balancing pipeline (background neutralization, channel gains, SCNR green cast removal, HSV saturation boost).
- `feather_mask(mask, radius)`: Bilinear downsampled/upscaled progressive Gaussian blur for seamless boundary blending.
- `stack_chunk(chunk, stack_mode, remove_trails, sigma_factor)`: Vectorized multi-frame stacking (Average, Median, or Sigma-Clipped Outlier Rejection).
- `stack_parallel_chunks(img_list, ...)`: Multi-core CPU slicing across image height.
- `align_single_frame(...)`: Process worker that extracts alignment transforms on 8-bit buffers and warps full-precision 16-bit raw frames.
- `stack_images(...)`: Main stacking orchestrator managing worker pools, reference frame selection, separate sky/ground stacking, gamma adjustments, and final 16-bit compositing.

### B. `core/aligner.py`
- `detect_stars_centroids(img, mask, contrast, sigma, max_stars)`: Dynamic thresholding and contour centroid extraction for starry skies.
- `detect_and_filter_stars(img, mask, ...)`: Homogeneous 6x6 spatial grid binning to ensure balanced star distribution.
- `align_landscape_optical_flow(ref_img, target_img, mask)`: Pure translation Fourier Phase Correlation on CLAHE-enhanced Canny edge maps.
- `detect_and_align(...)`: High-level alignment router with multi-threshold `max_stars` fallback for Astroalign.
- `check_features(...)`: Fast diagnostic analyzer reporting star counts and ground points for "Pre Launch statistics".
- `draw_constellations(img, mask, cancel_event)`: Offline plate-solving engine identifying constellations from `sky_catalog.json` and rendering 16-bit / 8-bit annotations.

### C. `ui/canvas.py`
- `ZeroLagCanvas`: Tkinter Canvas subclass featuring high-speed brush painting, interactive circle cursor, semi-transparent red sky mask overlay, and 16-bit to 8-bit downscaling for screen display.

### D. `ui/app.py`
- `MilkyWayStackerApp`: CustomTkinter interface orchestrating user actions, asynchronous worker threads, cancellation handles, progress bars, real-time preview rendering during slider dragging, and 16-bit TIFF/PNG file saving dialogs.

---

## 6. How to Run & Build

### Run locally:
```powershell
python main.py
```

### Build Standalone Executable (.exe):
```powershell
python build_app.py
```
* Generates standalone binary in `dist/MilkyWayStacker.exe` with bundled astrometry catalog, camera RAW libraries, and custom app icon.
