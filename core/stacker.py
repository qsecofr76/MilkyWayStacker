import os
import cv2
import numpy as np
import concurrent.futures
import multiprocessing
from core.aligner import detect_and_align

import json

# Preset channel multipliers and color correction matrices (CCM) for astronomical CMOS sensors and DSLRs
SENSOR_PRESETS = {
    "camera_calibration.json (ASI294MC)": {
        "r_gain": 1.51, "g_gain": 1.00, "b_gain": 1.416,
        "ccm": np.array([
            [ 1.0865, -0.0921,  0.0056],
            [ 0.1236,  0.7271,  0.1493],
            [-0.0250, -0.1518,  1.1768]
        ], dtype=np.float32),
        "use_scurve": True
    },
    "Reflex / DSLR Mode (CCM + S-Curve)": {
        "r_gain": 1.25, "g_gain": 1.00, "b_gain": 1.05,
        "ccm": np.array([
            [ 1.25, -0.20, -0.05],
            [-0.10,  1.20, -0.10],
            [-0.05, -0.20,  1.25]
        ], dtype=np.float32),
        "use_scurve": True
    },
    "Sony IMX294 / IMX492 (ASI294MC)": {
        "r_gain": 1.51, "g_gain": 1.00, "b_gain": 1.416,
        "ccm": None,
        "use_scurve": False
    },
    "Sony IMX571 (ASI2600MC)": {
        "r_gain": 1.65, "g_gain": 1.00, "b_gain": 1.25,
        "ccm": None,
        "use_scurve": False
    },
    "Sony IMX533 (ASI533MC)": {
        "r_gain": 1.68, "g_gain": 1.00, "b_gain": 1.30,
        "ccm": None,
        "use_scurve": False
    },
    "Sony IMX585 (ASI585MC)": {
        "r_gain": 1.50, "g_gain": 1.00, "b_gain": 1.20,
        "ccm": None,
        "use_scurve": False
    },
    "Sony IMX462 / IMX662": {
        "r_gain": 1.60, "g_gain": 1.00, "b_gain": 1.40,
        "ccm": None,
        "use_scurve": False
    },
    "DSLR / Mirrorless Daylight": {
        "r_gain": 2.10, "g_gain": 1.00, "b_gain": 1.50,
        "ccm": None,
        "use_scurve": False
    },
    "Auto Stars (Photometric)": None,  # Calculated automatically from star flux
    "None / As-Is": {
        "r_gain": 1.00, "g_gain": 1.00, "b_gain": 1.00,
        "ccm": None,
        "use_scurve": False
    },
    "Custom": None
}

def load_camera_calibration(json_path="camera_calibration.json"):
    """
    Loads white balance multipliers, 3x3 color correction matrix (CCM),
    and camera metadata from a calibration JSON file.
    """
    candidates = [json_path, "camera_calibration.json", os.path.join(os.path.dirname(__file__), "..", "camera_calibration.json")]
    target_path = None
    for p in candidates:
        if p and os.path.exists(p):
            target_path = p
            break

    if target_path:
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            wb_dict = data.get("white_balance_gains", {})
            wb_r = float(wb_dict.get("WB_R", 1.51))
            wb_g = float(wb_dict.get("WB_G", 1.00))
            wb_b = float(wb_dict.get("WB_B", 1.416))
            ccm_raw = data.get("color_correction_matrix_3x3", None)
            if ccm_raw is not None:
                ccm = np.array(ccm_raw, dtype=np.float32)
            else:
                ccm = np.eye(3, dtype=np.float32)
            camera_name = data.get("camera_model", "Calibrated Camera")
            return {
                "camera_model": camera_name,
                "r_gain": wb_r,
                "g_gain": wb_g,
                "b_gain": wb_b,
                "ccm": ccm,
                "path": target_path
            }
        except Exception as e:
            print(f"Error reading calibration JSON {target_path}: {e}")

    # Fallback default calibrated parameters for Sony IMX294 (ZWO ASI294MC)
    return {
        "camera_model": "ZWO ASI294MC Pro (Default CCM)",
        "r_gain": 1.51,
        "g_gain": 1.00,
        "b_gain": 1.416,
        "ccm": np.array([
            [ 1.0865, -0.0921,  0.0056],
            [ 0.1236,  0.7271,  0.1493],
            [-0.0250, -0.1518,  1.1768]
        ], dtype=np.float32),
        "path": None
    }

def apply_s_curve(img_16, strength=1.0):
    """
    Applies a photographic S-Curve tone mapping in 16-bit precision:
    f(x) = x^2 * (3 - 2x), weighted by strength.
    Deepens and cleans the dark sky background while preserving Milky Way dust structures
    and providing soft highlight roll-off without star clipping.
    """
    if img_16 is None or strength <= 0.0:
        return img_16
        
    x = np.linspace(0.0, 1.0, 65536, dtype=np.float32)
    s_curve_val = x * x * (3.0 - 2.0 * x)
    blended = (1.0 - float(strength)) * x + float(strength) * s_curve_val
    lut = np.clip(blended * 65535.0, 0.0, 65535.0).astype(np.uint16)
    
    return lut[img_16]

def calculate_auto_white_balance(img_bgr, mask=None):
    """
    Calculates optimal White Balance channel multipliers (R_gain, G_gain=1.0, B_gain)
    by detecting non-saturated stars in the starry sky region and measuring their integrated flux.
    """
    if img_bgr is None:
        return 1.0, 1.0, 1.0
        
    img_f = img_bgr.astype(np.float32)
    if mask is not None and np.any(mask > 128):
        valid = mask > 128
        b_ch = img_f[valid, 0]
        g_ch = img_f[valid, 1]
        r_ch = img_f[valid, 2]
    else:
        b_ch = img_f[:, :, 0].ravel()
        g_ch = img_f[:, :, 1].ravel()
        r_ch = img_f[:, :, 2].ravel()
        
    if len(g_ch) < 100:
        return 1.0, 1.0, 1.0

    # Subtract baseline background offset per channel
    bg_b = np.percentile(b_ch, 2.0)
    bg_g = np.percentile(g_ch, 2.0)
    bg_r = np.percentile(r_ch, 2.0)
    b_sub = np.maximum(0.0, b_ch - bg_b)
    g_sub = np.maximum(0.0, g_ch - bg_g)
    r_sub = np.maximum(0.0, r_ch - bg_r)
    
    # Identify bright non-saturated star pixels (between 99.5th and 99.98th percentile, <= 60000 ADU)
    p_high = np.percentile(g_sub, 99.5)
    p_sat = np.percentile(g_sub, 99.99)
    star_idx = (g_sub >= p_high) & (g_sub <= min(p_sat, 60000.0))
    if np.sum(star_idx) < 100:
        star_idx = g_sub >= p_high
        
    star_b = np.sum(b_sub[star_idx])
    star_g = np.sum(g_sub[star_idx])
    star_r = np.sum(r_sub[star_idx])
    
    if star_b > 0 and star_r > 0 and star_g > 0:
        kw_r = float(np.clip(star_g / star_r, 0.5, 3.5))
        kw_b = float(np.clip(star_g / star_b, 0.5, 3.5))
    else:
        kw_r, kw_b = 1.0, 1.0
        
    return kw_r, 1.0, kw_b

def apply_color_calibration(img_bgr, r_gain=1.0, g_gain=1.0, b_gain=1.0, 
                             scnr_amount=0.0, saturation=1.0, neutralize_bg=True, 
                             ccm_matrix=None, apply_scurve=False, scurve_strength=1.0,
                             mask=None, target_bg_adu=1500.0):
    """
    Applies comprehensive astrophotography color calibration:
    1. Background Neutralization (aligns dark black/gray baseline across R, G, B channels)
    2. White balance / Sensor QE channel gains (R, G, B) + optional 3x3 Color Correction Matrix (CCM)
    3. SCNR (Subtractive Chromatic Noise Reduction to eliminate green Bayer cast)
    4. Color Saturation enhancement in HSV space
    5. Photographic S-Curve Tone Mapping for deep sky contrast
    Returns 16-bit uint16 image (0..65535).
    """
    if img_bgr is None:
        return None
        
    img_f = img_bgr.astype(np.float32)
    
    # 1. Background Neutralization
    if neutralize_bg:
        if mask is not None and np.any(mask > 128):
            valid = mask > 128
            bg_b = np.percentile(img_f[valid, 0], 2.0)
            bg_g = np.percentile(img_f[valid, 1], 2.0)
            bg_r = np.percentile(img_f[valid, 2], 2.0)
        else:
            bg_b = np.percentile(img_f[:, :, 0], 2.0)
            bg_g = np.percentile(img_f[:, :, 1], 2.0)
            bg_r = np.percentile(img_f[:, :, 2], 2.0)
            
        b = np.maximum(0.0, img_f[:, :, 0] - bg_b)
        g = np.maximum(0.0, img_f[:, :, 1] - bg_g)
        r = np.maximum(0.0, img_f[:, :, 2] - bg_r)
        pedestal = float(target_bg_adu)
    else:
        b = img_f[:, :, 0]
        g = img_f[:, :, 1]
        r = img_f[:, :, 2]
        pedestal = 0.0
        
    bgr_temp = np.stack([b, g, r], axis=2)
    
    # 2. Scale channels by gains and optional 3x3 CCM
    if ccm_matrix is not None:
        wb_diag_rgb = np.diag([float(r_gain), float(g_gain), float(b_gain)]).astype(np.float32)
        M_rgb = np.asarray(ccm_matrix, dtype=np.float32) @ wb_diag_rgb
        P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float32)
        M_bgr = P @ M_rgb @ P
        transformed = cv2.transform(bgr_temp, M_bgr)
        b = transformed[:, :, 0]
        g = transformed[:, :, 1]
        r = transformed[:, :, 2]
    else:
        b = b * float(b_gain)
        g = g * float(g_gain)
        r = r * float(r_gain)
    
    # 3. SCNR (Green noise suppression)
    if scnr_amount > 0.0:
        excess_g = np.maximum(0.0, g - np.maximum(r, b))
        g = g - float(scnr_amount) * excess_g
        
    # 4. Re-add neutral background baseline
    if neutralize_bg:
        b = b + pedestal
        g = g + pedestal
        r = r + pedestal
        
    out = np.stack([b, g, r], axis=2)
    out = np.clip(out, 0.0, 65535.0)
    
    # 5. Color Saturation boost in HSV color space
    if abs(saturation - 1.0) > 0.01 and out.shape[0] > 0 and out.shape[1] > 0:
        out_norm = (out / 65535.0).astype(np.float32)
        hsv = cv2.cvtColor(out_norm, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(saturation), 0.0, 1.0)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) * 65535.0
        out = np.clip(out, 0.0, 65535.0)
        
    out_16 = out.astype(np.uint16)
    
    # 6. Photographic S-Curve Tone Mapping (if enabled)
    if apply_scurve and scurve_strength > 0.0:
        out_16 = apply_s_curve(out_16, strength=scurve_strength)
        
    return out_16


def align_single_frame(path, ref_img_for_align_8bit, mask, contrast_threshold, edge_threshold, sigma, transform_type, freeze_ground, gamma):
    try:
        img_raw = load_image(path)
        if img_raw is None:
            return None, None, "Could not read image file."
        
        # Apply gamma and convert to 8-bit for fast, robust star & feature detection
        img_corrected = apply_gamma(img_raw, gamma)
        if img_corrected.dtype == np.uint16:
            img_for_align_8bit = (img_corrected >> 8).astype(np.uint8)
        else:
            img_for_align_8bit = img_corrected
            
        h, w, c = img_raw.shape
        
        # 1. Align Sky
        _, H_sky = detect_and_align(
            ref_img_for_align_8bit, img_for_align_8bit, mask, align_sky=True,
            contrast_threshold=contrast_threshold,
            edge_threshold=edge_threshold,
            sigma=sigma,
            transform_type=transform_type
        )
        # Warping is applied to the full-precision 16-bit raw image (img_raw)
        if H_sky.shape == (2, 3):
            sky_warped_raw = cv2.warpAffine(img_raw, H_sky, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        else:
            sky_warped_raw = cv2.warpPerspective(img_raw, H_sky, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # 2. Align Ground
        ground_warped_raw = None
        has_ground = np.any(mask == 0)
        if has_ground and not freeze_ground:
            _, H_ground = detect_and_align(
                ref_img_for_align_8bit, img_for_align_8bit, mask, align_sky=False,
                contrast_threshold=contrast_threshold,
                edge_threshold=edge_threshold,
                sigma=sigma,
                transform_type=transform_type
            )
            # Warping is applied to the full-precision 16-bit raw image (img_raw)
            if H_ground.shape == (2, 3):
                ground_warped_raw = cv2.warpAffine(img_raw, H_ground, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
            else:
                ground_warped_raw = cv2.warpPerspective(img_raw, H_ground, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
            
        return sky_warped_raw, ground_warped_raw, None
    except Exception as e:
        return None, None, f"Alignment failed: {str(e)}"

def stack_small_chunk(chunk_imgs, stack_mode, remove_trails, sigma_factor=2.5):
    """
    Stacks a small chunk of N slices (each of shape H_chunk, W, C) along axis 0 in float32 precision.
    Returns 16-bit uint16 chunk.
    """
    arr = np.stack(chunk_imgs, axis=0).astype(np.float32)
    if remove_trails:
        if arr.shape[0] < 3:
            return np.clip(np.mean(arr, axis=0), 0, 65535).astype(np.uint16)
        med = np.median(arr, axis=0)
        abs_dev = np.abs(arr - med)
        mad = np.median(abs_dev, axis=0)
        mad = np.where(mad < 1.0, 1.0, mad)
        threshold = sigma_factor * 1.4826 * mad
        outlier_mask = (arr - med) > threshold
        masked_arr = np.where(outlier_mask, 0.0, arr)
        valid_counts = np.sum(~outlier_mask, axis=0)
        valid_counts = np.where(valid_counts < 1, 1, valid_counts)
        mean_img = np.sum(masked_arr, axis=0) / valid_counts
        return np.clip(mean_img, 0, 65535).astype(np.uint16)
    elif stack_mode == 'median':
        return np.clip(np.median(arr, axis=0), 0, 65535).astype(np.uint16)
    else:
        return np.clip(np.mean(arr, axis=0), 0, 65535).astype(np.uint16)

def stack_parallel_chunks(img_list, stack_mode, remove_trails, progress_callback=None, phase_name="sky"):
    """
    Memory-efficient multi-core stacking of 16-bit images.
    Avoids monolithic 4D array allocations by:
    1. Using lightweight streaming accumulation for standard average stacking.
    2. Using dynamic height-sliced streaming for median and sigma-clipping outlier rejection.
    """
    if not img_list:
        return None
        
    num_images = len(img_list)
    if num_images == 1:
        return img_list[0].astype(np.uint16)

    h, w, c = img_list[0].shape
    
    # 1. Ultra-fast and memory-minimal accumulator for Average stacking without trail rejection
    if stack_mode == 'average' and not remove_trails:
        acc = np.zeros((h, w, c), dtype=np.float64)
        for i, img in enumerate(img_list):
            acc += img.astype(np.float64)
            if progress_callback and (i % 5 == 0 or i == num_images - 1):
                progress_callback(i + 1, num_images, f"Averaging {phase_name} frame {i+1}/{num_images}...")
        mean_img = np.clip(acc / float(num_images), 0, 65535).astype(np.uint16)
        return mean_img

    # 2. Dynamic slice-based multi-threading for Median / Sigma-clipping
    # Target chunk memory <= 60MB per worker to avoid any OS memory allocation limits
    target_chunk_bytes = 60 * 1024 * 1024
    row_bytes = num_images * w * c * 4  # 4 bytes per float32
    chunk_h = max(16, min(128, target_chunk_bytes // max(1, row_bytes)))
    
    ranges = []
    y = 0
    while y < h:
        y_next = min(h, y + chunk_h)
        ranges.append((y, y_next))
        y = y_next
        
    total_chunks = len(ranges)
    stacked_img = np.empty((h, w, c), dtype=np.uint16)
    
    max_workers = min(4, max(1, multiprocessing.cpu_count() - 1))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for y0, y1 in ranges:
            chunk_slices = [img[y0:y1, :, :] for img in img_list]
            future = executor.submit(stack_small_chunk, chunk_slices, stack_mode, remove_trails, 2.5)
            futures[future] = (y0, y1)
            
        completed = 0
        for future in concurrent.futures.as_completed(futures.keys()):
            y0, y1 = futures[future]
            stacked_img[y0:y1, :, :] = future.result()
            completed += 1
            if progress_callback and (completed % max(1, total_chunks // 10) == 0 or completed == total_chunks):
                progress_callback(completed, total_chunks, f"Stacking {phase_name} (slice {completed}/{total_chunks})...")
                
    return stacked_img

def load_image(path):
    """
    Robustly loads images in full 16-bit high dynamic range (uint16, 0..65535).
    Supports 16-bit astronomical FITS, RAW files (CR2/CR3/NEF/ARW/DNG), 16-bit TIFFs,
    and standard 8-bit image formats (upscaled cleanly to 16-bit).
    """
    ext = os.path.splitext(path)[1].lower()
    
    if ext in ['.fit', '.fits']:
        try:
            from astropy.io import fits
            with fits.open(path) as hdul:
                # Find HDU with data
                hdu = hdul[0]
                data = hdu.data
                if data is None and len(hdul) > 1:
                    hdu = hdul[1]
                    data = hdu.data
                
                header = hdu.header
                
                # Check for Bayer pattern in header
                bayer_pat = None
                for key in ['BAYERPAT', 'BAYER', 'COLORTYP', 'DEBAYER']:
                    if key in header:
                        bayer_pat = str(header[key]).upper().strip()
                        break
                
                # Robustly normalize 16-bit / floating point data to 16-bit (0..65535) using percentiles to ignore hot pixels
                p_min = np.percentile(data, 0.01)
                p_max = np.percentile(data, 99.95)
                if p_max <= p_min:
                    p_min, p_max = float(data.min()), float(data.max())
                
                if p_max > p_min:
                    normalized = np.clip((data.astype(np.float32) - p_min) / (p_max - p_min) * 65535.0, 0, 65535).astype(np.uint16)
                else:
                    normalized = np.zeros_like(data, dtype=np.uint16)
                
                # Handle 3D color FITS (typically channels, height, width or vice versa)
                if len(normalized.shape) == 3:
                    if normalized.shape[0] in [3, 4]:
                        normalized = np.transpose(normalized, (1, 2, 0))
                    if normalized.shape[2] == 3:
                        return cv2.cvtColor(normalized, cv2.COLOR_RGB2BGR)
                    elif normalized.shape[2] == 4:
                        return cv2.cvtColor(normalized, cv2.COLOR_RGBA2BGR)
                
                # Handle 2D FITS (either Mono or Bayer Color raw)
                if len(normalized.shape) == 2:
                    if bayer_pat:
                        code = None
                        if 'RGGB' in bayer_pat:
                            code = cv2.COLOR_BayerRG2RGB
                        elif 'BGGR' in bayer_pat:
                            code = cv2.COLOR_BayerBG2RGB
                        elif 'GRBG' in bayer_pat:
                            code = cv2.COLOR_BayerGR2RGB
                        elif 'GBRG' in bayer_pat:
                            code = cv2.COLOR_BayerGB2RGB
                        
                        if code is not None:
                            try:
                                return cv2.cvtColor(normalized, code)
                            except Exception as db_err:
                                print(f"Debayering FITS failed: {db_err}")
                    
                    # Fallback to monochrome converted to BGR
                    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            print(f"Failed to load FITS file {path} with astropy: {e}")
            
    elif ext == '.dng' or ext in ['.nef', '.cr2', '.cr3', '.arw', '.dcr']:
        try:
            import rawpy
            with rawpy.imread(path) as raw:
                # Postprocess with output_bps=16 for native full 16-bit color depth
                rgb_16 = raw.postprocess(output_bps=16, use_camera_wb=True, half_size=False, no_auto_bright=True)
                return cv2.cvtColor(rgb_16, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Failed to load RAW/DNG file {path} with rawpy: {e}")
            
    # Standard image loading with 16-bit TIFF check
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None:
        if img.dtype == np.uint16:
            if len(img.shape) == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img
        elif img.dtype == np.uint8:
            # Upscale 8-bit image to 16-bit cleanly
            img_16 = (img.astype(np.uint16) * 257)
            if len(img_16.shape) == 2:
                return cv2.cvtColor(img_16, cv2.COLOR_GRAY2BGR)
            return img_16
        return img
    
    fallback = cv2.imread(path)
    if fallback is not None and fallback.dtype == np.uint8:
        return (fallback.astype(np.uint16) * 257)
    return fallback

def apply_gamma(img, gamma=1.0):
    """
    Applies gamma correction to adjust image brightness in 16-bit (or 8-bit).
    gamma < 1.0 brightens shadows/midtones (ideal for dark FITS/RAW files).
    gamma > 1.0 darkens shadows/midtones.
    """
    if img is None or abs(gamma - 1.0) < 0.005:
        return img
    
    if img.dtype == np.uint16:
        # Build 16-bit lookup table for ultra-fast indexing (65536 entries)
        table = np.array([((i / 65535.0) ** gamma) * 65535.0 for i in range(65536)]).clip(0, 65535).astype(np.uint16)
        return table[img]
    else:
        table = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)]).clip(0, 255).astype(np.uint8)
        return cv2.LUT(img, table)

def feather_mask(mask, radius):
    """
    Feathers/blurs a binary mask to create smooth, progressive transitions.
    Optimized using downsampling for large blur radii to prevent lag and enhance smoothness.
    """
    if radius <= 0:
        return mask.astype(np.float32) / 255.0
        
    h, w = mask.shape[:2]
    # If the radius is large or the mask is huge, blur a downscaled version and upscale it.
    # The bilinear upscaling creates an even softer, more progressive gradient.
    if radius > 30 or max(h, w) > 2000:
        scale = 1000.0 / max(h, w)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            small_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Scale the radius down correspondingly
            scaled_radius = max(1, int(radius * scale))
            ksize = int(scaled_radius * 2) + 1
            blurred_small = cv2.GaussianBlur(small_mask, (ksize, ksize), 0)
            
            # Upscale back to original size using bilinear interpolation
            blurred = cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)
            return blurred.astype(np.float32) / 255.0

    # Fallback for small sizes / small radii
    ksize = int(radius * 2) + 1
    blurred = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    return blurred.astype(np.float32) / 255.0

def stack_images(image_paths, mask=None, stack_mode='average', feather_radius=10, 
                 contrast_threshold=0.04, edge_threshold=10.0, sigma=1.6,
                 transform_type="affine", freeze_ground=False, gamma_sky=1.0, gamma_ground=1.0,
                 progress_callback=None, cancel_event=None, remove_trails=False):
    """
    Stacks a list of images by separately aligning sky and ground, and then blending.
    Maintains full 16-bit dynamic range precision throughout the entire pipeline.
    
    Uses the middle image of the sequence as the reference frame to minimize overall distortion.
    
    freeze_ground: if True, skips ground stacking and uses the landscape from the reference frame.
    
    Returns: (final_image_16bit, success_count, failed_reports, sky_stack_16bit, ground_stack_16bit)
    """
    num_images = len(image_paths)
    if num_images == 0:
        return None, 0, []

    # Choose the middle frame as reference to reduce coordinate warp distortion
    ref_idx = num_images // 2
    ref_path = image_paths[ref_idx]
    ref_img_raw = load_image(ref_path)
    if ref_img_raw is None:
        raise ValueError(f"Could not load reference image: {ref_path}")
        
    ref_img_corrected = apply_gamma(ref_img_raw, gamma_sky)
    if ref_img_corrected.dtype == np.uint16:
        ref_img_for_align_8bit = (ref_img_corrected >> 8).astype(np.uint8)
    else:
        ref_img_for_align_8bit = ref_img_corrected
    
    # If no mask is provided, treat the entire image as sky
    if mask is None:
        mask = np.ones((ref_img_raw.shape[0], ref_img_raw.shape[1]), dtype=np.uint8) * 255
  
    sky_list = []
    ground_list = []
    failed_reports = []
    
    has_ground = np.any(mask == 0)
 
    # Pre-populate lists to preserve indexing order
    for i in range(num_images):
        sky_list.append(None)
        ground_list.append(None)
 
    # Insert reference frame directly (uncorrected 16-bit uint16)
    sky_list[ref_idx] = ref_img_raw
    ground_list[ref_idx] = ref_img_raw
 
    if progress_callback:
        progress_callback(1, num_images, f"Selected middle frame as reference: {os.path.basename(ref_path)}")
 
    futures = {}
    max_workers = min(4, max(1, multiprocessing.cpu_count() - 1))
    
    if progress_callback:
        progress_callback(1, num_images, f"Spawning parallel alignment workers (cores used: {max_workers})...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i in range(num_images):
            if i == ref_idx:
                continue
                
            path = image_paths[i]
            future = executor.submit(
                align_single_frame,
                path, ref_img_for_align_8bit, mask,
                contrast_threshold, edge_threshold, sigma,
                transform_type, freeze_ground, gamma_sky
            )
            futures[future] = (i, path)
            
        completed_count = 0
        total_to_process = num_images - 1
        
        for future in concurrent.futures.as_completed(futures.keys()):
            if cancel_event is not None and cancel_event.is_set():
                executor.shutdown(wait=False, cancel_futures=True)
                raise InterruptedError("Stacking cancelled by user.")
                
            idx, path = futures[future]
            parent_dir = os.path.basename(os.path.dirname(path))
            filename = f"[{parent_dir}] {os.path.basename(path)}" if parent_dir else os.path.basename(path)
            completed_count += 1
            
            try:
                sky_warped, ground_warped, err_msg = future.result()
                if err_msg:
                    failed_reports.append({"file": filename, "error": err_msg})
                    if progress_callback:
                        progress_callback(completed_count + 1, num_images, f"Frame {completed_count}/{total_to_process} finished: {filename} (Failed: {err_msg})")
                else:
                    if sky_warped is not None:
                        sky_list[idx] = sky_warped.astype(np.uint16)
                    if ground_warped is not None:
                        ground_list[idx] = ground_warped.astype(np.uint16)
                        
                    if progress_callback:
                        progress_callback(completed_count + 1, num_images, f"Frame {completed_count}/{total_to_process} finished: {filename} (Aligned Sky/Ground successfully)")
            except Exception as e:
                failed_reports.append({"file": filename, "error": f"Process error: {str(e)}"})
                if progress_callback:
                    progress_callback(completed_count + 1, num_images, f"Frame {completed_count}/{total_to_process} finished: {filename} (Failed: {str(e)})")

    # Filter out None values (failed frames)
    valid_sky_list = [img for img in sky_list if img is not None]
    valid_ground_list = [img for img in ground_list if img is not None]

    success_count = len(valid_sky_list)

    # 3. Stack Sky (in full 16-bit dynamic range)
    if progress_callback:
        progress_callback(num_images, num_images, "Stacking sky frames (16-bit)...")
    
    sky_stack = stack_parallel_chunks(valid_sky_list, stack_mode, remove_trails, progress_callback, phase_name="sky")

    # 4. Stack Ground (in full 16-bit dynamic range)
    if has_ground:
        if freeze_ground:
            # Tack-sharp single exposure landscape from reference frame (uncorrected 16-bit)
            ground_stack = ref_img_raw
        else:
            if progress_callback:
                progress_callback(num_images, num_images, "Stacking ground frames (16-bit)...")
            ground_stack = stack_parallel_chunks(valid_ground_list, stack_mode, remove_trails, progress_callback, phase_name="ground")
    else:
        ground_stack = sky_stack

    # 5. Composite Sky and Ground in 16-bit
    if progress_callback:
        progress_callback(num_images, num_images, "Blending sky and ground (16-bit)...")
    
    sky_stack_gamma = apply_gamma(sky_stack, gamma_sky)
    ground_stack_gamma = apply_gamma(ground_stack, gamma_ground)
    
    f_mask = feather_mask(mask, feather_radius)
    f_mask_3d = np.expand_dims(f_mask, axis=2)

    final_img = (sky_stack_gamma.astype(np.float32) * f_mask_3d + 
                 ground_stack_gamma.astype(np.float32) * (1.0 - f_mask_3d))
    
    final_img = np.clip(final_img, 0, 65535).astype(np.uint16)

    return final_img, success_count, failed_reports, sky_stack, ground_stack
