import os
import cv2
import numpy as np
import concurrent.futures
import multiprocessing
from core.aligner import detect_and_align

def align_single_frame(path, ref_img, mask, contrast_threshold, edge_threshold, sigma, transform_type, freeze_ground, gamma):
    try:
        img = load_image(path)
        if img is None:
            return None, None, "Could not read image file."
        img = apply_gamma(img, gamma)
        
        # 1. Align Sky
        sky_warped, H_sky = detect_and_align(
            ref_img, img, mask, align_sky=True,
            contrast_threshold=contrast_threshold,
            edge_threshold=edge_threshold,
            sigma=sigma,
            transform_type=transform_type
        )
        
        # 2. Align Ground
        ground_warped = None
        has_ground = np.any(mask == 0)
        if has_ground and not freeze_ground:
            ground_warped, H_ground = detect_and_align(
                ref_img, img, mask, align_sky=False,
                contrast_threshold=contrast_threshold,
                edge_threshold=edge_threshold,
                sigma=sigma,
                transform_type=transform_type
            )
            
        return sky_warped, ground_warped, None
    except Exception as e:
        return None, None, f"Alignment failed: {str(e)}"

def stack_chunk(chunk, stack_mode, remove_trails, sigma_factor=2.5):
    """
    Stacks a chunk of shape (N, H_i, W, C) along axis 0 using the selected stacking mode.
    """
    if remove_trails:
        if chunk.shape[0] < 3:
            return np.mean(chunk, axis=0).astype(np.uint8)
        med = np.median(chunk, axis=0)
        abs_dev = np.abs(chunk - med)
        mad = np.median(abs_dev, axis=0)
        mad = np.where(mad < 1.0, 1.0, mad)
        threshold = sigma_factor * 1.4826 * mad
        outlier_mask = (chunk - med) > threshold
        masked_arr = np.where(outlier_mask, 0.0, chunk)
        valid_counts = np.sum(~outlier_mask, axis=0)
        valid_counts = np.where(valid_counts < 1, 1, valid_counts)
        mean_img = np.sum(masked_arr, axis=0) / valid_counts
        return np.clip(mean_img, 0, 255).astype(np.uint8)
    elif stack_mode == 'median':
        return np.median(chunk, axis=0).astype(np.uint8)
    else:
        return np.mean(chunk, axis=0).astype(np.uint8)

def stack_parallel_chunks(img_list, stack_mode, remove_trails, progress_callback=None, phase_name="sky"):
    """
    Stacks list of images by partitioning them along height into parallel chunks
    to utilize multiple CPU cores for CPU-bound median / sigma-clipping operations.
    """
    if not img_list:
        return None
        
    num_images = len(img_list)
    if num_images == 1:
        return img_list[0].astype(np.uint8)

    arr = np.stack(img_list, axis=0)  # Shape: (N, H, W, C)
    
    # Cap parallel workers at min(4, max(1, CPU cores - 1)) to prevent over-subscription
    num_chunks = min(4, max(1, multiprocessing.cpu_count() - 1))
    
    if num_chunks <= 1:
        return stack_chunk(arr, stack_mode, remove_trails)

    # Split the array along the height axis (axis 1)
    chunks = np.array_split(arr, num_chunks, axis=1)
    
    futures = {}
    results = [None] * num_chunks
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_chunks) as executor:
        for idx, chunk in enumerate(chunks):
            future = executor.submit(stack_chunk, chunk, stack_mode, remove_trails, 2.5)
            futures[future] = idx
            
        completed = 0
        for future in concurrent.futures.as_completed(futures.keys()):
            idx = futures[future]
            results[idx] = future.result()
            completed += 1
            if progress_callback:
                progress_callback(num_images, num_images, f"Stacking {phase_name} (chunk {completed}/{num_chunks} finished)...")
                
    stacked_img = np.concatenate(results, axis=0)
    return stacked_img

def load_image(path):
    """
    Robustly loads images supporting standard formats (JPG, PNG, TIFF)
    as well as RAW DNG and FITS astronomical files.
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
                
                # Normalize data to 0-255 range
                data_min = data.min()
                data_max = data.max()
                if data_max > data_min:
                    normalized = ((data - data_min) / (data_max - data_min) * 255.0).astype(np.uint8)
                else:
                    normalized = np.zeros_like(data, dtype=np.uint8)
                
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
                            code = cv2.COLOR_BayerRG2BGR
                        elif 'BGGR' in bayer_pat:
                            code = cv2.COLOR_BayerBG2BGR
                        elif 'GRBG' in bayer_pat:
                            code = cv2.COLOR_BayerGR2BGR
                        elif 'GBRG' in bayer_pat:
                            code = cv2.COLOR_BayerGB2BGR
                        
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
                rgb = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Failed to load RAW/DNG file {path} with rawpy: {e}")
            
    return cv2.imread(path)

def apply_gamma(img, gamma=1.0):
    """
    Applies gamma correction to adjust image brightness.
    gamma < 1.0 brightens shadows/midtones (ideal for dark FITS/RAW files).
    gamma > 1.0 darkens shadows/midtones.
    """
    if img is None or abs(gamma - 1.0) < 0.01:
        return img
    # Build a lookup table (LUT)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def feather_mask(mask, radius):
    """
    Feathers/blurs a binary mask to create smooth transitions.
    radius: blur radius in pixels.
    """
    if radius <= 0:
        return mask.astype(np.float32) / 255.0
    
    # Ensure radius is odd
    ksize = int(radius * 2) + 1
    blurred = cv2.GaussianBlur(mask, (ksize, ksize), 0)
    return blurred.astype(np.float32) / 255.0

def stack_images(image_paths, mask=None, stack_mode='average', feather_radius=10, 
                 contrast_threshold=0.04, edge_threshold=10.0, sigma=1.6,
                 transform_type="affine", freeze_ground=False, gamma=1.0, progress_callback=None, cancel_event=None, remove_trails=False):
    """
    Stacks a list of images by separately aligning sky and ground, and then blending.
    Skips images where alignment fails and records the error details.
    
    Uses the middle image of the sequence as the reference frame to minimize overall distortion.
    
    freeze_ground: if True, skips ground stacking and uses the landscape from the reference frame.
    
    Returns: (final_image, success_count, failed_reports)
    """
    num_images = len(image_paths)
    if num_images == 0:
        return None, 0, []

    # Choose the middle frame as reference to reduce coordinate warp distortion
    ref_idx = num_images // 2
    ref_path = image_paths[ref_idx]
    ref_img = load_image(ref_path)
    if ref_img is None:
        raise ValueError(f"Could not load reference image: {ref_path}")
        
    ref_img = apply_gamma(ref_img, gamma)
    
    # If no mask is provided, treat the entire image as sky
    if mask is None:
        mask = np.ones((ref_img.shape[0], ref_img.shape[1]), dtype=np.uint8) * 255
  
    sky_list = []
    ground_list = []
    failed_reports = []
    
    has_ground = np.any(mask == 0)
 
    # Pre-populate lists to preserve indexing order
    for i in range(num_images):
        sky_list.append(None)
        ground_list.append(None)
 
    # Insert reference frame directly
    sky_list[ref_idx] = ref_img.astype(np.float32)
    ground_list[ref_idx] = ref_img.astype(np.float32)
 
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
                path, ref_img, mask,
                contrast_threshold, edge_threshold, sigma,
                transform_type, freeze_ground, gamma
            )
            futures[future] = (i, path)
            
        completed_count = 0
        total_to_process = num_images - 1
        
        for future in concurrent.futures.as_completed(futures.keys()):
            if cancel_event is not None and cancel_event.is_set():
                executor.shutdown(wait=False, cancel_futures=True)
                raise InterruptedError("Stacking cancelled by user.")
                
            idx, path = futures[future]
            filename = os.path.basename(path)
            completed_count += 1
            
            try:
                sky_warped, ground_warped, err_msg = future.result()
                if err_msg:
                    failed_reports.append({"file": filename, "error": err_msg})
                    if progress_callback:
                        progress_callback(completed_count + 1, num_images, f"Frame {completed_count}/{total_to_process} finished: {filename} (Failed: {err_msg})")
                else:
                    if sky_warped is not None:
                        sky_list[idx] = sky_warped.astype(np.float32)
                    if ground_warped is not None:
                        ground_list[idx] = ground_warped.astype(np.float32)
                        
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

    # 3. Stack Sky
    if progress_callback:
        progress_callback(num_images, num_images, "Stacking sky frames...")
    
    sky_stack = stack_parallel_chunks(valid_sky_list, stack_mode, remove_trails, progress_callback, phase_name="sky")

    # 4. Stack Ground
    if has_ground:
        if freeze_ground:
            # Tack-sharp single exposure landscape from reference frame
            ground_stack = ref_img
        else:
            if progress_callback:
                progress_callback(num_images, num_images, "Stacking ground frames...")
            ground_stack = stack_parallel_chunks(valid_ground_list, stack_mode, remove_trails, progress_callback, phase_name="ground")
    else:
        ground_stack = sky_stack

    # 5. Composite Sky and Ground
    if progress_callback:
        progress_callback(num_images, num_images, "Blending sky and ground...")
    
    f_mask = feather_mask(mask, feather_radius)
    f_mask_3d = np.expand_dims(f_mask, axis=2)

    final_img = (sky_stack.astype(np.float32) * f_mask_3d + 
                 ground_stack.astype(np.float32) * (1.0 - f_mask_3d))
    
    final_img = np.clip(final_img, 0, 255).astype(np.uint8)

    return final_img, success_count, failed_reports
