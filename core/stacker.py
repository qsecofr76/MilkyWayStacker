import os
import cv2
import numpy as np
from core.aligner import detect_and_align

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
            
    elif ext == '.dng' or ext in ['.nef', '.cr2', '.arw', '.dcr']:
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
                 transform_type="affine", freeze_ground=False, gamma=1.0, progress_callback=None):
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
    # (will fill them in the loop below)
    for i in range(num_images):
        sky_list.append(None)
        ground_list.append(None)
 
    # Insert reference frame directly
    sky_list[ref_idx] = ref_img.astype(np.float32)
    ground_list[ref_idx] = ref_img.astype(np.float32)
 
    if progress_callback:
        progress_callback(1, num_images, f"Selected middle frame as reference: {os.path.basename(ref_path)}")
 
    for i in range(num_images):
        if i == ref_idx:
            continue
            
        path = image_paths[i]
        filename = os.path.basename(path)
        img = load_image(path)
        if img is None:
            failed_reports.append({"file": filename, "error": "Could not read image file."})
            continue
        img = apply_gamma(img, gamma)

        frame_failed = False
        
        # 1. Align Sky
        if progress_callback:
            progress_callback(i + 1, num_images, f"Aligning sky for frame {i+1}/{num_images} ({filename})...")
        
        try:
            sky_warped, H_sky = detect_and_align(
                ref_img, img, mask, align_sky=True,
                contrast_threshold=contrast_threshold,
                edge_threshold=edge_threshold,
                sigma=sigma,
                transform_type=transform_type
            )
            sky_list[i] = sky_warped.astype(np.float32)
        except Exception as e:
            failed_reports.append({"file": filename, "error": f"Sky alignment failed: {str(e)}"})
            frame_failed = True

        # 2. Align Ground (only if ground is NOT frozen and there is a ground region)
        if not frame_failed and has_ground:
            if freeze_ground:
                # Ground is frozen; do not compute alignment or warp for target frames
                pass
            else:
                if progress_callback:
                    progress_callback(i + 1, num_images, f"Aligning ground for frame {i+1}/{num_images} ({filename})...")
                try:
                    ground_warped, H_ground = detect_and_align(
                        ref_img, img, mask, align_sky=False,
                        contrast_threshold=contrast_threshold,
                        edge_threshold=edge_threshold,
                        sigma=sigma,
                        transform_type=transform_type
                    )
                    ground_list[i] = ground_warped.astype(np.float32)
                except Exception as e:
                    failed_reports.append({"file": filename, "error": f"Landscape alignment failed: {str(e)}"})
                    # If ground alignment fails, we remove the corresponding sky warp we just added
                    sky_list[i] = None
                    frame_failed = True

    # Filter out None values (failed frames)
    valid_sky_list = [img for img in sky_list if img is not None]
    valid_ground_list = [img for img in ground_list if img is not None]

    success_count = len(valid_sky_list)

    # 3. Stack Sky
    if progress_callback:
        progress_callback(num_images, num_images, "Stacking sky frames...")
    
    if stack_mode == 'median':
        sky_stack = np.median(valid_sky_list, axis=0).astype(np.uint8)
    else: # average
        sky_stack = np.mean(valid_sky_list, axis=0).astype(np.uint8)

    # 4. Stack Ground
    if has_ground:
        if freeze_ground:
            # Tack-sharp single exposure landscape from reference frame
            ground_stack = ref_img
        else:
            if progress_callback:
                progress_callback(num_images, num_images, "Stacking ground frames...")
            if stack_mode == 'median':
                ground_stack = np.median(valid_ground_list, axis=0).astype(np.uint8)
            else:
                ground_stack = np.mean(valid_ground_list, axis=0).astype(np.uint8)
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
