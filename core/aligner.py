import cv2
import numpy as np
import astroalign as aa
import itertools
import time

def dilate_mask(mask, radius=30):
    """Dilates a binary mask to allow keypoint matching in slightly shifted frames."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(mask, kernel)

def erode_mask(mask, radius=25):
    """
    Erodes a binary mask to create an exclusion zone along the mask boundary.
    Optimized using downsampling for large erosion radii to keep performance ultra-fast.
    """
    if mask is None or radius <= 0:
        return mask
        
    h, w = mask.shape[:2]
    # If the radius is large or the mask is huge, erode a downscaled version and upscale it.
    if radius > 15 or max(h, w) > 2000:
        scale = 800.0 / max(h, w)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            small_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            scaled_radius = max(1, int(radius * scale))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (scaled_radius * 2 + 1, scaled_radius * 2 + 1))
            eroded_small = cv2.erode(small_mask, kernel)
            
            # Upscale back to original size (INTER_NEAREST is perfect for binary masks)
            return cv2.resize(eroded_small, (w, h), interpolation=cv2.INTER_NEAREST)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.erode(mask, kernel)

def detect_stars_centroids(img, mask=None, contrast_threshold=0.04, sigma=1.6, max_stars=150):
    """
    Detects star centroids in the image using dynamic thresholding based on the contrast slider.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mask is not None:
        gray = cv2.bitwise_and(gray, mask)
        
    # Apply Gaussian blur based on the sigma slider
    ksize = int(2 * round(2 * sigma) + 1)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    
    # Calculate sky statistics
    if mask is not None and np.any(mask > 0):
        sky_pixels = blurred[mask > 0]
        mean_val = np.mean(sky_pixels)
        std_val = np.std(sky_pixels)
    else:
        mean_val = np.mean(blurred)
        std_val = np.std(blurred)
        
    if np.isnan(mean_val) or np.isnan(std_val):
        mean_val = 127.0
        std_val = 20.0
        
    # Contrast threshold slider maps to standard deviation multiplier
    # Lower slider values (e.g. 0.01) detect fainter stars, higher values detect only bright stars
    multiplier = contrast_threshold * 100.0
    thresh_val = int(mean_val + multiplier * std_val)
    # Ensure threshold is at least slightly above background mean
    thresh_val = max(int(mean_val + 4), thresh_val)
    
    _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centroids = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 120 or area < 1:  # Filter out noise and massive bright regions
            continue
            
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = M["m10"] / M["m00"]
            cY = M["m01"] / M["m00"]
            centroids.append([cX, cY])
            
    if not centroids:
        return []
        
    centroids = np.array(centroids)
    
    # Sort centroids by pixel intensity (brightness)
    intensities = []
    for cX, cY in centroids:
        x_idx = min(gray.shape[1] - 1, max(0, int(cX)))
        y_idx = min(gray.shape[0] - 1, max(0, int(cY)))
        intensities.append(gray[y_idx, x_idx])
        
    sorted_idx = np.argsort(intensities)[::-1]
    return centroids[sorted_idx][:max_stars]

def detect_and_filter_stars(img, mask=None, contrast_threshold=0.04, sigma=1.6, max_stars=60):
    """
    Detects stars on the FULL image to avoid artificial mask boundary edge gradients,
    then filters them to only keep stars inside the sky mask, and distributes
    them evenly using a 6x6 spatial grid.
    """
    # 1. Detect on full image to prevent fake edge gradients
    all_stars = detect_stars_centroids(img, mask=None, contrast_threshold=contrast_threshold, sigma=sigma, max_stars=300)
    if len(all_stars) == 0:
        return np.empty((0, 2), dtype=np.float32)
        
    # 2. Filter to keep stars well inside the sky mask (applying 25px ridge exclusion zone)
    eroded_mask = erode_mask(mask, radius=25) if mask is not None else None
    
    filtered = []
    for pt in all_stars:
        x, y = int(pt[0]), int(pt[1])
        if eroded_mask is not None:
            if x >= 0 and x < eroded_mask.shape[1] and y >= 0 and y < eroded_mask.shape[0]:
                if eroded_mask[y, x] > 0:
                    filtered.append(pt)
        else:
            filtered.append(pt)
            
    if not filtered:
        return np.empty((0, 2), dtype=np.float32)
        
    filtered = np.array(filtered)
    
    # 3. Spatial Grid Binning (homogeneous coverage, no clustering)
    h, w = img.shape[:2]
    grid_rows, grid_cols = 6, 6
    cell_h = h / grid_rows
    cell_w = w / grid_cols
    
    bins = {}
    for pt in filtered:
        c_idx = int(pt[0] / cell_w)
        r_idx = int(pt[1] / cell_h)
        key = (r_idx, c_idx)
        if key not in bins:
            bins[key] = []
        bins[key].append(pt)
        
    # Take up to 2 stars from each cell to ensure grid distribution
    homogenized = []
    for key, cell_pts in bins.items():
        homogenized.extend(cell_pts[:2])
        
    return np.array(homogenized[:max_stars], dtype=np.float32)

def filter_landscape_contours(edges, min_length=45):
    """
    Removes small isolated edge segments (like stars or reflections) from the Canny edge map,
    retaining only long continuous outlines (like mountain ridges or roof boundaries).
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    filtered_edges = np.zeros_like(edges)
    for c in contours:
        length = cv2.arcLength(c, closed=False)
        if length > min_length:
            cv2.drawContours(filtered_edges, [c], -1, 255, thickness=1)
    return filtered_edges

def estimate_rigid_transform(src, dst):
    """
    Estimates a pure rigid/Euclidean transform (translation + rotation, NO scale or skew)
    between two sets of 2D points using SVD (Procrustes analysis).
    """
    if len(src) < 2:
        return None
    
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    
    src_c = src - src_mean
    dst_c = dst - dst_mean
    
    # Covariance matrix
    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Special reflection case
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
        
    t = dst_mean - R @ src_mean
    
    M = np.hstack((R, t.reshape(2, 1)))
    return M

def align_landscape_optical_flow(ref_img, target_img, mask, transform_type="affine"):
    """
    Aligns target_img landscape to ref_img using silhouette contour matching 
    via Fourier Phase Correlation on Canny edge maps, filtered to remove stars/reflections.
    Guarantees pure translation-only alignment (zero scale, zero skew, zero rotation deformation).
    Handles extremely large displacements (500px+) instantly and perfectly.
    """
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    
    # 1. Apply CLAHE to stretch the dynamic range and reveal details in dark shadows
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    gray_ref_enh = clahe.apply(gray_ref)
    gray_tgt_enh = clahe.apply(gray_target)
    
    # 2. Apply Bilateral Filter to smooth out sensor noise while preserving sharp silhouettes
    ref_smooth = cv2.bilateralFilter(gray_ref_enh, d=9, sigmaColor=50, sigmaSpace=50)
    tgt_smooth = cv2.bilateralFilter(gray_tgt_enh, d=9, sigmaColor=50, sigmaSpace=50)
    
    # 3. Compute edge maps using highly sensitive thresholds to capture faint profiles
    edges_ref = cv2.Canny(ref_smooth, 10, 30)
    edges_tgt = cv2.Canny(tgt_smooth, 10, 30)
    
    # 4. Mask the reference edge map to only keep contours in the ground region
    if mask is not None:
        ground_mask = cv2.bitwise_not(mask)
        # Apply a wide 75px exclusion zone to prevent selecting points near the mask transition boundary
        ground_mask = erode_mask(ground_mask, radius=75)
        edges_ref = cv2.bitwise_and(edges_ref, ground_mask)
        # Do NOT mask target edges because they have shifted, preventing alignment truncation mismatch
    else:
        ground_mask = None
        
    # 5. Filter out small contours (stars, reflections) - minimum length 80px for high robustness
    edges_ref = filter_landscape_contours(edges_ref, min_length=80)
    edges_tgt = filter_landscape_contours(edges_tgt, min_length=80)
    
    # 6. Convert to float32 for Fourier Transform
    ref_f = edges_ref.astype(np.float32)
    tgt_f = edges_tgt.astype(np.float32)
    
    # 7. Perform Phase Correlation with a Hanning Window to prevent edge leakage
    h, w = gray_ref.shape
    hann = cv2.createHanningWindow((w, h), cv2.CV_32F)
    
    try:
        (dx, dy), response = cv2.phaseCorrelate(ref_f, tgt_f, hann)
        # Construct translation matrix to warp Target -> Reference
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        return M
    except Exception as e:
        print(f"Contour phase correlation failed: {e}")
        return None

def detect_and_align(ref_img, target_img, mask=None, align_sky=True, 
                      contrast_threshold=0.04, edge_threshold=10.0, sigma=1.6,
                      transform_type="affine"):
    """
    Main alignment function.
    """
    h, w, c = ref_img.shape
    
    if align_sky:
        # Pre-detect and filter stars homogeneously
        ref_stars = detect_and_filter_stars(ref_img, mask, contrast_threshold=contrast_threshold, sigma=sigma, max_stars=50)
        target_stars = detect_and_filter_stars(target_img, mask, contrast_threshold=contrast_threshold, sigma=sigma, max_stars=50)
        
        if len(ref_stars) < 3 or len(target_stars) < 3:
            raise RuntimeError("Less than 3 stars detected in the sky region.")
            
        try:
            transf, (s_list, t_list) = aa.find_transform(target_stars, ref_stars)
            M = transf.params
            
            # Sanity check: camera lens zoom scale must be very close to 1.0 (no focal length changes)
            if M is not None:
                s_x = np.linalg.norm(M[0, :2])
                s_y = np.linalg.norm(M[1, :2])
                if abs(s_x - 1.0) > 0.03 or abs(s_y - 1.0) > 0.03:
                    raise ValueError(f"Scale anomaly (sx={s_x:.3f}, sy={s_y:.3f}). Rejected false star matches.")
        except Exception as e:
            raise RuntimeError(f"Astroalign failed: {str(e)}")
    else:
        M = align_landscape_optical_flow(ref_img, target_img, mask, transform_type)
        if M is None:
            raise RuntimeError("No landscape alignment features could be tracked between frames.")

    # Apply Lanczos4 interpolation for landscape (sharper details) and linear for sky
    flags = cv2.INTER_LANCZOS4 if not align_sky else cv2.INTER_LINEAR

    if M.shape == (2, 3):
        warped_img = cv2.warpAffine(target_img, M, (w, h), flags=flags, borderMode=cv2.BORDER_REFLECT)
    else:
        warped_img = cv2.warpPerspective(target_img, M, (w, h), flags=flags, borderMode=cv2.BORDER_REFLECT)
        
    return warped_img, M

def draw_outlined_text(img, text, position, font_scale=0.8, color=(0, 255, 0), thickness=2):
    """
    Draws text with a black outline to make it highly legible on dark/bright regions.
    """
    # Black outline
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 3, lineType=cv2.LINE_AA)
    # Foreground color
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, lineType=cv2.LINE_AA)

def get_debug_matches_image(ref_img, target_img, mask=None, align_sky=True,
                            contrast_threshold=0.04, edge_threshold=10.0, sigma=1.6):
    """
    Diagnostic matching visualization drawing matching numbers on aligned points
    with a safety exclusion zone applied along the mask boundary.
    """
    h, w, c = ref_img.shape
    combined = np.hstack((ref_img.copy(), target_img.copy()))
    
    # Dynamic scaling for text labels and shapes based on image resolution
    font_scale = max(0.6, w / 1600.0)
    title_scale = max(1.0, w / 1200.0)
    thickness = max(2, int(w / 500.0))
    circle_radius = max(6, int(w / 200.0))
    circle_thickness = max(2, int(w / 800.0))
    
    if align_sky:
        ref_stars = detect_and_filter_stars(ref_img, mask, contrast_threshold=contrast_threshold, sigma=sigma, max_stars=50)
        target_stars = detect_and_filter_stars(target_img, mask, contrast_threshold=contrast_threshold, sigma=sigma, max_stars=50)
        
        try:
            transf, (s_list, t_list) = aa.find_transform(target_stars, ref_stars)
            for idx, (s_pt, t_pt) in enumerate(zip(s_list[:20], t_list[:20])):
                rx, ry = int(t_pt[0]), int(t_pt[1])
                tx, ty = int(s_pt[0]), int(s_pt[1])
                
                label = f"{idx + 1}"
                
                # Left image
                cv2.circle(combined, (rx, ry), circle_radius, (0, 255, 0), circle_thickness)
                draw_outlined_text(combined, label, (rx + circle_radius + 4, ry - circle_radius - 4), font_scale=font_scale, color=(0, 255, 0), thickness=thickness)
                
                # Right image
                cv2.circle(combined, (tx + w, ty), circle_radius, (0, 255, 0), circle_thickness)
                draw_outlined_text(combined, label, (tx + w + circle_radius + 4, ty - circle_radius - 4), font_scale=font_scale, color=(0, 255, 0), thickness=thickness)
                
            draw_outlined_text(combined, f"Sky Star Matching (Astroalign). Top 20 matches. [Grid sampling]", (20, int(40 * title_scale)), 
                               font_scale=title_scale, color=(255, 255, 255), thickness=thickness)
        except Exception as e:
            draw_outlined_text(combined, f"Astroalign failed: {str(e)}", (20, int(40 * title_scale)), 
                               font_scale=title_scale, color=(0, 0, 255), thickness=thickness)
    else:
        # Ground debug (contour-based rigid registration on first & last frames)
        gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        gray_target = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        
        # 1. Apply CLAHE to stretch the dynamic range and reveal details in dark shadows
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        gray_ref_enh = clahe.apply(gray_ref)
        gray_tgt_enh = clahe.apply(gray_target)
        
        # 2. Apply Bilateral Filter to smooth out sensor noise while preserving sharp silhouettes
        ref_smooth = cv2.bilateralFilter(gray_ref_enh, d=9, sigmaColor=50, sigmaSpace=50)
        tgt_smooth = cv2.bilateralFilter(gray_tgt_enh, d=9, sigmaColor=50, sigmaSpace=50)
        
        # 3. Compute edge maps using highly sensitive thresholds to capture faint profiles
        edges_ref = cv2.Canny(ref_smooth, 10, 30)
        edges_tgt = cv2.Canny(tgt_smooth, 10, 30)
        
        if mask is not None:
            ground_mask = cv2.bitwise_not(mask)
            # Match the wide 75px exclusion zone
            ground_mask = erode_mask(ground_mask, radius=75)
        else:
            ground_mask = None
            
        if ground_mask is not None:
            edges_ref = cv2.bitwise_and(edges_ref, ground_mask)
            # Do NOT mask target edges because they have shifted, preventing alignment truncation mismatch
            
        # 4. Filter out small contours (stars, reflections) - minimum length 80px for high robustness
        edges_ref = filter_landscape_contours(edges_ref, min_length=80)
        edges_tgt = filter_landscape_contours(edges_tgt, min_length=80)
        
        # 5. Calculate shifts
        M = align_landscape_optical_flow(ref_img, target_img, mask)
        if M is not None:
            # Warp the target (last) image to reference (first) image coordinates using Lanczos4
            warped_tgt = cv2.warpAffine(target_img, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
            
            # Create a sum/average blend image of the first (ref) and aligned last (target) images
            sum_img = cv2.addWeighted(ref_img, 0.5, warped_tgt, 0.5, 0)
            
            # Warp the target edge map using the same rigid transform
            edges_tgt_warped = cv2.warpAffine(edges_tgt, M, (w, h), flags=cv2.INTER_NEAREST)
            
            # Identify matching contours (intersection of reference edges and warped target edges)
            matching_edges = cv2.bitwise_and(edges_ref, edges_tgt_warped)
            
            # Dilate the matching edges for high-visibility Cyan overlay
            dilate_radius = max(1, int(w / 800.0))
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_radius * 2 + 1, dilate_radius * 2 + 1))
            matching_thick = cv2.dilate(matching_edges, kernel_dilate)
            
            # Color matching contours in Cyan (BGR = [255, 255, 0])
            sum_img[matching_thick > 0] = [255, 255, 0]
            
            dx = M[0, 2]
            dy = M[1, 2]
            # Extract rotation angle from rotation matrix components
            theta_rad = np.arctan2(M[1, 0], M[0, 0])
            theta_deg = np.degrees(theta_rad)
            
            draw_outlined_text(sum_img, f"Landscape Rigid Sum (First vs Last). Shift: dX = {dx:.2f}px, dY = {dy:.2f}px, Rot = {theta_deg:.3f}*", (20, int(40 * title_scale)), 
                               font_scale=title_scale, color=(0, 255, 255), thickness=thickness)
            return sum_img
        else:
            combined = np.hstack((ref_img.copy(), target_img.copy()))
            draw_outlined_text(combined, "Landscape rigid alignment failed.", (20, int(40 * title_scale)), 
                               font_scale=title_scale, color=(0, 0, 255), thickness=thickness)
            return combined

def get_debug_stars_image(img, mask=None, contrast_threshold=0.04, sigma=1.6):
    """
    Creates a debug visualization showing all recognizable stars in red circles
    distributed across the selected sky region.
    """
    out_img = img.copy()
    h, w = img.shape[:2]
    
    # Dynamic scaling based on resolution
    font_scale = max(0.5, w / 1600.0)
    title_scale = max(1.0, w / 1200.0)
    thickness = max(2, int(w / 500.0))
    circle_radius = max(6, int(w / 200.0))
    circle_thickness = max(2, int(w / 800.0))
    
    stars = detect_and_filter_stars(img, mask, contrast_threshold=contrast_threshold, sigma=sigma, max_stars=120)
    
    for idx, pt in enumerate(stars):
        cX, cY = int(pt[0]), int(pt[1])
        cv2.circle(out_img, (cX, cY), circle_radius, (0, 0, 255), circle_thickness)
        draw_outlined_text(out_img, str(idx+1), (cX + circle_radius + 4, cY - circle_radius - 4), font_scale=font_scale, color=(0, 0, 255), thickness=thickness)
        
    draw_outlined_text(out_img, f"Detected Recognizable Stars: {len(stars)} (Homogeneous grid sampling)", (20, int(40 * title_scale)),
                       font_scale=title_scale, color=(0, 0, 255), thickness=thickness)
    return out_img

def check_features(ref_img, mask=None, contrast_threshold=0.04, edge_threshold=10.0, sigma=1.6):
    """
    Returns (sky_star_count, ground_feature_count) for the reference image.
    """
    ref_stars = detect_and_filter_stars(ref_img, mask, contrast_threshold=contrast_threshold, sigma=sigma, max_stars=300)
    sky_count = len(ref_stars)
    
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    if mask is not None:
        ground_mask = cv2.bitwise_not(mask)
        # Apply a very wide 150px exclusion zone to prevent selecting points near the mask transition boundary
        ground_mask = erode_mask(ground_mask, radius=150)
    else:
        ground_mask = None
        
    p0 = cv2.goodFeaturesToTrack(gray_ref, maxCorners=400, qualityLevel=0.01, minDistance=10, mask=ground_mask)
    ground_count = len(p0) if p0 is not None else 0
    
    return sky_count, ground_count

def find_constellation(stars, template_coords, connections, min_matches=4):
    """
    Finds a constellation pattern using rotation/scale-invariant template matching.
    """
    if len(stars) < min_matches:
        return []

    best_score = 0
    best_projected_lines = []

    # Test every pair of detected stars as potential mapping for the first two template stars
    for a_idx in range(len(stars)):
        for b_idx in range(len(stars)):
            if a_idx == b_idx:
                continue
            
            pA = stars[a_idx]
            pB = stars[b_idx]
            
            d_img = np.linalg.norm(pA - pB)
            if d_img < 40.0:
                continue
                
            pT0 = template_coords[0]
            pT1 = template_coords[1]
            d_temp = np.linalg.norm(pT0 - pT1)
            
            scale = d_img / d_temp
            if scale < 30.0 or scale > 800.0:
                continue
                
            vec_temp = pT1 - pT0
            vec_img = pB - pA
            
            angle_temp = np.arctan2(vec_temp[1], vec_temp[0])
            angle_img = np.arctan2(vec_img[1], vec_img[0])
            angle = angle_img - angle_temp
            
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            R = scale * np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            T = pA - np.dot(R, pT0)
            
            projected_pts = []
            for pt in template_coords:
                projected_pts.append(np.dot(R, pt) + T)
            projected_pts = np.array(projected_pts)
            
            matches_count = 0
            matched_indices = []
            tolerance = scale * 0.08
            tolerance = max(8.0, min(25.0, tolerance))
            
            for proj in projected_pts:
                dists = np.linalg.norm(stars - proj, axis=1)
                min_idx = np.argmin(dists)
                if dists[min_idx] < tolerance:
                    matches_count += 1
                    matched_indices.append(min_idx)
                    
            if len(set(matched_indices)) == matches_count and matches_count >= min_matches:
                if matches_count > best_score:
                    best_score = matches_count
                    lines = []
                    for c in connections:
                        pt1 = projected_pts[c[0]]
                        pt2 = projected_pts[c[1]]
                        lines.append(((int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1]))))
                    best_projected_lines = lines
                    
    if best_score >= min_matches:
        return best_projected_lines
    return []

# Celestial Coordinates Database for Plate Solving (loaded from sky_catalog.json)
_db_triangles = []
_db_stars_coords = None
_db_stars_ids = None
_db_stars_mags = None
_db_constellations = None
_db_to_cat_indices = None
_db_triangles_angle0 = None

def _gnomonic_project_vectorized(ra_deg_arr, dec_deg_arr, ra0_deg, dec0_deg):
    ra = np.radians(ra_deg_arr)
    dec = np.radians(dec_deg_arr)
    ra0 = np.radians(ra0_deg)
    dec0 = np.radians(dec0_deg)
    cos_c = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
    cos_c = np.where(cos_c < 1e-5, 1e-5, cos_c)
    x = np.cos(dec) * np.sin(ra - ra0) / cos_c
    y = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / cos_c
    return np.column_stack((x, y))

def _get_triangle_angles(p1, p2, p3):
    a = np.linalg.norm(p2 - p3)
    b = np.linalg.norm(p1 - p3)
    c = np.linalg.norm(p1 - p2)
    if a < 1e-5 or b < 1e-5 or c < 1e-5:
        return None
    try:
        cosA = np.clip((b**2 + c**2 - a**2) / (2 * b * c), -1.0, 1.0)
        cosB = np.clip((a**2 + c**2 - b**2) / (2 * a * c), -1.0, 1.0)
        cosC = np.clip((a**2 + b**2 - c**2) / (2 * a * b), -1.0, 1.0)
        return sorted([np.degrees(np.arccos(cosA)), np.degrees(np.arccos(cosB)), np.degrees(np.arccos(cosC))])
    except:
        return None

def _load_sky_catalog():
    print("--- Aligner Module: Loading Sky Catalog v2.1 ---")
    global _db_triangles, _db_stars_coords, _db_stars_ids, _db_stars_mags, _db_constellations, _db_to_cat_indices, _db_bright_cat_indices, _db_triangles_angle0
    if _db_stars_coords is not None:
        return
        
    catalog_path = os.path.join(os.path.dirname(__file__), "sky_catalog.json")
    if not os.path.exists(catalog_path):
        return
        
    with open(catalog_path, "r") as f:
        catalog = json.load(f)
        
    stars_dict = catalog["stars"]
    _db_constellations = catalog["constellations"]
    
    star_ids = []
    star_coords = []
    star_mags = []
    
    for sid_str, info in stars_dict.items():
        vmag = info["vmag"]
        if vmag <= 5.0: # Filter catalog to Bortle 5 stars only (clean, visual view)
            star_ids.append(int(sid_str))
            star_coords.append((info["ra"], info["dec"]))
            star_mags.append(vmag)
        
    _db_stars_ids = np.array(star_ids, dtype=np.int32)
    _db_stars_coords = np.array(star_coords, dtype=np.float32)
    _db_stars_mags = np.array(star_mags, dtype=np.float32)
    
    # Filter database stars for triangle matching (vmag <= 4.0 to optimize triangle counts to 12563)
    mask_bright = _db_stars_mags <= 4.0
    db_pts_raw = _db_stars_coords[mask_bright]
    db_ids_bright = _db_stars_ids[mask_bright]
    n_db = len(db_pts_raw)
    
    # Precompute mapping from solver database indices to catalog indices
    db_to_cat_list = []
    for idx in range(len(db_ids_bright)):
        cat_idx = np.where(_db_stars_ids == db_ids_bright[idx])[0][0]
        db_to_cat_list.append(cat_idx)
    _db_to_cat_indices = np.array(db_to_cat_list, dtype=np.int32)
    
    # Precompute indices of top 25 brightest stars in the catalog
    _db_bright_cat_indices = np.argsort(_db_stars_mags)[:25]
    
    max_dist = 20.0
    dist_matrix = np.zeros((n_db, n_db))
    for i in range(n_db):
        for j in range(i+1, n_db):
            d = np.linalg.norm(db_pts_raw[i] - db_pts_raw[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
            
    for i in range(n_db):
        for j in range(i+1, n_db):
            if dist_matrix[i, j] > max_dist:
                continue
            for k in range(j+1, n_db):
                if dist_matrix[i, k] > max_dist or dist_matrix[j, k] > max_dist:
                    continue
                    
                pts = db_pts_raw[[i, j, k]]
                ra0 = np.mean(pts[:, 0])
                dec0 = np.mean(pts[:, 1])
                
                # Project locally (vectorized)
                p1_xy = _gnomonic_project_vectorized(pts[:, 0], pts[:, 1], ra0, dec0)
                
                angles = _get_triangle_angles(p1_xy[0], p1_xy[1], p1_xy[2])
                if angles is not None:
                    # Find 15 nearest database stars to the center of the triangle
                    dists_to_center = np.linalg.norm(db_pts_raw - [ra0, dec0], axis=1)
                    nearest_indices = np.argsort(dists_to_center)[:15]
                    
                    # Precompute the projected coordinates of the 15 nearest stars
                    nearest_coords_raw = db_pts_raw[nearest_indices]
                    nearest_coords_proj = _gnomonic_project_vectorized(nearest_coords_raw[:, 0], nearest_coords_raw[:, 1], ra0, dec0).astype(np.float32)
                    
                    _db_triangles.append({
                        "angles": angles,
                        "db_indices": [i, j, k],
                        "ra0": ra0,
                        "dec0": dec0,
                        "nearest_coords_proj": nearest_coords_proj
                    })
                    
    # Sort database triangles by first angle to enable binary search
    _db_triangles = sorted(_db_triangles, key=lambda x: x["angles"][0])
    _db_triangles_angle0 = np.array([x["angles"][0] for x in _db_triangles], dtype=np.float32)

import json
import os

def draw_constellations(img, mask=None, cancel_event=None):
    """
    Solves the starry sky using a local offline plate-solving engine (triangle-based hashing).
    Projects and overlays identified constellations onto the image canvas.
    """
    print("--- Aligner Module: Starting Solve v2.1 ---")
    t_start = time.time()
    _load_sky_catalog()
    if _db_stars_coords is None:
        return img.copy(), False
        
    eroded = erode_mask(mask, radius=15) if mask is not None else None
    
    # Safe contrast tuning loop to keep stars close to 30 without discarding key stars
    contrast = 0.04
    sigma = 1.6
    stars = detect_stars_centroids(img, eroded, contrast_threshold=contrast, sigma=sigma, max_stars=80)
    
    if len(stars) > 30:
        for attempt_contrast in [0.05, 0.06]:
            attempt_stars = detect_stars_centroids(img, eroded, contrast_threshold=attempt_contrast, sigma=sigma, max_stars=80)
            if len(attempt_stars) < 30:
                stars = attempt_stars
                contrast = attempt_contrast
                break
            stars = attempt_stars
            contrast = attempt_contrast
                
    if len(stars) < 4:
        print(f"--- Aligner Module: Too few stars detected ({len(stars)}) ---")
        return img.copy(), False
        
    # Only take top 15 stars for combinations to ensure extreme speed
    candidate_stars = stars[:15]
    h, w, c = img.shape
    
    # Filter database stars for triangle matching (vmag <= 4.0)
    mask_bright = _db_stars_mags <= 4.0
    db_ids_bright = _db_stars_ids[mask_bright]
    
    print(f"--- Aligner Module: Precomputed triangles = {len(_db_triangles)} ---")
    print(f"--- Aligner Module: Total stars in mask = {len(stars)} (contrast = {contrast}) ---")
    print(f"--- Aligner Module: Candidate stars = {len(candidate_stars)} (combinations = {len(candidate_stars)*(len(candidate_stars)-1)*(len(candidate_stars)-2)//6}) ---")
    
    best_score = 0
    best_transform = None
    best_ra0 = 0.0
    best_dec0 = 0.0
    
    perms = list(itertools.permutations([0, 1, 2]))
    
    # Try finding matching triangles
    for i in range(len(candidate_stars)):
        if cancel_event is not None and cancel_event.is_set():
            return img.copy(), False
        for j in range(i+1, len(candidate_stars)):
            for k in range(j+1, len(candidate_stars)):
                angles = _get_triangle_angles(candidate_stars[i], candidate_stars[j], candidate_stars[k])
                if angles is None:
                    continue
                
                img_tri_pts = np.array([candidate_stars[i], candidate_stars[j], candidate_stars[k]], dtype=np.float32)
                
                # Binary search to find matching database triangles (within 0.15 deg tolerance)
                idx_start = np.searchsorted(_db_triangles_angle0, angles[0] - 0.15)
                idx_end = np.searchsorted(_db_triangles_angle0, angles[0] + 0.15)
                
                for idx in range(idx_start, idx_end):
                    dbt = _db_triangles[idx]
                    if abs(angles[1] - dbt["angles"][1]) < 0.15:
                        db_idx = dbt["db_indices"]
                        ra0 = dbt["ra0"]
                        dec0 = dbt["dec0"]
                        
                        # Project ONLY the 3 stars of the database triangle (extremely fast!)
                        db_idx_in_all = _db_to_cat_indices[db_idx]
                        db_tri_pts_raw = _db_stars_coords[db_idx_in_all]
                        db_tri_pts = _gnomonic_project_vectorized(db_tri_pts_raw[:, 0], db_tri_pts_raw[:, 1], ra0, dec0).astype(np.float32)
                        
                        for p in perms:
                            ordered_db = db_tri_pts[list(p)]
                            M, _ = cv2.estimateAffinePartial2D(ordered_db, img_tri_pts)
                            if M is not None:
                                # Conformal scale check to avoid verifying unphysical mappings (pixels per radian)
                                scale = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
                                if scale < 1000.0 or scale > 80000.0:
                                    continue
                                    
                                # Vectorized Micro-verification using preprojected coordinates of the 15 nearest database stars
                                db_pts_proj_micro = dbt["nearest_coords_proj"]
                                proj_micro = cv2.transform(np.expand_dims(db_pts_proj_micro, axis=0), M)[0]
                                
                                # Pairwise distance calculation using numpy broadcasting (extremely fast in C)
                                diff = proj_micro[:, np.newaxis, :] - stars[np.newaxis, :, :] # shape (15, len(stars), 2)
                                dists2 = np.sum(diff**2, axis=2) # shape (15, len(stars))
                                min_dists2 = np.min(dists2, axis=1) # shape (15,)
                                micro_match = np.sum(min_dists2 < 625.0) # 25^2 = 625
                                        
                                if micro_match < 5: # 3 triangle stars + at least 2 extra
                                    continue
                                    
                                # Project all database stars ONLY when micro-verification passes
                                db_pts_proj = _gnomonic_project_vectorized(_db_stars_coords[:, 0], _db_stars_coords[:, 1], ra0, dec0).astype(np.float32)
                                proj = cv2.transform(np.expand_dims(db_pts_proj, axis=0), M)[0]
                                
                                # Enforce 1-to-1 matching constraint to prevent collapsed cluster bug
                                matched_img_indices = set()
                                match_count = 0
                                for pt in proj:
                                    dists = np.linalg.norm(stars - pt, axis=1)
                                    min_idx = np.argmin(dists)
                                    if dists[min_idx] < 25.0 and min_idx not in matched_img_indices:
                                        match_count += 1
                                        matched_img_indices.add(min_idx)
                                        
                                if match_count > best_score:
                                    best_score = match_count
                                    best_transform = M
                                    best_ra0 = ra0
                                    best_dec0 = dec0
                                
    out_img = img.copy()
    if best_transform is not None and best_score >= 5:
        # Refine the initial 3-star partial affine transform using a Homography (8 DoF) fit on all matched stars
        db_pts_proj = _gnomonic_project_vectorized(_db_stars_coords[:, 0], _db_stars_coords[:, 1], best_ra0, best_dec0).astype(np.float32)
        proj_pts_all = cv2.transform(np.expand_dims(db_pts_proj, axis=0), best_transform)[0]
        
        matched_db_pts = []
        matched_img_pts = []
        matched_img_indices = set()
        for idx, pt in enumerate(proj_pts_all):
            dists = np.linalg.norm(stars - pt, axis=1)
            min_idx = np.argmin(dists)
            # Use a wide 120px threshold to capture perspective shifts at the edges
            if dists[min_idx] < 120.0 and min_idx not in matched_img_indices:
                matched_db_pts.append(db_pts_proj[idx])
                matched_img_pts.append(stars[min_idx])
                matched_img_indices.add(min_idx)
                
        H_refined = None
        if len(matched_db_pts) >= 4:
            matched_db_pts = np.array(matched_db_pts, dtype=np.float32)
            matched_img_pts = np.array(matched_img_pts, dtype=np.float32)
            H_refined, _ = cv2.findHomography(matched_db_pts, matched_img_pts, method=cv2.RANSAC, ransacReprojThreshold=8.0)
            
        if H_refined is not None:
            # Re-project all catalog stars to pixel space using the refined Homography (perfect perspective matching!)
            proj_pts_all = cv2.perspectiveTransform(np.expand_dims(db_pts_proj, axis=1), H_refined)[:, 0, :]
        else:
            # Fallback to the initial transform if homography estimation fails
            proj_pts_all = cv2.transform(np.expand_dims(db_pts_proj, axis=0), best_transform)[0]
        
        proj_pts = {}
        for idx, name in enumerate(_db_stars_ids):
            proj_pts[int(name)] = (int(proj_pts_all[idx][0]), int(proj_pts_all[idx][1]))
            
        found_names = []
        for cname, conn_lines in _db_constellations.items():
            all_stars_in_constellation = list(set([s for conn in conn_lines for s in conn]))
            visible_stars = 0
            for sname in all_stars_in_constellation:
                if sname in proj_pts:
                    px, py = proj_pts[sname]
                    if 0 <= px < w and 0 <= py < h:
                        visible_stars += 1
                        
            if visible_stars >= max(2, int(len(all_stars_in_constellation) * 0.5)):
                found_names.append(cname)
                # Draw lines
                for s1, s2 in conn_lines:
                    if s1 in proj_pts and s2 in proj_pts:
                        cv2.line(out_img, proj_pts[s1], proj_pts[s2], (0, 255, 0), 2, lineType=cv2.LINE_AA)
                # Draw small circles for visible stars of the constellation
                xs = []
                ys = []
                for sname in all_stars_in_constellation:
                    if sname in proj_pts:
                        px, py = proj_pts[sname]
                        if 0 <= px < w and 0 <= py < h:
                            cv2.circle(out_img, (px, py), 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)
                            xs.append(px)
                            ys.append(py)
                
                # Draw constellation name next to its geometric center with drop shadow
                if xs and ys:
                    cx = int(np.mean(xs))
                    cy = int(np.mean(ys))
                    cv2.putText(out_img, cname, (cx + 10, cy - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, lineType=cv2.LINE_AA)
                    cv2.putText(out_img, cname, (cx + 10, cy - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, lineType=cv2.LINE_AA)
                            
        t_end = time.time()
        if found_names:
            names_str = ", ".join(found_names)
            print(f"--- Aligner Module: SUCCESS! Solved: {names_str} (time: {t_end - t_start:.3f}s, best_score: {best_score}) ---")
            return out_img, True
            
    t_end = time.time()
    print(f"--- Aligner Module: FAILED to solve (time: {t_end - t_start:.3f}s, best_score: {best_score}) ---")
    return out_img, False
