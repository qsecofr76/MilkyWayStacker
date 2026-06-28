import cv2
import numpy as np
import os
import itertools

# Stars database
stars_db = {
    "Deneb": (20.69 * 15, 45.28),
    "Sadr": (20.37 * 15, 40.26),
    "Albireo": (19.51 * 15, 27.96),
    "Gienah": (20.77 * 15, 33.97),
    "Fawaris": (19.61 * 15, 45.13),
    "Vega": (18.62 * 15, 38.78),
    "Sheliak": (18.84 * 15, 33.36),
    "Sulafat": (18.98 * 15, 32.69),
    "d_Lyrae": (18.91 * 15, 36.97),
    "z_Lyrae": (18.75 * 15, 37.60),
    "Altair": (19.85 * 15, 8.87),
    "Tarazed": (19.77 * 15, 10.61),
    "Alshain": (19.92 * 15, 6.41),
    "a_Sge": (19.67 * 15, 18.01),
    "b_Sge": (19.68 * 15, 17.47),
    "g_Sge": (19.98 * 15, 19.49),
    "d_Sge": (19.79 * 15, 18.53),
    "e_Sge": (20.08 * 15, 20.11),
    "aSualocin": (20.66 * 15, 15.91),
    "bRotanev": (20.62 * 15, 14.59),
    "gDelphini": (20.78 * 15, 16.12),
    "dDelphini": (20.72 * 15, 15.08),
}

constellations = {
    "Cygnus": {
        "conn": [("Deneb", "Sadr"), ("Sadr", "Albireo"), ("Sadr", "Gienah"), ("Sadr", "Fawaris")]
    },
    "Lyra": {
        "conn": [("Vega", "z_Lyrae"), ("z_Lyrae", "d_Lyrae"), ("d_Lyrae", "Sulafat"), ("Sulafat", "Sheliak"), ("Sheliak", "z_Lyrae")]
    },
    "Sagitta": {
        "conn": [("d_Sge", "a_Sge"), ("d_Sge", "b_Sge"), ("a_Sge", "b_Sge"), ("d_Sge", "g_Sge"), ("g_Sge", "e_Sge")]
    },
    "Aquila": {
        "conn": [("Altair", "Tarazed"), ("Altair", "Alshain")]
    },
    "Delphinus": {
        "conn": [("aSualocin", "bRotanev"), ("bRotanev", "dDelphini"), ("dDelphini", "gDelphini"), ("gDelphini", "aSualocin")]
    }
}

# Helper to compute angles of a triangle formed by 3 points
def get_triangle_angles(p1, p2, p3):
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

db_names = list(stars_db.keys())
db_pts = np.array([stars_db[name] for name in db_names], dtype=np.float32)
db_triangles = []
n_db = len(db_names)

for i in range(n_db):
    for j in range(i+1, n_db):
        for k in range(j+1, n_db):
            angles = get_triangle_angles(db_pts[i], db_pts[j], db_pts[k])
            if angles is not None:
                db_triangles.append({
                    "angles": angles,
                    "indices": [i, j, k]
                })

print(f"Precomputed {len(db_triangles)} triangles in database.")

img_path = "caseracigno_00001.tiff"
if not os.path.exists(img_path):
    img_path = "NovaAstrometry.jpg"

img = cv2.imread(img_path)
if img is None:
    print("No test image found!")
    exit()

from core.aligner import detect_stars_centroids
stars = detect_stars_centroids(img, mask=None, contrast_threshold=0.04, sigma=1.6, max_stars=80)
print(f"Detected {len(stars)} stars.")

stars = stars[:25]

best_score = 0
best_transform = None

# Permutations of indices [0, 1, 2] to test all correspondences
perms = list(itertools.permutations([0, 1, 2]))

for i in range(len(stars)):
    for j in range(i+1, len(stars)):
        for k in range(j+1, len(stars)):
            angles = get_triangle_angles(stars[i], stars[j], stars[k])
            if angles is None:
                continue
                
            img_tri_pts = np.array([stars[i], stars[j], stars[k]], dtype=np.float32)
            
            for dbt in db_triangles:
                # Match angles
                if (abs(angles[0] - dbt["angles"][0]) < 1.5 and 
                    abs(angles[1] - dbt["angles"][1]) < 1.5):
                    
                    db_idx = dbt["indices"]
                    db_tri_pts = db_pts[db_idx]
                    
                    # Test all 6 vertex mappings
                    for p in perms:
                        ordered_db = db_tri_pts[list(p)]
                        
                        # Estimate partial affine (translation, rotation, uniform scale)
                        # We use estimateAffinePartial2D to prevent skew distortion
                        M, _ = cv2.estimateAffinePartial2D(ordered_db, img_tri_pts)
                        if M is not None:
                            # Project all DB points
                            proj = cv2.transform(np.expand_dims(db_pts, axis=0), M)[0]
                            
                            # 1-to-1 Verification
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

print(f"Best solution matches {best_score} database stars.")
if best_score >= 5:
    print("SUCCESS! Plate solving resolved successfully!")
    out_img = img.copy()
    proj_pts = {}
    for idx, name in enumerate(db_names):
        radec = np.array([[db_pts[idx]]], dtype=np.float32)
        pixel = cv2.transform(radec, best_transform)[0][0]
        proj_pts[name] = (int(pixel[0]), int(pixel[1]))
        cv2.circle(out_img, proj_pts[name], 5, (0, 0, 255), -1)
        cv2.putText(out_img, name, (proj_pts[name][0] + 8, proj_pts[name][1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
    for name, cinfo in constellations.items():
        for s1, s2 in cinfo["conn"]:
            if s1 in proj_pts and s2 in proj_pts:
                cv2.line(out_img, proj_pts[s1], proj_pts[s2], (0, 255, 0), 2, lineType=cv2.LINE_AA)
                
    cv2.imwrite("solved_triangles.png", out_img)
    print("Saved output to solved_triangles.png")
else:
    print("Failed to solve.")
