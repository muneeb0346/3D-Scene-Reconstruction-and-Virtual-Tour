import cv2
import numpy as np
import os
from typing import Any, Dict, Mapping, cast

def load_images_from_folder(folder_path, target_width=1024):
    """
    Loads images, converts to grayscale, and resizes them if they are too large.
    """
    images = []
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} not found.")
        return []

    filenames = sorted(os.listdir(folder_path))
    
    for filename in filenames:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                h, w = img.shape
                if w > target_width:
                    scale = target_width / w
                    new_h = int(h * scale)
                    img = cv2.resize(img, (target_width, new_h))
                images.append(img)
            
    print(f"Successfully loaded {len(images)} images from {folder_path}")
    print(f"Images resized to max width: {target_width}px")
    return images

def get_sift_matches(img1, img2, ratio_threshold=0.65, max_features=5000):
    """
    Finds SIFT matches using Lowe's Ratio Test.
    """
    sift = cv2.SIFT_create(nfeatures=max_features)  # type: ignore[attr-defined]
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        return kp1, kp2, []
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50) 
    
    flann = cv2.FlannBasedMatcher(cast(Dict[str, Any], index_params), cast(Dict[str, Any], search_params))  # type: ignore
    
    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except cv2.error:
        return kp1, kp2, []

    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
            
    return kp1, kp2, good_matches

def get_dense_combined_matches(img1, img2,
                               sift_nfeatures=20000,
                               sift_ratio=0.85,
                               orb_nfeatures=10000,
                               orb_ratio=0.85,
                               subpixel=True):
    """Dense matching using high-sensitivity SIFT + ORB combined.

    Returns:
        pts1 (np.ndarray): Nx2 array of points in image 1
        pts2 (np.ndarray): Nx2 array of points in image 2
        stats (dict): diagnostic information
    """
    gray1 = img1 if len(img1.shape) == 2 else cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = img2 if len(img2.shape) == 2 else cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # SIFT
    # High feature count SIFT (available in opencv-python >=4.4)
    sift = cv2.SIFT_create(nfeatures=sift_nfeatures, contrastThreshold=0.01)  # type: ignore[attr-defined]
    kp1_s, des1_s = sift.detectAndCompute(gray1, None)
    kp2_s, des2_s = sift.detectAndCompute(gray2, None)

    pts1_s = []
    pts2_s = []
    if des1_s is not None and des2_s is not None:
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=100)
        # FLANN matcher (keys are ints but stub may complain, cast to Dict[str, Any])
        flann = cv2.FlannBasedMatcher(cast(Dict[str, Any], index_params), cast(Dict[str, Any], search_params))  # type: ignore
        matches_s = flann.knnMatch(des1_s, des2_s, k=2)
        for m, n in matches_s:
            if m.distance < sift_ratio * n.distance:
                pts1_s.append(kp1_s[m.queryIdx].pt)
                pts2_s.append(kp2_s[m.trainIdx].pt)

    # ORB
    orb = cv2.ORB_create(nfeatures=orb_nfeatures)  # type: ignore[attr-defined]
    kp1_o, des1_o = orb.detectAndCompute(gray1, None)
    kp2_o, des2_o = orb.detectAndCompute(gray2, None)
    pts1_o = []
    pts2_o = []
    if des1_o is not None and des2_o is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches_o = bf.knnMatch(des1_o, des2_o, k=2)
        for m, n in matches_o:
            if m.distance < orb_ratio * n.distance:
                pts1_o.append(kp1_o[m.queryIdx].pt)
                pts2_o.append(kp2_o[m.trainIdx].pt)

    # Combine
    if pts1_s and pts1_o:
        pts1 = np.vstack([np.array(pts1_s, dtype=np.float32), np.array(pts1_o, dtype=np.float32)])
        pts2 = np.vstack([np.array(pts2_s, dtype=np.float32), np.array(pts2_o, dtype=np.float32)])
    elif pts1_s:
        pts1 = np.array(pts1_s, dtype=np.float32)
        pts2 = np.array(pts2_s, dtype=np.float32)
    else:
        pts1 = np.array(pts1_o, dtype=np.float32)
        pts2 = np.array(pts2_o, dtype=np.float32)

    # Optional sub-pixel refinement
    if subpixel and len(pts1) > 0:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        # cornerSubPix expects float32 and points inside image
        h, w = gray1.shape[:2]
        valid_mask = (pts1[:,0] > 1) & (pts1[:,0] < w-2) & (pts1[:,1] > 1) & (pts1[:,1] < h-2)
        pts1_refine = pts1[valid_mask].copy()
        pts2_refine = pts2[valid_mask].copy()
        if len(pts1_refine) > 0:
            pts1_refine = cv2.cornerSubPix(gray1, pts1_refine, (5,5), (-1,-1), criteria)
            pts2_refine = cv2.cornerSubPix(gray2, pts2_refine, (5,5), (-1,-1), criteria)
            pts1[valid_mask] = pts1_refine
            pts2[valid_mask] = pts2_refine

    stats = {
        'sift_keypoints_img1': len(kp1_s),
        'sift_keypoints_img2': len(kp2_s),
        'orb_keypoints_img1': len(kp1_o),
        'orb_keypoints_img2': len(kp2_o),
        'sift_matches': len(pts1_s),
        'orb_matches': len(pts1_o),
        'combined_matches': len(pts1)
    }
    return pts1, pts2, stats

def draw_matches(img1, kp1, img2, kp2, matches):
    # Provide an explicit output image to satisfy static analysis tools
    h = max(img1.shape[0], img2.shape[0])
    w = img1.shape[1] + img2.shape[1]
    out_img = np.zeros((h, w, 3), dtype=np.uint8)
    if len(img1.shape) == 2:
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    else:
        img1_color = img1
    if len(img2.shape) == 2:
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    else:
        img2_color = img2
    out_img[:, :img1.shape[1]] = img1_color
    out_img[:, img1.shape[1]:] = img2_color
    img_matches = cv2.drawMatches(img1_color, kp1, img2_color, kp2, matches, out_img,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)  # type: ignore[arg-type]
    return img_matches

def save_point_cloud_to_ply(points_3d, filename):
    """
    Saves 3D points to a .ply file (Manual Requirement Phase 1).
    """
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points_3d)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        for p in points_3d:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")
            
    print(f"Saved point cloud to {filename}")

def save_colored_point_cloud_to_ply(points_3d, colors, filename):
    """Save colored 3D points to PLY (ASCII). Colors expected in RGB 0-255 or 0-1.
    """
    if colors.max() <= 1.0:
        colors_out = (colors * 255).astype(np.uint8)
    else:
        colors_out = colors.astype(np.uint8)

    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(points_3d)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for p, c in zip(points_3d, colors_out):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")
    print(f"Saved colored point cloud to {filename}")

def sample_colors(img, pts):
    """Sample RGB colors from image at floating point coordinates pts Nx2.
    Returns colors in 0-255 uint8.
    """
    if len(img.shape) == 2:
        # grayscale expand
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_rgb = img
    h, w = img_rgb.shape[:2]
    colors = []
    for x, y in pts:
        xi = min(max(int(round(x)), 0), w-1)
        yi = min(max(int(round(y)), 0), h-1)
        colors.append(img_rgb[yi, xi])
    return np.array(colors, dtype=np.uint8)