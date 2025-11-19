import cv2
import numpy as np
import os

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

def get_sift_matches(img1, img2, ratio_threshold=0.85, max_features=5000):
    """
    Finds SIFT matches using Lowe's Ratio Test.
    """
    sift = cv2.SIFT_create(nfeatures=max_features)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        return kp1, kp2, []
    
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50) 
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except cv2.error:
        return kp1, kp2, []

    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
            
    return kp1, kp2, good_matches

def draw_matches(img1, kp1, img2, kp2, matches):
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
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