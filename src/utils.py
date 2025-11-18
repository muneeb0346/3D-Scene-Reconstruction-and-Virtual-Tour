import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def load_images_from_folder(folder_path):
    """
    Loads all images from a specified folder in sorted order.
    Returns a list of images in grayscale.
    """
    images = []
    # Ensure files are sorted to maintain consecutive order (01, 02, 03...)
    filenames = sorted(os.listdir(folder_path))
    
    for filename in filenames:
        img_path = os.path.join(folder_path, filename)
        # Load image in grayscale as required for SIFT/ORB
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
        if img is not None:
            images.append(img)
            
    print(f"Successfully loaded {len(images)} images from {folder_path}")
    return images

def get_sift_matches(img1, img2, ratio_threshold=0.7):
    """
    Finds and filters SIFT matches between two images using 
    Lowe's Ratio Test.
    """
    # 1. Create SIFT detector
    sift = cv2.SIFT_create()

    # 2. Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 3. Use FLANN based matcher for speed
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50) # or pass empty dictionary
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # 4. Find all matches (k=2) to apply Lowe's test
    all_matches = flann.knnMatch(des1, des2, k=2)

    # 5. Apply Lowe's Ratio Test to filter good matches
    good_matches = []
    for m, n in all_matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
            
    return kp1, kp2, good_matches

def draw_matches(img1, kp1, img2, kp2, matches):
    """
    Creates a visualization of the matches between two images.
    """
    # cv2.drawMatches needs lists, not just the 'matches' object
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return img_matches