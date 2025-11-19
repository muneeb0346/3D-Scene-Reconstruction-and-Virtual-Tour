import cv2
import numpy as np

def get_intrinsic_matrix(image_shape):
    """
    Approximates K assuming f = image_width and center = image_center.
    """
    h, w = image_shape[:2]
    return np.array([
        [w, 0, w/2],
        [0, w, h/2],
        [0, 0,   1]
    ], dtype=np.float32)

def calculate_essential_matrix(kp1, kp2, matches, K):
    """
    Estimates the Essential Matrix (E) using RANSAC.
    """
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    E, mask = cv2.findEssentialMat(pts1, pts2, K, 
                                   method=cv2.RANSAC, 
                                   prob=0.999, 
                                   threshold=3.0) 
    
    return E, mask, pts1, pts2

def recover_camera_pose(E, pts1, pts2, K):
    """
    Decomposes E into R and t.
    """
    points_passed, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    return R, t, mask

def triangulate_points(R, t, K, pts1, pts2):
    """
    Triangulates 3D points from two views.
    """
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))
    
    pts1_t = pts1.T
    pts2_t = pts2.T
    
    points_4d = cv2.triangulatePoints(P1, P2, pts1_t, pts2_t)
    points_3d = (points_4d[:3] / points_4d[3]).T
    return points_3d

def get_inlier_points(mask, pts1, pts2):
    """
    Helper to filter points based on a RANSAC/OpenCV mask.
    OpenCV masks use 255 for inliers (or any non-zero value).
    """
    bool_mask = mask.ravel() > 0
    return pts1[bool_mask], pts2[bool_mask]