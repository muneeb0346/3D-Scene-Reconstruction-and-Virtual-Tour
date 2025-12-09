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

def calculate_essential_matrix_from_points(pts1, pts2, K, ransac_thresh=3.0, prob=0.999):
    """Estimate Essential matrix directly from Nx2 point arrays.
    Returns (E, mask) where mask is Nx1 inlier mask.
    """
    pts1_f = np.asarray(pts1, dtype=np.float32)
    pts2_f = np.asarray(pts2, dtype=np.float32)
    E, mask = cv2.findEssentialMat(pts1_f, pts2_f, K, method=cv2.RANSAC, prob=prob, threshold=ransac_thresh)
    return E, mask, pts1_f, pts2_f

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

def filter_cheirality(R, t, K, pts1, pts2):
    """Perform cheirality check and return mask selecting points in front of both cameras."""
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = K @ np.hstack((R, t))
    pts1_t = pts1.T
    pts2_t = pts2.T
    points_4d = cv2.triangulatePoints(P1, P2, pts1_t, pts2_t)
    points_3d = (points_4d[:3] / points_4d[3]).T
    # depth in camera 1 is Z, in camera 2 is third component of R*X + t
    depth_cam1 = points_3d[:, 2]
    depth_cam2 = (R @ points_3d.T + t).T[:, 2]
    mask = (depth_cam1 > 0) & (depth_cam2 > 0)
    return mask, points_3d

def get_inlier_points(mask, pts1, pts2):
    """
    Helper to filter points based on a RANSAC/OpenCV mask.
    OpenCV masks use 255 for inliers (or any non-zero value).
    """
    bool_mask = mask.ravel() > 0
    return pts1[bool_mask], pts2[bool_mask]


def reconstruct_pair(img1, img2, K, idx1, idx2, 
                    min_matches=40, min_inliers=25, min_cheirality=15,
                    ransac_thresh=1.5, max_dist_percentile=98):
    """
    Reconstruct 3D points from a single image pair using dense matching and pose refinement.
    
    Args:
        img1, img2: Input grayscale images
        K: 3x3 intrinsic matrix
        idx1, idx2: Image indices (for logging)
        min_matches: Minimum required combined matches
        min_inliers: Minimum RANSAC inliers
        min_cheirality: Minimum points passing cheirality check
        ransac_thresh: RANSAC threshold in pixels
        max_dist_percentile: Percentile for outlier filtering
    
    Returns:
        Tuple of (points_3d, colors, pose) or (None, None, None) if reconstruction fails
        pose is (R, t) tuple representing camera transformation
    """
    from utils import get_dense_combined_matches, sample_colors, filter_points
    
    print(f'  Pair {idx1} -> {idx2}')
    pts1_raw, pts2_raw, stats = get_dense_combined_matches(
        img1, img2,
        sift_nfeatures=15000,
        sift_ratio=0.80,
        orb_nfeatures=8000,
        orb_ratio=0.80,
        subpixel=True,
    )
    print(f"    Matches: SIFT={stats['sift_matches']} ORB={stats['orb_matches']} Combined={stats['combined_matches']}")
    if len(pts1_raw) < min_matches:
        print('    Skipped: too few matches.')
        return None, None, None

    E, mask, pts1, pts2 = calculate_essential_matrix_from_points(
        pts1_raw, pts2_raw, K, ransac_thresh=ransac_thresh, prob=0.999)
    mask_b = mask.ravel() > 0 if mask is not None else np.zeros(len(pts1), dtype=bool)
    pts1_in, pts2_in = pts1[mask_b], pts2[mask_b]
    print(f'    RANSAC inliers: {len(pts1_in)}')
    if len(pts1_in) < min_inliers:
        print('    Skipped: insufficient inliers.')
        return None, None, None

    R, t, pose_mask = recover_camera_pose(E, pts1_in, pts2_in, K)
    pose_b = pose_mask.ravel() > 0 if pose_mask is not None else np.zeros(len(pts1_in), dtype=bool)
    pts1_final, pts2_final = pts1_in[pose_b], pts2_in[pose_b]
    print(f'    Cheirality passed: {len(pts1_final)}')
    if len(pts1_final) < min_cheirality:
        print('    Skipped: cheirality rejected too many points.')
        return None, None, None

    # Normalize translation scale to 1 to reduce drift
    t_norm = np.linalg.norm(t)
    if t_norm > 0:
        t = t / t_norm

    points_3d = triangulate_points(R, t, K, pts1_final, pts2_final)
    colors = 0.5 * (sample_colors(img1, pts1_final).astype(np.float32) +
                    sample_colors(img2, pts2_final).astype(np.float32))

    points_3d, colors, keep_mask = filter_points(points_3d, colors, max_percentile=max_dist_percentile)
    pts1_final = pts1_final[keep_mask]
    pts2_final = pts2_final[keep_mask]
    if colors is not None:
        colors = colors.astype(np.uint8)

    if len(points_3d) == 0:
        print('    Skipped: no finite points after filtering.')
        return None, None, None

    # Pose refinement with PnP
    success, rvec, tvec, _ = cv2.solvePnPRansac(
        points_3d.astype(np.float32), pts2_final.astype(np.float32), K, None,
        iterationsCount=300, reprojectionError=3.0, confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if success:
        R_refined, _ = cv2.Rodrigues(rvec)
        t_refined = tvec.reshape(3,1)
        t_ref_norm = np.linalg.norm(t_refined)
        if t_ref_norm > 0:
            t_refined = t_refined / t_ref_norm
        # Re-triangulate with refined pose
        points_3d = triangulate_points(R_refined, t_refined, K, pts1_final, pts2_final)
        points_3d, colors, keep_mask = filter_points(points_3d, colors, max_percentile=max_dist_percentile)
        pts1_final = pts1_final[keep_mask]
        pts2_final = pts2_final[keep_mask]
        pose_out = (R_refined, t_refined)
        print(f'    Pose refined via PnP (inliers: {len(points_3d)})')
    else:
        pose_out = (R, t)
        print('    Pose refinement skipped (PnP failed).')

    if colors is not None:
        colors = colors.astype(np.uint8)
    print(f'    Reconstructed points: {len(points_3d)}')
    return points_3d, colors, pose_out


def reconstruct_range(range_tuple, images, K, 
                     min_matches=40, min_inliers=25, min_cheirality=15,
                     ransac_thresh=1.5, max_dist_percentile=98):
    """
    Reconstruct 3D point cloud for a range of consecutive images.
    Uses incremental SfM: chain consecutive pairs and merge their point clouds.
    
    Args:
        range_tuple: (start_idx, end_idx) tuple defining image range (inclusive)
        images: List of all images
        K: 3x3 intrinsic matrix
        min_matches, min_inliers, min_cheirality: Thresholds for pair reconstruction
        ransac_thresh: RANSAC threshold in pixels
        max_dist_percentile: Percentile for outlier filtering
    
    Returns:
        Tuple of (merged_points, merged_colors) or (None, None) if reconstruction fails
    """
    start_idx, end_idx = range_tuple
    print(f'\\n=== Reconstructing Range ({start_idx}, {end_idx}): {end_idx - start_idx + 1} images ===')
    
    all_points = []
    all_colors = []
    camera_poses = {}  # idx -> (R_global, t_global)
    
    # Process consecutive pairs within the range
    for i in range(start_idx, end_idx):
        j = i + 1
        pts3d, cols, pose = reconstruct_pair(
            images[i], images[j], K, i, j,
            min_matches=min_matches, min_inliers=min_inliers, 
            min_cheirality=min_cheirality, ransac_thresh=ransac_thresh,
            max_dist_percentile=max_dist_percentile
        )
        
        if pts3d is None or cols is None:
            print(f'  Warning: Pair {i}-{j} failed. Skipping.')
            continue
        
        # Chain poses incrementally
        if i == start_idx:
            # First pair: set as reference frame
            R_rel, t_rel = pose
            camera_poses[i] = (np.eye(3), np.zeros((3,1)))
            camera_poses[j] = (R_rel, t_rel)
            pts_global = pts3d  # Already in reference frame
            all_points.append(pts_global)
            all_colors.append(cols)
            print(f'  Set pair {i}-{j} as reference frame.')
        elif i in camera_poses:
            # Chain to existing pose
            R_rel, t_rel = pose
            R_i, t_i = camera_poses[i]
            R_j = R_i @ R_rel
            t_j = R_i @ t_rel + t_i
            camera_poses[j] = (R_j, t_j)
            pts_global = (R_i @ pts3d.T).T + t_i.T
            all_points.append(pts_global)
            all_colors.append(cols)
            print(f'  Chained pair {i}-{j} to global frame.')
        else:
            print(f'  Warning: Cannot chain pair {i}-{j} (camera {i} pose unknown).')
    
    if not all_points:
        print(f'Range ({start_idx}, {end_idx}): No valid reconstructions.')
        return None, None
    
    # Merge all points from the range
    merged_pts = np.vstack(all_points)
    merged_cols = np.vstack(all_colors) if all_colors else None
    
    print(f'Range ({start_idx}, {end_idx}): Merged {len(merged_pts)} points from {len(all_points)} pairs.')
    print(f'  Camera poses recovered: {len(camera_poses)}/{end_idx - start_idx + 1}')
    
    return merged_pts, merged_cols