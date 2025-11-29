from typing import Dict, List, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares

# Use absolute imports so notebooks that add `src` to sys.path can import `sfm` directly.
from reconstruction import (
    calculate_essential_matrix_from_points,
    recover_camera_pose,
    triangulate_points,
)


class IncrementalSfM:
    def __init__(self, images: List[np.ndarray], K: np.ndarray):
        self.images = images
        self.K = K.astype(np.float32)

        # State
        self.point_cloud: List[np.ndarray] = []
        self.point_colors: List[List[int]] = []
        self.camera_poses: List[Tuple[np.ndarray, np.ndarray]] = []
        self.image_pose_index: Dict[int, int] = {}
        self.matches_2d_3d: Dict[Tuple[int, int], int] = {}

        # Cache features per image
        self.features = [self.detect_features(img) for img in images]

    def detect_features(self, img: np.ndarray):
        sift = cv2.SIFT_create(nfeatures=5000)
        kp, des = sift.detectAndCompute(img, None)
        return kp or [], des

    def get_matches(self, idx1: int, idx2: int, ratio: float = 0.75) -> List[cv2.DMatch]:
        kp1, des1 = self.features[idx1]
        kp2, des2 = self.features[idx2]
        if des1 is None or des2 is None:
            return []
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=75)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        try:
            matches = flann.knnMatch(des1, des2, k=2)
        except cv2.error:
            return []
        good: List[cv2.DMatch] = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good.append(m)
        return good

    def _register_pose(self, image_idx: int, R: np.ndarray, t: np.ndarray) -> None:
        if image_idx in self.image_pose_index:
            self.camera_poses[self.image_pose_index[image_idx]] = (R, t)
        else:
            self.image_pose_index[image_idx] = len(self.camera_poses)
            self.camera_poses.append((R, t))

    def _grab_color(self, img: np.ndarray, pt: Tuple[float, float]) -> List[int]:
        x = min(max(int(round(pt[0])), 0), img.shape[1] - 1)
        y = min(max(int(round(pt[1])), 0), img.shape[0] - 1)
        pixel = img[y, x]
        if np.ndim(pixel) == 0:
            val = int(pixel)
            return [val, val, val]
        return [int(c) for c in pixel[:3]]

    def _point_in_front(self, point: np.ndarray, pose_a, pose_b) -> bool:
        R_a, t_a = pose_a
        R_b, t_b = pose_b
        pt = point.reshape(3, 1)
        depth_a = (R_a @ pt + t_a)[2]
        depth_b = (R_b @ pt + t_b)[2]
        return bool(depth_a > 0 and depth_b > 0)

    def initialize_structure(self, idx1: int = 0, idx2: int = 1) -> None:
        matches = self.get_matches(idx1, idx2)
        if len(matches) < 20:
            raise RuntimeError("Insufficient matches for initialization.")

        kp1, _ = self.features[idx1]
        kp2, _ = self.features[idx2]
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        idxs1 = np.array([m.queryIdx for m in matches])
        idxs2 = np.array([m.trainIdx for m in matches])

        E, mask, pts1_in, pts2_in = calculate_essential_matrix_from_points(pts1, pts2, self.K)
        if mask is None:
            raise RuntimeError("Essential matrix estimation failed.")
        mask_b = mask.ravel() > 0
        pts1_in = pts1_in[mask_b]
        pts2_in = pts2_in[mask_b]
        idxs1 = idxs1[mask_b]
        idxs2 = idxs2[mask_b]
        if len(pts1_in) < 10:
            raise RuntimeError("Too few inliers after RANSAC for initialization.")

        R, t, pose_mask = recover_camera_pose(E, pts1_in, pts2_in, self.K)
        pose_mask_b = pose_mask.ravel() > 0
        pts1_final = pts1_in[pose_mask_b]
        pts2_final = pts2_in[pose_mask_b]
        idxs1 = idxs1[pose_mask_b]
        idxs2 = idxs2[pose_mask_b]
        if len(pts1_final) == 0:
            raise RuntimeError("Cheirality check rejected all points.")

        points_3d = triangulate_points(R, t, self.K, pts1_final, pts2_final)
        finite_mask = np.isfinite(points_3d).all(axis=1)
        pts1_final = pts1_final[finite_mask]
        idxs1 = idxs1[finite_mask]
        idxs2 = idxs2[finite_mask]
        points_3d = points_3d[finite_mask]

        # Register poses
        self._register_pose(idx1, np.eye(3), np.zeros((3, 1)))
        self._register_pose(idx2, R, t)

        # Seed map
        for p3d, pt1, idx_q, idx_t in zip(points_3d, pts1_final, idxs1, idxs2):
            color = self._grab_color(self.images[idx1], pt1)
            p3d_idx = len(self.point_cloud)
            self.point_cloud.append(p3d)
            self.point_colors.append(color)
            self.matches_2d_3d[(idx1, idx_q)] = p3d_idx
            self.matches_2d_3d[(idx2, idx_t)] = p3d_idx

    def process_next_view(self, new_idx: int) -> bool:
        available_prev = [idx for idx in range(new_idx) if idx in self.image_pose_index]
        if not available_prev:
            return False

        kp_curr, _ = self.features[new_idx]
        if len(kp_curr) == 0:
            return False

        object_points: List[np.ndarray] = []
        image_points: List[Tuple[float, float]] = []
        pnp_match_sources: List[Tuple[int, cv2.DMatch]] = []
        triangulation_map: Dict[int, List[cv2.DMatch]] = {}

        for prev_idx in available_prev:
            matches = self.get_matches(prev_idx, new_idx)
            if not matches:
                continue
            kp_prev, _ = self.features[prev_idx]
            if len(kp_prev) == 0:
                continue
            for m in matches:
                key = (prev_idx, m.queryIdx)
                if key in self.matches_2d_3d:
                    object_points.append(self.point_cloud[self.matches_2d_3d[key]])
                    image_points.append(kp_curr[m.trainIdx].pt)
                    pnp_match_sources.append((prev_idx, m))
                else:
                    triangulation_map.setdefault(prev_idx, []).append(m)

        if len(object_points) < 6:
            return False

        object_points_np = np.asarray(object_points, dtype=np.float32)
        image_points_np = np.asarray(image_points, dtype=np.float32)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points_np,
            image_points_np,
            self.K,
            None,
            iterationsCount=300,
            reprojectionError=3.0,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success or inliers is None or len(inliers) < 6:
            return False

        R_curr, _ = cv2.Rodrigues(rvec)
        t_curr = tvec.reshape(3, 1)
        self._register_pose(new_idx, R_curr, t_curr)

        # Wire inlier associations
        for idx in inliers.flatten():
            prev_idx, match = pnp_match_sources[idx]
            p3d_idx = self.matches_2d_3d[(prev_idx, match.queryIdx)]
            self.matches_2d_3d[(new_idx, match.trainIdx)] = p3d_idx

        # Triangulate new points
        total_added = 0
        for prev_idx, match_list in triangulation_map.items():
            added = self._triangulate_new_points(prev_idx, new_idx, match_list)
            total_added += added
        return True

    def _triangulate_new_points(self, prev_idx: int, curr_idx: int, matches: List[cv2.DMatch]) -> int:
        if len(matches) < 4:
            return 0
        kp_prev, _ = self.features[prev_idx]
        kp_curr, _ = self.features[curr_idx]
        pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in matches]).T
        pts_curr = np.float32([kp_curr[m.trainIdx].pt for m in matches]).T

        pose_prev = self.camera_poses[self.image_pose_index[prev_idx]]
        pose_curr = self.camera_poses[self.image_pose_index[curr_idx]]
        P_prev = self.K @ np.hstack((pose_prev[0], pose_prev[1]))
        P_curr = self.K @ np.hstack((pose_curr[0], pose_curr[1]))

        pts4d = cv2.triangulatePoints(P_prev, P_curr, pts_prev, pts_curr)
        points_3d = (pts4d[:3] / pts4d[3]).T

        added = 0
        for p3d, match, pt_curr in zip(points_3d, matches, pts_curr.T):
            if not np.all(np.isfinite(p3d)):
                continue
            if not self._point_in_front(p3d, pose_prev, pose_curr):
                continue
            p3d_idx = len(self.point_cloud)
            self.point_cloud.append(p3d)
            self.point_colors.append(self._grab_color(self.images[curr_idx], pt_curr))
            self.matches_2d_3d[(prev_idx, match.queryIdx)] = p3d_idx
            self.matches_2d_3d[(curr_idx, match.trainIdx)] = p3d_idx
            added += 1
        return added

    def save_ply(self, filename: str) -> None:
        with open(filename, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(self.point_cloud)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for p, c in zip(self.point_cloud, self.point_colors):
                f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


# ---------- Bundle Adjustment (optional refinement) ----------

def _rotate(points: np.ndarray, rot_vecs: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid="ignore"):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return cos_theta * points + sin_theta * np.cross(v, points) + (1 - cos_theta) * dot * v


def _project(points: np.ndarray, camera_params: np.ndarray, K: np.ndarray) -> np.ndarray:
    # camera_params: [rvec(3), tvec(3)]
    points_proj = _rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    # Normalize
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    # Apply intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    points_proj[:, 0] = points_proj[:, 0] * fx + cx
    points_proj[:, 1] = points_proj[:, 1] * fy + cy
    return points_proj


def _ba_residuals(
    params: np.ndarray,
    n_cameras: int,
    n_points: int,
    camera_indices: np.ndarray,
    point_indices: np.ndarray,
    points_2d: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    camera_params = params[: n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6 :].reshape((n_points, 3))
    points_proj = _project(points_3d[point_indices], camera_params[camera_indices], K)
    return (points_proj - points_2d).ravel()


def run_bundle_adjustment(sfm_system: IncrementalSfM) -> IncrementalSfM:
    n_cameras = len(sfm_system.camera_poses)
    n_points = len(sfm_system.point_cloud)
    if n_cameras < 2 or n_points < 10:
        return sfm_system

    camera_params: List[np.ndarray] = []
    for R, t in sfm_system.camera_poses:
        rvec, _ = cv2.Rodrigues(R)
        camera_params.append(np.hstack((rvec.flatten(), t.flatten())))
    camera_params_np = np.array(camera_params).reshape((n_cameras, 6))
    points_3d_np = np.array(sfm_system.point_cloud)

    camera_indices: List[int] = []
    point_indices: List[int] = []
    points_2d: List[Tuple[float, float]] = []

    pose_lookup = getattr(sfm_system, "image_pose_index", {})
    for (img_idx, feat_idx), p3d_idx in sfm_system.matches_2d_3d.items():
        pose_idx = pose_lookup.get(img_idx)
        if pose_idx is None or p3d_idx >= n_points:
            continue
        kp, _ = sfm_system.features[img_idx]
        if feat_idx >= len(kp):
            continue
        camera_indices.append(pose_idx)
        point_indices.append(p3d_idx)
        points_2d.append(kp[feat_idx].pt)

    camera_indices_np = np.array(camera_indices)
    point_indices_np = np.array(point_indices)
    points_2d_np = np.array(points_2d)
    if len(points_2d_np) < max(2 * n_cameras, 20):
        return sfm_system

    x0 = np.hstack((camera_params_np.ravel(), points_3d_np.ravel()))
    res = least_squares(
        _ba_residuals,
        x0,
        verbose=0,
        x_scale="jac",
        ftol=1e-4,
        method="trf",
        max_nfev=45,
        args=(n_cameras, n_points, camera_indices_np, point_indices_np, points_2d_np, sfm_system.K),
    )

    optimized_params = res.x
    new_camera_params = optimized_params[: n_cameras * 6].reshape((n_cameras, 6))
    new_points_3d = optimized_params[n_cameras * 6 :].reshape((n_points, 3))

    sfm_system.point_cloud = list(new_points_3d)
    updated_poses: List[Tuple[np.ndarray, np.ndarray]] = []
    for cp in new_camera_params:
        R, _ = cv2.Rodrigues(cp[:3])
        t = cp[3:].reshape(3, 1)
        updated_poses.append((R, t))
    sfm_system.camera_poses = updated_poses

    return sfm_system


__all__ = ["IncrementalSfM", "run_bundle_adjustment"]
