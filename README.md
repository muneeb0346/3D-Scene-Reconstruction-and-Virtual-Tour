# 3D-Scene-Reconstruction-and-Virtual-Tour
This is an implementation of a complete Structure from Motion (SfM) and visualization pipeline for the CS436: Computer Vision Fundamentals course. The goal is to take a collection of 2D photographs of a static scene and transform them into a sparse 3D point cloud and an interactive "virtual tour" viewer, inspired by applications like Photosynth.

## Phase 3: Three.js Virtual Tour

A lightweight, dependency-free viewer lives in `web/` to explore the reconstructed scene (phase 2 output) as a virtual tour.

### Quick start

1. Start a static server from the repository root:
	 ```pwsh
	 python -m http.server 8000 --directory web
	 ```
2. Open http://localhost:8000 in a browser.
3. Load your reconstructed PLY (with or without colors) using **Load PLY**, paste a remote URL, drag & drop, or click **Sample Cloud**.
4. (Optional) Load camera poses (`poses.json`) via **Load Poses** or **Sample Poses**, then press **Play Tour** to fly through viewpoints.

### Pose JSON schema

```json
{
	"cameras": [
		{ "name": "cam0", "position": [x, y, z], "target": [tx, ty, tz] }
	]
}
```

- If you have extrinsics, the viewer also accepts `{ "R": [[...],[...],[...]], "t": [tx, ty, tz] }` and converts to camera centers using `C = -R^T t`.
- `target` is optional; defaults to the scene origin.

### Controls

- Orbit: drag with mouse / trackpad
- Pan: right-drag or two-finger drag
- Zoom: scroll / pinch
- Auto-rotate toggle, grid toggle, point size slider, reset view, and screenshot capture are built in.

### Sample data

- `web/data/demo_point_cloud.ply` — tiny colored cube point cloud for smoke testing.
- `web/data/sample_poses.json` — four camera viewpoints around the origin to demo the tour playback.

### Integrating your reconstruction

- Export your phase 2 colored point cloud to a PLY file (ASCII or binary). The viewer will display vertex colors when present.
- Save camera centers/targets to `poses.json` following the schema above, place it in `web/data/`, and load it from the UI.
- Large PLY files load faster when served over HTTP rather than `file://`, so prefer the local server command above.
