# WA Perception Challenge: Ego Trajectory Estimation in Bird’s-Eye View

This repository contains a Python solution for the Wisconsin Autonomous Perception Challenge: estimating the ego-vehicle trajectory from a traffic light observation, using the traffic light as the world reference and rendering the result in a bird’s-eye view (BEV).

## What the application does

The program in `main.py` implements the following workflow:

1. Reads traffic-light bounding boxes from the CSV file.
   - Expected columns: `frame_id, x_min, y_min, x_max, y_max`
   - It normalizes them to `frame, x1, y1, x2, y2`.

2. Computes the traffic-light center in each frame.
   - The center of the box is treated as the pixel location of the traffic light.

3. Loads the corresponding 3D point cloud (`.npz`) for each frame.
   - It supports files named like:
     - `frame_0001.npz`
     - `frame_0000.npz`
     - `depth000001.npz`
   - It looks for the `points` key or `xyz` key, or falls back to the first available array.

4. Extracts the 3D position of the traffic light in camera coordinates.
   - It takes a small patch around the bounding-box center (default: 5x5 pixels).
   - It removes invalid depth values and locally inconsistent forward-range samples before computing a robust mean 3D point `(X, Y, Z)` in camera space.

5. Performs a conservative temporal consistency check.
   - Since the traffic light is static, isolated large camera-relative 3D jumps are treated as depth/detection outliers.
   - The threshold is adaptive (median displacement rate plus a MAD-based tolerance) and accounts for gaps in frame IDs.

6. Converts the traffic light’s relative motion into ego-vehicle trajectory.
   - The car is considered to be at the camera origin.
   - The traffic light is used as the world reference point.
   - The script rotates the coordinate system so that the mean bearing of the first few valid frames aligns the line from the car to the traffic light with the +X axis. This is a repeatable choice of BEV-axis orientation, not an external pose calibration.
   - This produces BEV coordinates `x_m` and `y_m` representing the ego trajectory on the ground plane.

6. Saves the outputs:
   - `trajectory.png`: static BEV trajectory plot
   - `trajectory.mp4`: animated BEV trajectory video

## Repository structure

```text
.
├── main.py
├── README.md
├── dataset/
│   ├── rgb/
│   ├── xyz/
│   └── bbox_light.csv
├── trajectory.png
├── trajectory.mp4
└── ...
```

## Key implementation details

The code is designed around a straightforward geometry-based solution rather than a learned model:

- `load_bboxes()`: reads and validates the detection CSV.
- `bbox_center()`: computes the box center.
- `patch_mean_xyz()`: averages valid 3D points around the traffic-light center, reducing noise.
- `traffic_light_xyz()`: maps each frame to a 3D traffic-light point in camera coordinates.
- `ego_trajectory()`: transforms those camera-frame positions into a ground-centered BEV trajectory.
- `plot_trajectory()`: saves a static plot.
- `animate_trajectory()`: saves an animation of the estimated motion over time.

## Exact result produced by `main.py`

Running:

```bash
python main.py
```

will:

- load the traffic-light boxes,
- estimate the 3D location of the traffic light in each frame,
- derive the ego-vehicle’s motion relative to that light,
- generate the visual outputs in the repository root.

This is a classic geometry challenge solution: estimate the motion from the apparent motion of a static world reference point (the traffic light), then plot that motion in a bird’s-eye-view coordinate system.

## Setup

Install the dependencies:

```bash
pip install numpy pandas matplotlib opencv-python
```

Then place the dataset in the expected structure or ensure the CSV and `xyz` folder are in the project root. The script accepts both of these layouts:

- `dataset/xyz/...` and `dataset/bbox_light.csv`
- root-level `xyz/...` and `bbox_light.csv`

## Outputs

The solution generates:

- `trajectory.png`: a BEV plot of the ego trajectory
- `trajectory.mp4`: an animation of the trajectory over time

These outputs are intended to communicate the estimated vehicle path in a ground-fixed frame, with the traffic light acting as the origin.

## Quantitative validation metrics

At the end of every `main.py` execution, the program reports reproducible trajectory metrics calculated only from frames that have a valid traffic-light box, matching `.npz` file, and finite 5x5 XYZ depth sample:

- **Frames processed successfully:** `valid_frames / total_csv_frames` and percentage. This is a data-availability and depth-validity measure; it does not claim that excluded frames were necessarily visually noisy.
- **Raw valid depth samples / temporal outliers rejected:** reports the effect of the consistency gate explicitly instead of silently changing the trajectory.
- **Total distance travelled (m):** sum of Euclidean distances between consecutive valid BEV positions.
- **Mean / maximum valid-frame displacement (m):** descriptive checks that help identify discontinuities or outlier depth estimates.
- **Mean speed (m/s):** calculated only when the actual camera acquisition rate is supplied in `SOURCE_FPS`. The 10 FPS used to encode `trajectory.mp4` is deliberately not used as a camera rate, since it is a visualization setting rather than sensor metadata.

For the dataset currently included in this repository, the trajectory computation produced:

```text
Frames processed successfully: 125 / 299 (41.8%)
Raw valid depth samples: 125
Temporal outliers rejected: 0
Total distance travelled: 22.21 m
Mean valid-frame displacement: 0.179 m
Maximum valid-frame displacement: 1.242 m
Mean speed: not calculated (set SOURCE_FPS to the camera acquisition rate)
```

To report estimated speed, set `SOURCE_FPS` near the constants at the top of `main.py` to the documented capture frequency of the source sequence. For example, `SOURCE_FPS = 10.0` would treat adjacent frame IDs as 0.1 seconds apart.

## Code organization

Keeping this challenge in `main.py` is appropriate: it is a small, linear pipeline and its geometry remains easy to audit in one file. If the project grows, split it by stable responsibilities rather than prematurely: `io.py` for dataset loading, `geometry.py` for BEV transforms and metrics, `barrels.py` for OpenCV detection, and `visualization.py` for plots/animation. Add tests for `patch_mean_xyz`, `camera_point_to_world_xy`, and `trajectory_metrics` at the same time.

## Summary

This project is not a general perception system; it is a focused computer-vision / geometry task that estimates the car’s ego trajectory using the traffic light’s projected 3D position and a traffic-light-centered BEV transformation.

The core idea is simple and robust:

- traffic light = static reference point,
- its movement across frames reveals the ego motion,
- transform to a ground frame,
- visualize the trajectory.
