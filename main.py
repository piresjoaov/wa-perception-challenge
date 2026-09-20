"""Part A: ego-vehicle trajectory in a traffic-light-centered BEV."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from matplotlib.animation import FFMpegWriter, FuncAnimation

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
CSV_PATH = DATASET_DIR / "bbox_light.csv" if (DATASET_DIR / "bbox_light.csv").exists() else ROOT / "bbox_light.csv"
XYZ_DIR = DATASET_DIR / "xyz" if (DATASET_DIR / "xyz").exists() else ROOT / "xyz"
RGB_DIR = DATASET_DIR / "rgb" if (DATASET_DIR / "rgb").exists() else ROOT / "rgb"
PNG_PATH = ROOT / "output" / "trajectory.png"
MP4_PATH = ROOT / "output" / "trajectory.mp4"
PATCH = 5  # 5x5 window around (u, v)
FPS = 10
# Set this to the acquisition rate of the source sequence (not the MP4 FPS)
# when it is known.  It is left unset to avoid reporting an invented speed.
SOURCE_FPS: float | None = None
# Conservative orange range for daytime traffic barrels. Tune these if a
# different camera, lighting, or barrel paint is used.
BARREL_HSV_LOWER = np.array([0, 70, 70], dtype=np.uint8)
BARREL_HSV_UPPER = np.array([25, 255, 255], dtype=np.uint8)
MIN_BARREL_AREA_PX = 250


def load_bboxes(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rename = {
        "frame_id": "frame",
        "x_min": "x1",
        "y_min": "y1",
        "x_max": "x2",
        "y_max": "y2",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    required = {"frame", "x1", "y1", "x2", "y2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    return df


def bbox_center(row: pd.Series) -> tuple[float, float] | None:
    x1, y1, x2, y2 = map(float, (row.x1, row.y1, row.x2, row.y2))
    if (x1, y1, x2, y2) == (0.0, 0.0, 0.0, 0.0):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def npz_path(frame_id: int) -> Path | None:
    candidates = [
        XYZ_DIR / f"depth{frame_id:06d}.npz",
        XYZ_DIR / f"frame_{frame_id:04d}.npz",
        XYZ_DIR / f"frame_{frame_id:06d}.npz",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_xyz(path: Path) -> np.ndarray:
    with np.load(path) as data:
        if "points" in data:
            arr = data["points"]
        elif "xyz" in data:
            arr = data["xyz"]
        else:
            arr = data[data.files[0]]
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"Unexpected array in {path}: shape={arr.shape}")
    return arr[..., :3]


def rgb_path(frame_id: int) -> Path | None:
    """Return the RGB image path for a frame, accepting common conventions."""
    candidates = [
        RGB_DIR / f"frame_{frame_id:04d}.png",
        RGB_DIR / f"frame_{frame_id:06d}.png",
        RGB_DIR / f"left{frame_id:04d}.png",
        RGB_DIR / f"left{frame_id:06d}.png",
    ]
    return next((path for path in candidates if path.exists()), None)


def load_rgb_image(frame_id: int) -> np.ndarray | None:
    """Load an RGB frame as OpenCV's BGR array; return None when unavailable."""
    path = rgb_path(frame_id)
    if path is None:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return image if image is not None else None


def detect_barrels_hsv(image: np.ndarray | None) -> list[tuple[float, float]]:
    """Detect orange barrel-like regions and return their contour centres (u, v).

    The filters reject small colour noise and wide, horizontal orange objects
    such as road barriers.  They intentionally return an empty list when the
    frame is unavailable or no reliable contour is present.
    """
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        return []

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BARREL_HSV_LOWER, BARREL_HSV_UPPER)
    open_kernel = np.ones((3, 3), dtype=np.uint8)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_width = image.shape[1]
    centres: list[tuple[float, float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_BARREL_AREA_PX:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        aspect = height / max(width, 1)
        # A barrel is compact and normally taller than it is wide.  The width
        # cap prevents a long orange barrier from becoming a false barrel.
        if not 0.55 <= aspect <= 4.5 or width > 0.12 * image_width:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        centres.append((moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]))
    return centres


def patch_mean_xyz(xyz: np.ndarray, u: float, v: float, size: int = PATCH) -> np.ndarray | None:
    """Mean X,Y,Z in a size x size window around pixel (u, v)."""
    h, w, _ = xyz.shape
    ui = int(round(u))
    vi = int(round(v))
    half = size // 2
    v0, v1 = max(0, vi - half), min(h, vi + half + 1)
    u0, u1 = max(0, ui - half), min(w, ui + half + 1)
    patch = xyz[v0:v1, u0:u1].reshape(-1, 3)
    finite = np.isfinite(patch).all(axis=1)
    # Drop non-positive forward range (invalid / background holes).
    finite &= patch[:, 0] > 0
    if not np.any(finite):
        return None
    return patch[finite].mean(axis=0)


def traffic_light_xyz(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in df.itertuples(index=False):
        center = bbox_center(row)
        if center is None:
            continue
        u, v = center
        path = npz_path(int(row.frame))
        if path is None:
            continue
        xyz = load_xyz(path)
        mean_xyz = patch_mean_xyz(xyz, u, v)
        if mean_xyz is None:
            continue
        x_cam, y_cam, z_cam = mean_xyz.tolist()
        rows.append(
            {
                "frame": int(row.frame),
                "u": u,
                "v": v,
                "X": x_cam,
                "Y": y_cam,
                "Z": z_cam,
            }
        )
    if not rows:
        raise RuntimeError("No valid traffic-light 3D samples found.")
    return pd.DataFrame(rows)


def world_rotation(tl: pd.DataFrame) -> tuple[float, float]:
    """Return the fixed 2-D rotation used by the traffic-light BEV frame."""
    theta = np.arctan2(tl["Y"].iloc[0], tl["X"].iloc[0])
    return float(np.cos(theta)), float(np.sin(theta))


def camera_point_to_world_xy(
    x_cam: float,
    y_cam: float,
    light_x_cam: float,
    light_y_cam: float,
    c: float,
    s: float,
) -> tuple[float, float]:
    """Map a camera-frame ground point into the traffic-light-centred BEV."""
    dx = x_cam - light_x_cam
    dy = y_cam - light_y_cam
    return c * dx + s * dy, -s * dx + c * dy


def ego_trajectory(tl: pd.DataFrame) -> pd.DataFrame:
    """Place the traffic light at the ground origin.

    Camera frame: +X forward, +Y right, +Z up.
    Ego (camera) is at the camera origin, so in a frame centered on the
    light the car is at (-X, -Y) on the ground plane.

    At the first valid frame, rotate around Z so the car-to-light line
    aligns with +X (world forward).
    """
    out = tl.copy()
    c, s = world_rotation(out)
    # The ego point is (0, 0) in each camera frame.  Applying the same
    # transform as any other point keeps the formula shared with barrels.
    out[["x_m", "y_m"]] = [
        camera_point_to_world_xy(0.0, 0.0, row.X, row.Y, c, s)
        for row in out.itertuples(index=False)
    ]
    return out


def barrels_to_world(traj: pd.DataFrame) -> pd.DataFrame:
    """Detect and project barrel observations into the established BEV frame."""
    columns = ["frame", "u", "v", "X", "Y", "Z", "x_m", "y_m"]
    c, s = world_rotation(traj)
    rows: list[dict[str, float | int]] = []

    # Iterate only frames with a valid traffic-light depth observation: these
    # provide the per-frame camera-to-light translation needed for projection.
    for row in traj.itertuples(index=False):
        image = load_rgb_image(int(row.frame))
        centres = detect_barrels_hsv(image)
        if not centres:
            continue
        path = npz_path(int(row.frame))
        if path is None:
            continue
        try:
            xyz = load_xyz(path)
        except (OSError, ValueError, KeyError):
            continue

        for u, v in centres:
            point = patch_mean_xyz(xyz, u, v)
            if point is None:
                continue
            x_cam, y_cam, z_cam = map(float, point)
            x_world, y_world = camera_point_to_world_xy(x_cam, y_cam, row.X, row.Y, c, s)
            rows.append(
                {
                    "frame": int(row.frame),
                    "u": u,
                    "v": v,
                    "X": x_cam,
                    "Y": y_cam,
                    "Z": z_cam,
                    "x_m": x_world,
                    "y_m": y_world,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def trajectory_metrics(
    traj: pd.DataFrame,
    total_frames: int,
    source_fps: float | None = SOURCE_FPS,
) -> dict[str, float | int | None]:
    """Return reproducible validation statistics for valid trajectory samples."""
    xy = traj[["x_m", "y_m"]].to_numpy(dtype=float)
    step_distances = np.linalg.norm(np.diff(xy, axis=0), axis=1) if len(xy) > 1 else np.array([])
    total_distance = float(step_distances.sum())
    valid_frames = len(traj)
    metrics: dict[str, float | int | None] = {
        "valid_frames": valid_frames,
        "total_frames": total_frames,
        "valid_percent": 100.0 * valid_frames / total_frames if total_frames else 0.0,
        "total_distance_m": total_distance,
        "mean_step_m": float(step_distances.mean()) if len(step_distances) else 0.0,
        "max_step_m": float(step_distances.max()) if len(step_distances) else 0.0,
        "mean_speed_mps": None,
    }
    if source_fps is not None and source_fps > 0:
        # Frame IDs may have gaps, so use their actual temporal separation.
        frame_gaps = np.diff(traj["frame"].to_numpy(dtype=float))
        duration_s = float((traj["frame"].iloc[-1] - traj["frame"].iloc[0]) / source_fps)
        if duration_s > 0 and np.all(frame_gaps > 0):
            metrics["mean_speed_mps"] = total_distance / duration_s
    return metrics


def bev_limits(x: np.ndarray, y: np.ndarray, pad_frac: float = 0.08) -> tuple[float, float, float, float]:
    """Fixed equal-aspect limits covering the full trajectory and the origin."""
    xs = np.concatenate([x, [0.0]])
    ys = np.concatenate([y, [0.0]])
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    dx = max(xmax - xmin, 1.0)
    dy = max(ymax - ymin, 1.0)
    xmin -= pad_frac * dx
    xmax += pad_frac * dx
    ymin -= pad_frac * dy
    ymax += pad_frac * dy
    dx, dy = xmax - xmin, ymax - ymin
    if dx > dy:
        extra = (dx - dy) / 2.0
        ymin -= extra
        ymax += extra
    else:
        extra = (dy - dx) / 2.0
        xmin -= extra
        xmax += extra
    return xmin, xmax, ymin, ymax


def style_bev_axes(ax: plt.Axes, limits: tuple[float, float, float, float]) -> None:
    xmin, xmax, ymin, ymax = limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Forward (X, m)")
    ax.set_ylabel("Lateral (Y, m)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.axhline(0, color="k", linewidth=0.6)
    ax.axvline(0, color="k", linewidth=0.6)


def plot_trajectory(
    traj: pd.DataFrame,
    output_path: Path,
    barrels: pd.DataFrame | None = None,
) -> tuple[float, float, float, float]:
    x = traj["x_m"].to_numpy()
    y = traj["y_m"].to_numpy()
    if barrels is not None and not barrels.empty:
        limits = bev_limits(
            np.concatenate([x, barrels["x_m"].to_numpy()]),
            np.concatenate([y, barrels["y_m"].to_numpy()]),
        )
    else:
        limits = bev_limits(x, y)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x, y, "-", color="C0", linewidth=1.5, alpha=0.7, label="Ego trajectory")
    ax.scatter(x, y, s=12, color="C0", zorder=3)
    ax.scatter([x[0]], [y[0]], marker="x", s=80, color="red", zorder=4, label="Start")
    ax.scatter([x[-1]], [y[-1]], s=60, color="green", zorder=4, label="End")
    ax.scatter([0], [0], marker="*", s=160, color="black", zorder=5, label="Traffic light (origin)")
    if barrels is not None and not barrels.empty:
        ax.scatter(
            barrels["x_m"],
            barrels["y_m"],
            s=28,
            color="orange",
            marker="o",
            edgecolors="saddlebrown",
            linewidths=0.4,
            alpha=0.65,
            zorder=4,
            label="Barrel observations",
        )
    style_bev_axes(ax, limits)
    ax.set_title("Ego Trajectory (BEV, traffic-light origin)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return limits


def animate_trajectory(
    traj: pd.DataFrame,
    output_path: Path,
    limits: tuple[float, float, float, float],
    barrels: pd.DataFrame | None = None,
    fps: int = FPS,
) -> None:
    x = traj["x_m"].to_numpy()
    y = traj["y_m"].to_numpy()
    frame_ids = traj["frame"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    (line,) = ax.plot([], [], "-", color="C0", linewidth=1.5, alpha=0.7, label="Ego trajectory")
    pts = ax.scatter([], [], s=12, color="C0", zorder=3)
    ego = ax.scatter([], [], s=90, color="green", edgecolors="k", zorder=6, label="Ego")
    barrel_pts = ax.scatter(
        [], [], s=28, color="orange", marker="o", edgecolors="saddlebrown",
        linewidths=0.4, zorder=4, label="Barrel observations"
    )
    ax.scatter([x[0]], [y[0]], marker="x", s=80, color="red", zorder=4, label="Start")
    ax.scatter([0], [0], marker="*", s=160, color="black", zorder=5, label="Traffic light (origin)")
    style_bev_axes(ax, limits)
    ax.legend(loc="best")
    title = ax.set_title("Ego Trajectory (BEV)")
    fig.tight_layout()

    def init():
        line.set_data([], [])
        pts.set_offsets(np.empty((0, 2)))
        ego.set_offsets(np.empty((0, 2)))
        barrel_pts.set_offsets(np.empty((0, 2)))
        return line, pts, ego, barrel_pts, title

    def update(i: int):
        line.set_data(x[: i + 1], y[: i + 1])
        pts.set_offsets(np.column_stack((x[: i + 1], y[: i + 1])))
        ego.set_offsets(np.array([[x[i], y[i]]]))
        if barrels is not None and not barrels.empty:
            observed = barrels[barrels["frame"] <= frame_ids[i]]
            barrel_pts.set_offsets(observed[["x_m", "y_m"]].to_numpy())
        else:
            barrel_pts.set_offsets(np.empty((0, 2)))
        title.set_text(f"Ego Trajectory (BEV)  |  frame {int(frame_ids[i])}")
        return line, pts, ego, barrel_pts, title

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(x),
        interval=1000 / fps,
        blit=False,
    )
    writer = FFMpegWriter(fps=fps)
    anim.save(str(output_path), writer=writer, dpi=120)
    plt.close(fig)


def main() -> None:
    bboxes = load_bboxes(CSV_PATH)
    tl = traffic_light_xyz(bboxes)
    traj = ego_trajectory(tl)
    barrels = barrels_to_world(traj)
    limits = plot_trajectory(traj, PNG_PATH, barrels)
    animate_trajectory(traj, MP4_PATH, limits, barrels)
    metrics = trajectory_metrics(traj, total_frames=len(bboxes))
    print("\nTrajectory validation metrics")
    print(f"Frames processed successfully: {metrics['valid_frames']} / {metrics['total_frames']} "
          f"({metrics['valid_percent']:.1f}%)")
    print(f"Total distance travelled: {metrics['total_distance_m']:.2f} m")
    print(f"Mean valid-frame displacement: {metrics['mean_step_m']:.3f} m")
    print(f"Maximum valid-frame displacement: {metrics['max_step_m']:.3f} m")
    if metrics["mean_speed_mps"] is None:
        print("Mean speed: not calculated (set SOURCE_FPS to the camera acquisition rate)")
    else:
        print(f"Mean speed: {metrics['mean_speed_mps']:.2f} m/s")
    print(f"Start (x, y) = ({traj['x_m'].iloc[0]:.2f}, {traj['y_m'].iloc[0]:.2f}) m")
    print(f"End   (x, y) = ({traj['x_m'].iloc[-1]:.2f}, {traj['y_m'].iloc[-1]:.2f}) m")
    print(f"Barrel observations projected: {len(barrels)}")
    print(f"Saved {PNG_PATH}")
    print(f"Saved {MP4_PATH}")


if __name__ == "__main__":
    main()
