"""Part A: ego-vehicle trajectory in a traffic-light-centered BEV."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
CSV_PATH = DATASET_DIR / "bbox_light.csv" if (DATASET_DIR / "bbox_light.csv").exists() else ROOT / "bbox_light.csv"
XYZ_DIR = DATASET_DIR / "xyz" if (DATASET_DIR / "xyz").exists() else ROOT / "xyz"
PNG_PATH = ROOT / "trajectory.png"
MP4_PATH = ROOT / "trajectory.mp4"
PATCH = 5  # 5x5 window around (u, v)
FPS = 10


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


def ego_trajectory(tl: pd.DataFrame) -> pd.DataFrame:
    """Place the traffic light at the ground origin.

    Camera frame: +X forward, +Y right, +Z up.
    Ego (camera) is at the camera origin, so in a frame centered on the
    light the car is at (-X, -Y) on the ground plane.

    At the first valid frame, rotate around Z so the car-to-light line
    aligns with +X (world forward).
    """
    out = tl.copy()
    ego_x_cam = -out["X"].to_numpy()
    ego_y_cam = -out["Y"].to_numpy()

    x0, y0 = out["X"].iloc[0], out["Y"].iloc[0]
    theta = np.arctan2(y0, x0)
    c, s = np.cos(theta), np.sin(theta)
    # R maps the t=0 camera (X, Y) of the light onto (+range, 0).
    out["x_m"] = c * ego_x_cam + s * ego_y_cam
    out["y_m"] = -s * ego_x_cam + c * ego_y_cam
    return out


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


def plot_trajectory(traj: pd.DataFrame, output_path: Path) -> tuple[float, float, float, float]:
    x = traj["x_m"].to_numpy()
    y = traj["y_m"].to_numpy()
    limits = bev_limits(x, y)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x, y, "-", color="C0", linewidth=1.5, alpha=0.7, label="Ego trajectory")
    ax.scatter(x, y, s=12, color="C0", zorder=3)
    ax.scatter([x[0]], [y[0]], marker="x", s=80, color="red", zorder=4, label="Start")
    ax.scatter([x[-1]], [y[-1]], s=60, color="green", zorder=4, label="End")
    ax.scatter([0], [0], marker="*", s=160, color="black", zorder=5, label="Traffic light (origin)")
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
    fps: int = FPS,
) -> None:
    x = traj["x_m"].to_numpy()
    y = traj["y_m"].to_numpy()
    frame_ids = traj["frame"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    (line,) = ax.plot([], [], "-", color="C0", linewidth=1.5, alpha=0.7, label="Ego trajectory")
    pts = ax.scatter([], [], s=12, color="C0", zorder=3)
    ego = ax.scatter([], [], s=90, color="green", edgecolors="k", zorder=6, label="Ego")
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
        return line, pts, ego, title

    def update(i: int):
        line.set_data(x[: i + 1], y[: i + 1])
        pts.set_offsets(np.column_stack((x[: i + 1], y[: i + 1])))
        ego.set_offsets(np.array([[x[i], y[i]]]))
        title.set_text(f"Ego Trajectory (BEV)  |  frame {int(frame_ids[i])}")
        return line, pts, ego, title

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
    limits = plot_trajectory(traj, PNG_PATH)
    animate_trajectory(traj, MP4_PATH, limits)
    print(f"Frames used: {len(traj)} / {len(bboxes)}")
    print(f"Start (x, y) = ({traj['x_m'].iloc[0]:.2f}, {traj['y_m'].iloc[0]:.2f}) m")
    print(f"End   (x, y) = ({traj['x_m'].iloc[-1]:.2f}, {traj['y_m'].iloc[-1]:.2f}) m")
    print(f"Saved {PNG_PATH}")
    print(f"Saved {MP4_PATH}")


if __name__ == "__main__":
    main()
