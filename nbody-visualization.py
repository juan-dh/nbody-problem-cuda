import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import imageio_ffmpeg as ffmpeg
import matplotlib as mpl

# Use ffmpeg installed through pip install imageio-ffmpeg
mpl.rcParams["animation.ffmpeg_path"] = ffmpeg.get_ffmpeg_exe()


def load_nbody_binary(filename: str):
    with open(filename, "rb") as f:
        n_bodies = np.fromfile(f, dtype=np.int32, count=1)[0]
        n_steps = np.fromfile(f, dtype=np.int32, count=1)[0]
        data = np.fromfile(f, dtype=np.float32)

    expected = (n_steps + 1) * n_bodies * 3
    if data.size != expected:
        raise ValueError(
            f"Unexpected file size.\n"
            f"Expected: {expected} floats\n"
            f"Found: {data.size} floats"
        )

    data = data.reshape((n_steps + 1, n_bodies, 3))
    return n_bodies, n_steps, data


def random_sample_particles(
    data: np.ndarray,
    max_particles: int | None = None,
    seed: int = 42,
):
    n_particles = data.shape[1]

    if max_particles is None or max_particles >= n_particles:
        return data, np.arange(n_particles)

    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(n_particles, size=max_particles, replace=False))
    sampled = data[:, indices, :]
    return sampled, indices


def sample_frames(data: np.ndarray, frame_stride: int = 1):
    return data[::frame_stride]


def auto_marker_size(n_particles: int) -> float:
    size = 4000.0 / np.sqrt(max(n_particles, 1))
    return max(0.2, min(size, 8.0))


def auto_alpha(n_particles: int) -> float:
    alpha = 4000.0 / max(n_particles, 1)
    return max(0.08, min(alpha, 0.5))


def visible_mask(pts: np.ndarray, xlim, ylim, zlim):
    return (
        (pts[:, 0] >= xlim[0]) & (pts[:, 0] <= xlim[1]) &
        (pts[:, 1] >= ylim[0]) & (pts[:, 1] <= ylim[1]) &
        (pts[:, 2] >= zlim[0]) & (pts[:, 2] <= zlim[1])
    )


def draw_axes(ax, xlim, ylim, zlim, scale=0.25, lw=1.0, alpha=0.9):
    x_len = max(0.0, xlim[1] * scale)
    y_len = max(0.0, ylim[1] * scale)
    z_len = max(0.0, zlim[1] * scale)

    # Positive X axis
    ax.plot([0, x_len], [0, 0], [0, 0], color="red", linewidth=lw, alpha=alpha)

    # Positive Y axis
    ax.plot([0, 0], [0, y_len], [0, 0], color="green", linewidth=lw, alpha=alpha)

    # Positive Z axis
    ax.plot([0, 0], [0, 0], [0, z_len], color="blue", linewidth=lw, alpha=alpha)


def create_figure(elev, azim, xlim, ylim, zlim, zoom):
    fig = plt.figure(figsize=(14, 8), facecolor="black")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection="3d")
    ax.set_facecolor("black")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    ax.set_proj_type("persp")
    ax.view_init(elev=elev, azim=azim)

    ax.set_box_aspect(
        (xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]),
        zoom=zoom,
    )

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_position([0.0, 0.0, 1.0, 1.0])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.grid(False)

    # Keep the GIF cleanup style as the shared look for both outputs.
    try:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor((0, 0, 0, 0))
        ax.yaxis.pane.set_edgecolor((0, 0, 0, 0))
        ax.zaxis.pane.set_edgecolor((0, 0, 0, 0))
    except Exception:
        pass

    try:
        ax.w_xaxis.line.set_color((0, 0, 0, 0))
        ax.w_yaxis.line.set_color((0, 0, 0, 0))
        ax.w_zaxis.line.set_color((0, 0, 0, 0))
    except Exception:
        pass

    draw_axes(ax, xlim, ylim, zlim, lw=1.0, alpha=0.9)
    return fig, ax


def get_writer(output_path: str, fps: int):
    extension = os.path.splitext(output_path)[1].lower()

    if extension == ".gif":
        writer = PillowWriter(fps=fps)
        savefig_kwargs = {
            "facecolor": "black",
            "pad_inches": 0,
        }
        output_label = "GIF"
    elif extension == ".mp4":
        writer = FFMpegWriter(
            fps=fps,
            bitrate=3000,
            metadata={"artist": "Juan Diego Haro"},
        )
        savefig_kwargs = {
            "facecolor": "black",
            "edgecolor": "black",
            "transparent": False,
            "pad_inches": 0,
        }
        output_label = "Video"
    else:
        raise ValueError(f"Unsupported output format: {extension}")

    return writer, savefig_kwargs, output_label


def make_animation(
    input_file: str = "nbody_data.bin",
    output_path: str = "nbody_visualization.gif",
    max_particles: int | None = None,
    frame_stride: int = 5,
    fps: int = 15,
    marker_size: float | None = None,
    alpha: float | None = None,
    elev: float = 20,
    azim: float = 45,
    dpi: int = 120,
    xlim: tuple[float, float] = (-150, 150),
    ylim: tuple[float, float] = (-150, 150),
    zlim: tuple[float, float] = (-150, 150),
    seed: int = 42,
    zoom: float = 1.9,
):
    if os.path.exists(output_path):
        os.remove(output_path)

    n_bodies, n_steps, data = load_nbody_binary(input_file)
    print(f"Original bodies: {n_bodies}")
    print(f"Original steps: {n_steps + 1}")

    data, _ = random_sample_particles(
        data,
        max_particles=max_particles,
        seed=seed,
    )
    data = sample_frames(data, frame_stride=frame_stride)

    n_render = data.shape[1]
    n_frames = data.shape[0]

    if marker_size is None:
        marker_size = auto_marker_size(n_render)

    if alpha is None:
        alpha = auto_alpha(n_render)

    print(f"Bodies used for rendering: {n_render}")
    print(f"Frames used for rendering: {n_frames}")

    fig, ax = create_figure(elev, azim, xlim, ylim, zlim, zoom)

    pts0 = data[0]
    mask0 = visible_mask(pts0, xlim, ylim, zlim)
    pts0_vis = pts0[mask0]

    scatter = ax.scatter(
        pts0_vis[:, 0] if len(pts0_vis) else [],
        pts0_vis[:, 1] if len(pts0_vis) else [],
        pts0_vis[:, 2] if len(pts0_vis) else [],
        s=marker_size,
        c="white",
        alpha=alpha,
        depthshade=False,
        edgecolors="none",
    )

    simulation_particles_text = fig.text(
        0.02,
        0.06,
        f"Simulation particles: {n_bodies}",
        color="white",
        fontsize=12,
        ha="left",
        va="bottom",
    )

    render_particles_text = fig.text(
        0.02,
        0.09,
        f"Render particles: {n_render}",
        color="white",
        fontsize=12,
        ha="left",
        va="bottom",
    )

    step_text = fig.text(
        0.02,
        0.03,
        "Step: 0",
        color="white",
        fontsize=12,
        ha="left",
        va="bottom",
    )

    def update(frame_idx):
        pts = data[frame_idx]
        mask = visible_mask(pts, xlim, ylim, zlim)
        pts_vis = pts[mask]

        if len(pts_vis) == 0:
            scatter._offsets3d = ([], [], [])
        else:
            scatter._offsets3d = (pts_vis[:, 0], pts_vis[:, 1], pts_vis[:, 2])

        original_frame = frame_idx * frame_stride
        step_text.set_text(f"Step: {original_frame}")
        return scatter, simulation_particles_text, render_particles_text, step_text

    anim = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000 / fps,
        blit=False,
    )

    writer, savefig_kwargs, output_label = get_writer(output_path, fps)
    anim.save(
        output_path,
        writer=writer,
        dpi=dpi,
        savefig_kwargs=savefig_kwargs,
    )

    plt.close(fig)
    print(f"{output_label} saved to: {output_path}")


if __name__ == "__main__":
    make_animation()
