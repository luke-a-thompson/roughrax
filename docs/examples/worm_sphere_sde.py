"""Brownian motion on a visible spherical cap via SO(3).

This evolves an SO(3)-valued SDE, then displays the rotation applied to the
north pole as a path on S^2. The same sampled Brownian path drives both solves:

* Georax `GeometricEuler` uses all fine Brownian increments.
* Roughrax `LogODE(RKMK(Tsit5()))` uses a coarser log-signature grid.
* Georax `SRKMK(GeneralShARK())` on the fine grid is used as a reference.

Run with:

    uv run python docs/examples/worm_sphere_sde.py
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import diffrax
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from georax import RKMK, SO, SRKMK, GeometricEuler, GeometricTerm

from roughrax import LogODE, RoughTerm, SignatureInterpolation

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt

# jax.config.update("jax_enable_x64", True)

VIEW_ELEV = 24.0
VIEW_AZIM = -58.0
VISIBLE_DOT_MIN = 0.08
TARGET_CAP_RADIUS = 0.68
SOFT_CAP_RADIUS = 0.84
TARGET_VIDEO_SECONDS = 15.0
END_PAUSE_SECONDS = 5.0
WARMUP_BEFORE_TIMING = True
DEFAULT_SEED = 7
DEFAULT_T1 = 5.0
DEFAULT_FINE_STEPS = 1024
DEFAULT_COARSE_STEPS = 64
DEFAULT_DEPTH = 2
DEFAULT_DIFFUSION_SCALE = 0.48
DEFAULT_FPS = 20
DEFAULT_DPI = 120
DEFAULT_FORMATS = "mp4,gif"
# DEFAULT_FORMATS = "mp4"


@dataclass(frozen=True)
class SphereSolve:
    name: str
    ts: np.ndarray
    rotations: np.ndarray
    points: np.ndarray


@dataclass(frozen=True)
class TimedSolve:
    solve: SphereSolve
    elapsed_seconds: float
    finish_seconds: float


class PiecewiseLinearLevyPath(diffrax.AbstractPath):
    interpolation: diffrax.LinearInterpolation

    @property
    def t0(self):
        return self.interpolation.t0

    @property
    def t1(self):
        return self.interpolation.t1

    def evaluate(self, t0, t1=None, left=True, use_levy=False):
        if t1 is None:
            return self.interpolation.evaluate(t0, left=left)
        increment = self.interpolation.evaluate(t0, t1, left=left)
        if use_levy:
            return diffrax.SpaceTimeLevyArea(
                dt=t1 - t0,
                W=increment,
                H=jnp.zeros_like(increment),
            )
        return increment


def sample_brownian_path(
    *, seed: int, t1: float, steps: int, dim: int
) -> tuple[jax.Array, jax.Array]:
    ts = jnp.linspace(0.0, t1, steps + 1)
    dt = ts[1:] - ts[:-1]
    key = jax.random.PRNGKey(seed)
    increments = jax.random.normal(key, (steps, dim)) * jnp.sqrt(dt)[:, None]
    xs = jnp.concatenate(
        [jnp.zeros((1, dim), dtype=ts.dtype), jnp.cumsum(increments, axis=0)],
        axis=0,
    )
    return ts, xs


def sphere_points(rotations: np.ndarray) -> np.ndarray:
    north = np.array([0.0, 0.0, 1.0])
    return np.einsum("tij,j->ti", rotations, north)


def camera_direction(*, elev: float = VIEW_ELEV, azim: float = VIEW_AZIM) -> np.ndarray:
    elev_rad = np.deg2rad(elev)
    azim_rad = np.deg2rad(azim)
    return np.array(
        [
            np.cos(elev_rad) * np.cos(azim_rad),
            np.cos(elev_rad) * np.sin(azim_rad),
            np.sin(elev_rad),
        ]
    )


def camera_frame(dtype) -> tuple[jax.Array, jax.Array, jax.Array]:
    center = camera_direction()
    up = np.array([0.0, 0.0, 1.0])
    first = np.cross(up, center)
    first /= np.linalg.norm(first)
    second = np.cross(center, first)
    return (
        jnp.asarray(center, dtype=dtype),
        jnp.asarray(first, dtype=dtype),
        jnp.asarray(second, dtype=dtype),
    )


def rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    sine = np.linalg.norm(cross)
    cosine = float(np.dot(source, target))
    if sine < 1e-12:
        return np.eye(3) if cosine > 0.0 else np.diag([1.0, -1.0, -1.0])
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine**2)


def visible_sector_y0(dtype) -> jax.Array:
    north = np.array([0.0, 0.0, 1.0])
    rotation = rotation_between(north, camera_direction())
    return jnp.asarray(rotation, dtype=dtype)


def _ambient_from_cap_coords(q: jax.Array, e1: jax.Array, e2: jax.Array) -> jax.Array:
    return q[0] * e1 + q[1] * e2


def _tangent_to_so3_coords(y: jax.Array, tangent: jax.Array) -> jax.Array:
    body = y.T @ tangent
    return jnp.asarray([0.0, body[0], body[1]], dtype=y.dtype)


def cap_vector_fields(
    y: jax.Array, *, diffusion_scale: float
) -> tuple[jax.Array, jax.Array]:
    _, e1, e2 = camera_frame(y.dtype)
    point = y[:, 2]
    cap_coords = jnp.asarray([jnp.dot(point, e1), jnp.dot(point, e2)])
    radius_sq = jnp.dot(cap_coords, cap_coords)
    tangent_coords = jnp.asarray([-cap_coords[1], cap_coords[0]], dtype=y.dtype)
    seed_coords = jnp.exp(-radius_sq / 0.04) * jnp.asarray(
        [0.42, 0.10],
        dtype=y.dtype,
    )

    cap_radius_sq = 1.0 - VISIBLE_DOT_MIN**2
    soft_radius_sq = SOFT_CAP_RADIUS**2
    edge_fraction = jnp.clip(
        (cap_radius_sq - radius_sq) / (cap_radius_sq - soft_radius_sq),
        0.0,
        1.0,
    )
    boundary_pressure = jnp.maximum(radius_sq - soft_radius_sq, 0.0)
    boundary_pressure = boundary_pressure / (cap_radius_sq - soft_radius_sq)

    orbit = 1.15 * tangent_coords
    radial = 0.85 * (TARGET_CAP_RADIUS**2 - radius_sq) * cap_coords
    boundary = -3.5 * boundary_pressure**2 * cap_coords
    drift_coords = orbit + radial + boundary + seed_coords

    noise_scale = diffusion_scale * edge_fraction
    drift = _ambient_from_cap_coords(drift_coords, e1, e2)
    first_field = noise_scale * e1
    second_field = noise_scale * e2
    spin_field = jnp.asarray([0.18 * diffusion_scale, 0.0, 0.0], dtype=y.dtype)

    def project_tangent(vector):
        return vector - jnp.dot(vector, point) * point

    drift_frame = _tangent_to_so3_coords(y, project_tangent(drift))
    diffusion_rows = jnp.stack(
        [
            _tangent_to_so3_coords(y, project_tangent(first_field)),
            _tangent_to_so3_coords(y, project_tangent(second_field)),
            spin_field,
        ]
    )
    return drift_frame, diffusion_rows


def make_geometric_euler_solve(
    fine_ts: jax.Array,
    brownian: jax.Array,
    *,
    y0: jax.Array,
    diffusion_scale: float,
) -> Callable[[], SphereSolve]:
    geometry = SO(3)

    def drift_fn(t, y, args):
        del t, args
        drift, _ = cap_vector_fields(y, diffusion_scale=diffusion_scale)
        return drift

    def diffusion_fn(t, y, args):
        del t, args
        _, diffusion_rows = cap_vector_fields(y, diffusion_scale=diffusion_scale)
        return diffusion_rows.T

    terms = diffrax.MultiTerm(
        GeometricTerm(drift_fn, geometry),
        diffrax.ControlTerm(
            diffusion_fn,
            diffrax.LinearInterpolation(ts=fine_ts, ys=brownian),
        ),
    )
    solver = GeometricEuler()
    stepsize_controller = diffrax.StepTo(fine_ts)
    saveat = diffrax.SaveAt(ts=fine_ts)

    def solve() -> SphereSolve:
        sol = diffrax.diffeqsolve(
            terms,
            solver,
            t0=float(fine_ts[0]),
            t1=float(fine_ts[-1]),
            dt0=None,
            y0=y0,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            max_steps=fine_ts.shape[0] + 4,
            throw=True,
        )
        rotations = np.asarray(jax.block_until_ready(sol.ys))
        return SphereSolve(
            "Georax GeometricEuler",
            np.asarray(fine_ts),
            rotations,
            sphere_points(rotations),
        )

    return solve


def make_srkmk_reference_solve(
    fine_ts: jax.Array,
    brownian: jax.Array,
    *,
    y0: jax.Array,
    diffusion_scale: float,
) -> Callable[[], SphereSolve]:
    geometry = SO(3)

    def drift_fn(t, y, args):
        del t, args
        drift, _ = cap_vector_fields(y, diffusion_scale=diffusion_scale)
        return drift

    def diffusion_fn(t, y, args):
        del t, args
        _, diffusion_rows = cap_vector_fields(y, diffusion_scale=diffusion_scale)
        return diffusion_rows.T

    control = PiecewiseLinearLevyPath(
        diffrax.LinearInterpolation(ts=fine_ts, ys=brownian)
    )
    terms = diffrax.MultiTerm(
        GeometricTerm(drift_fn, geometry),
        diffrax.ControlTerm(diffusion_fn, control),
    )
    solver = SRKMK(diffrax.GeneralShARK())
    stepsize_controller = diffrax.StepTo(fine_ts)
    saveat = diffrax.SaveAt(ts=fine_ts)

    def solve() -> SphereSolve:
        sol = diffrax.diffeqsolve(
            terms,
            solver,
            t0=float(fine_ts[0]),
            t1=float(fine_ts[-1]),
            dt0=None,
            y0=y0,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            max_steps=fine_ts.shape[0] + 4,
            throw=True,
        )
        rotations = np.asarray(jax.block_until_ready(sol.ys))
        return SphereSolve(
            "SRKMK(GeneralShARK)",
            np.asarray(fine_ts),
            rotations,
            sphere_points(rotations),
        )

    return solve


def make_log_ode_solve(
    fine_ts: jax.Array,
    brownian: jax.Array,
    coarse_ts: jax.Array,
    *,
    y0: jax.Array,
    diffusion_scale: float,
    depth: int,
) -> Callable[[], SphereSolve]:
    driver_ys = jnp.concatenate([fine_ts[:, None], brownian], axis=1)
    driver = diffrax.LinearInterpolation(ts=fine_ts, ys=driver_ys)
    control = SignatureInterpolation(
        driver,
        coarse_ts,
        depth=depth,
        solution="stratonovich",
    )

    def vector_field(y):
        drift, diffusion_rows = cap_vector_fields(y, diffusion_scale=diffusion_scale)
        return jnp.concatenate([drift[None, :], diffusion_rows], axis=0)

    term = RoughTerm(vector_field, control, SO(3))
    solver = LogODE(RKMK(diffrax.Heun()))
    stepsize_controller = diffrax.StepTo(coarse_ts)
    saveat = diffrax.SaveAt(ts=coarse_ts)

    def solve() -> SphereSolve:
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=float(coarse_ts[0]),
            t1=float(coarse_ts[-1]),
            dt0=None,
            y0=y0,
            stepsize_controller=stepsize_controller,
            saveat=saveat,
            max_steps=coarse_ts.shape[0] + 4,
            throw=True,
        )
        rotations = np.asarray(jax.block_until_ready(sol.ys))
        return SphereSolve(
            "Roughrax LogODE + RKMK(Heun)",
            np.asarray(coarse_ts),
            rotations,
            sphere_points(rotations),
        )

    return solve


def time_solve(solve_fn) -> tuple[SphereSolve, float]:
    if WARMUP_BEFORE_TIMING:
        solve_fn()
    start = time.perf_counter()
    solve = solve_fn()
    elapsed = time.perf_counter() - start
    return solve, elapsed


def normalise_finish_times(
    euler: SphereSolve,
    euler_elapsed: float,
    log_ode: SphereSolve,
    log_ode_elapsed: float,
) -> tuple[TimedSolve, TimedSolve]:
    euler_elapsed = max(euler_elapsed, 1e-12)
    log_ode_elapsed = max(log_ode_elapsed, 1e-12)
    slowest = max(euler_elapsed, log_ode_elapsed, 1e-12)
    return (
        TimedSolve(
            euler,
            euler_elapsed,
            TARGET_VIDEO_SECONDS * euler_elapsed / slowest,
        ),
        TimedSolve(
            log_ode,
            log_ode_elapsed,
            TARGET_VIDEO_SECONDS * log_ode_elapsed / slowest,
        ),
    )


def validate_rotation_solve(solve: SphereSolve) -> None:
    eye = np.eye(3)
    orthogonality_error = np.linalg.norm(
        np.swapaxes(solve.rotations, -1, -2) @ solve.rotations - eye,
        axis=(-2, -1),
    ).max()
    min_det = np.linalg.det(solve.rotations).min()
    if not np.isfinite(orthogonality_error) or not np.isfinite(min_det):
        raise FloatingPointError(f"{solve.name} produced non-finite rotations.")
    if orthogonality_error > 2e-5 or min_det < 0.999:
        raise RuntimeError(
            f"{solve.name} left SO(3): "
            f"max ||R^T R - I||={orthogonality_error:.3e}, "
            f"min det(R)={min_det:.6f}."
        )


def validate_visible_sector(solves: list[SphereSolve]) -> None:
    direction = camera_direction()
    dots = [solve.points @ direction for solve in solves]
    min_dot = min(float(dot.min()) for dot in dots)
    if min_dot < VISIBLE_DOT_MIN:
        raise RuntimeError(
            "Path left the visible sphere sector. "
            f"Minimum camera-facing dot product was {min_dot:.3f}; "
            "try reducing --t1 or --diffusion-scale."
        )


def configure_axis(ax, *, title: str) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 48)
    v = np.linspace(0.0, np.pi, 24)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="#dfe7e2", alpha=0.25, linewidth=0, shade=False)
    ax.plot_wireframe(
        x,
        y,
        z,
        rstride=4,
        cstride=4,
        color="#8ea39a",
        alpha=0.22,
        linewidth=0.55,
    )
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    ax.set_axis_off()


def piecewise_linear_path_at(solve: SphereSolve, t: float) -> np.ndarray:
    if t <= solve.ts[0] or len(solve.ts) == 1:
        return solve.points[:1]
    if t >= solve.ts[-1]:
        return solve.points

    index = int(np.searchsorted(solve.ts, t, side="right") - 1)
    index = int(np.clip(index, 0, len(solve.ts) - 2))
    fraction = (t - solve.ts[index]) / (solve.ts[index + 1] - solve.ts[index])
    current = (1.0 - fraction) * solve.points[index] + fraction * solve.points[
        index + 1
    ]
    return np.concatenate([solve.points[: index + 1], current[None, :]], axis=0)


def final_point_error(solve: SphereSolve, reference: SphereSolve) -> float:
    return float(np.linalg.norm(solve.points[-1] - reference.points[-1]))


def make_animation(
    euler: TimedSolve,
    log_ode: TimedSolve,
    reference: SphereSolve,
    *,
    video_seconds: float,
    end_pause_seconds: float,
    fps: int,
):
    total_seconds = video_seconds + end_pause_seconds
    if total_seconds <= 0.0:
        raise ValueError("Total animation duration must be positive.")
    frames = max(2, int(round(total_seconds * fps)))
    frame_seconds = np.linspace(0.0, total_seconds, frames)
    fig = plt.figure(figsize=(10.5, 5.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.08))
    axes = [fig.add_subplot(grid[0, i], projection="3d") for i in range(2)]
    label_axes = [fig.add_subplot(grid[1, i]) for i in range(2)]
    solves = [euler, log_ode]
    colors = ["#2667ff", "#d64045"]
    artists = []

    for ax, label_ax, timed, color in zip(
        axes, label_axes, solves, colors, strict=True
    ):
        solve = timed.solve
        title = (
            f"{solve.name}\n"
            f"{len(solve.ts) - 1} steps, timed {timed.elapsed_seconds:.3f}s, "
            f"finishes at {timed.finish_seconds:.1f}s"
        )
        configure_axis(ax, title=title)
        (path_line,) = ax.plot([], [], [], color=color, alpha=0.9, linewidth=1.6)
        label_ax.text(
            0.5,
            0.5,
            f"final point error vs {reference.name}\n"
            f"{final_point_error(solve, reference):.3e}",
            ha="center",
            va="center",
            fontsize=9,
            color="#26332f",
        )
        label_ax.set_axis_off()
        artists.append(path_line)

    fig.suptitle("One Brownian path on S^2, solved two ways", fontsize=13)

    def update(frame: int):
        display_second = frame_seconds[frame]
        changed = []
        for timed, path_line in zip(solves, artists, strict=True):
            solve = timed.solve
            progress = min(display_second / timed.finish_seconds, 1.0)
            t = solve.ts[0] + progress * (solve.ts[-1] - solve.ts[0])
            path = piecewise_linear_path_at(solve, t)
            path_line.set_data_3d(path[:, 0], path[:, 1], path[:, 2])
            changed.append(path_line)
        return changed

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000 / fps,
        blit=False,
    )
    return fig, anim


def save_animation(anim, output: Path, *, fps: int, dpi: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".mp4":
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError("Matplotlib ffmpeg writer is not available.")
        writer = animation.FFMpegWriter(fps=fps, bitrate=2200)
    elif suffix == ".gif":
        if not animation.writers.is_available("pillow"):
            raise RuntimeError("Matplotlib pillow writer is not available.")
        writer = animation.PillowWriter(fps=fps)
    else:
        raise ValueError(f"Unsupported animation format: {output.suffix!r}.")
    anim.save(output, writer=writer, dpi=dpi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--t1", type=float, default=DEFAULT_T1)
    parser.add_argument("--fine-steps", type=int, default=DEFAULT_FINE_STEPS)
    parser.add_argument("--coarse-steps", type=int, default=DEFAULT_COARSE_STEPS)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument(
        "--diffusion-scale", type=float, default=DEFAULT_DIFFUSION_SCALE
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument(
        "--formats",
        default=DEFAULT_FORMATS,
        help="Comma-separated animation formats to save. Supported: mp4,gif.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=ROOT / "docs/examples/outputs/worm_sphere_sde_side_by_side",
        help="Output path without extension.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fine_steps % args.coarse_steps != 0:
        raise ValueError("--fine-steps must be divisible by --coarse-steps.")
    if args.fps < 1:
        raise ValueError("--fps must be at least 1.")

    fine_ts, brownian = sample_brownian_path(
        seed=args.seed,
        t1=args.t1,
        steps=args.fine_steps,
        dim=3,
    )
    stride = args.fine_steps // args.coarse_steps
    coarse_ts = fine_ts[::stride]
    y0 = visible_sector_y0(brownian.dtype)

    solve_euler = make_geometric_euler_solve(
        fine_ts,
        brownian,
        y0=y0,
        diffusion_scale=args.diffusion_scale,
    )
    solve_reference = make_srkmk_reference_solve(
        fine_ts,
        brownian,
        y0=y0,
        diffusion_scale=args.diffusion_scale,
    )
    solve_log_ode = make_log_ode_solve(
        fine_ts,
        brownian,
        coarse_ts,
        y0=y0,
        diffusion_scale=args.diffusion_scale,
        depth=args.depth,
    )
    euler, euler_elapsed = time_solve(solve_euler)
    log_ode, log_ode_elapsed = time_solve(solve_log_ode)
    reference = solve_reference()
    validate_rotation_solve(euler)
    validate_rotation_solve(log_ode)
    validate_rotation_solve(reference)
    validate_visible_sector([euler, log_ode, reference])
    timed_euler, timed_log_ode = normalise_finish_times(
        euler,
        euler_elapsed,
        log_ode,
        log_ode_elapsed,
    )
    print(
        "timed solves: "
        f"GeometricEuler {timed_euler.elapsed_seconds:.3f}s "
        f"(finishes at {timed_euler.finish_seconds:.1f}s), "
        f"LogODE {timed_log_ode.elapsed_seconds:.3f}s "
        f"(finishes at {timed_log_ode.finish_seconds:.1f}s)"
    )

    fig, anim = make_animation(
        timed_euler,
        timed_log_ode,
        reference,
        video_seconds=TARGET_VIDEO_SECONDS,
        end_pause_seconds=END_PAUSE_SECONDS,
        fps=args.fps,
    )
    for format_name in [item.strip() for item in args.formats.split(",") if item]:
        output = args.output_stem.with_suffix(f".{format_name}")
        print(f"saving {output}")
        save_animation(anim, output, fps=args.fps, dpi=args.dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
