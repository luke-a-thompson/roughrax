<p align="center">
  <picture>
    <source srcset="https://raw.githubusercontent.com/luke-a-thompson/roughrax/main/docs/_static/roughrax.dark.svg" media="(prefers-color-scheme: dark)">
    <source srcset="https://raw.githubusercontent.com/luke-a-thompson/roughrax/main/docs/_static/roughrax.light.svg" media="(prefers-color-scheme: light)">
    <img src="https://raw.githubusercontent.com/luke-a-thompson/roughrax/main/docs/_static/roughrax.light.svg" width="350" alt="Logo">
  </picture>
</p>

<h2 align='center'>Rough Differential Equation Integration with Diffrax and Georax.</h2>

Roughrax enables the solving of rough differential equations natively in [Diffrax](https://github.com/patrick-kidger/diffrax) via the log-ODE method. Leveraging [PySigLib](https://github.com/daniil-shmelev/pySigLib) for signatures, Roughrax supports Stratonovich and Itô integration over Euclidean spaces, with support for homogeneous spaces provided by [Georax](https://github.com/luke-a-thompson/georax).

## LogODE

`LogODE` solves a rough differential equation by lifting log-signatures of the driving path into a frozen vector field and integrating that field with a wrapped Diffrax solver. You pick accuracy/adaptivity by choosing the base solver — `LogODE` reuses its Runge-Kutta coefficients.

- Wrap a fixed-step ERK (for example `Heun()`) for fixed-step rough integration.
- Wrap an adaptive ERK (for example `Dopri5()`) to keep automatic stepsizing.
- Pass the base solver explicitly, for example `LogODE(diffrax.Tsit5())`.
- Wrap a geometric solver such as `georax.RKMK(diffrax.Tsit5())` when solving on a manifold.

## RoughTerm

`RoughTerm` holds the rough-path data. Pass it a fine driving path (any diffrax `LinearInterpolation`-like control), and it computes log-signatures over each interval of `interval_ts` internally via [pysiglib](https://github.com/luke-a-thompson/pysiglib).

| Argument | Purpose |
|----------|---------|
| `vector_field` | Function `y -> Array` returning the stacked vector fields. |
| `control` | Fine driving path with `.ts` / `.ys` (e.g. `diffrax.LinearInterpolation`). |
| `geometry` | `georax.Manifold` the solution lives on (defaults to `Euclidean`). |
| `depth` | Truncation depth of the log-signature. |
| `interval_ts` | Coarse grid the solver steps on. One log-signature per consecutive pair. Defaults to `control.ts`. |
| `solution` | `"stratonovich"` (log-signature, Lyndon basis) or `"ito"` (branched signature, rooted-tree basis). |

## Understanding Rough Path Integration
1. Sample a (rough) driving path on a fine grid: $X_t \in \mathbb{R}^d$
1. Compute log-signatures of $X$ over each coarse interval $[t_k, t_{k+1}]$
1. At each coarse step, freeze the lifted vector field at $y_k$ and integrate one unit of time on the manifold — the log-signature contracts against the lifted fields to produce the update

## Usage

```python
import diffrax
import jax.numpy as jnp
from georax import Euclidean
from roughrax import LogODE, RoughTerm, SignatureInterpolation

# Vector field returns the stacked driving fields f_1, ..., f_d at y.
def vector_field(y):
    return jnp.stack([jnp.cos(y), jnp.sin(y)])

# A fine driving path — here a deterministic 2D control on [0, 1].
fine_ts = jnp.linspace(0.0, 1.0, 257)
fine_xs = jnp.stack([jnp.sin(3 * fine_ts), jnp.cos(2 * fine_ts)], axis=-1)
driver = diffrax.LinearInterpolation(ts=fine_ts, ys=fine_xs)

# Coarse grid the solver steps on; one log-signature is computed per interval.
coarse_ts = fine_ts[::32]

control = SignatureInterpolation(
    driver,
    coarse_ts,
    depth=3,
    solution="stratonovich",
)
term = RoughTerm(
    vector_field,
    control,
    Euclidean(),
)

# Then solve with a Log-ODE step driving the wrapped Diffrax solver.
sol = diffrax.diffeqsolve(
    term,
    LogODE(diffrax.Tsit5()),
    t0=float(coarse_ts[0]),
    t1=float(coarse_ts[-1]),
    dt0=None,
    y0=jnp.asarray(1.0),
    stepsize_controller=diffrax.StepTo(coarse_ts),
    saveat=diffrax.SaveAt(ts=coarse_ts),
)
```

## Geometric usage

For manifold-valued equations, pass the target geometry to `RoughTerm` and wrap a geometric base solver with `LogODE`. The vector field should return the stacked driving fields in the coordinates expected by the manifold.

```python
import diffrax
import jax.numpy as jnp
from georax import CFEES25, SO
from roughrax import LogODE, RoughTerm, SignatureInterpolation


def so3_vector_field(y):
    del y
    return jnp.eye(3)


fine_ts = jnp.linspace(0.0, 1.0, 257)
fine_xs = jnp.stack(
    [
        0.2 * jnp.sin(3 * fine_ts),
        0.3 * jnp.cos(2 * fine_ts),
        0.1 * fine_ts,
    ],
    axis=-1,
)
driver = diffrax.LinearInterpolation(ts=fine_ts, ys=fine_xs)
coarse_ts = fine_ts[::32]

control = SignatureInterpolation(
    driver,
    coarse_ts,
    depth=3,
    solution="stratonovich",
)
term = RoughTerm(
    so3_vector_field,
    control,
    SO(3),
)

sol = diffrax.diffeqsolve(
    term,
    LogODE(CFEES25()),
    t0=float(coarse_ts[0]),
    t1=float(coarse_ts[-1]),
    dt0=None,
    y0=jnp.eye(3),
    stepsize_controller=diffrax.StepTo(coarse_ts),
    saveat=diffrax.SaveAt(ts=coarse_ts),
)
```

## Install

```bash
uv sync
```

## Convergence example

```bash
uv run python docs/examples/convergence.py
```

Solves a 2D rough ODE driven by Brownian motion at orders 1, 2, 3 against a fine Wong-Zakai reference and plots `h^(p/2)` convergence to `docs/examples/outputs/log_ode_convergence.png`.
