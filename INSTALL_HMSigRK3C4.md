# HM-SigRK degree-four completion overlay

This overlay adds two Euclidean, Stratonovich rough solvers:

- `HMSigRK3`: the original quadratic-cloud degree-three method.
- `HMSigRK3C4`: the replacement that matches the complete degree-four B-series of
  the canonical truncated-log-signature flow.

The files target the current `main` API of `luke-a-thompson/roughrax` at commit
`36382fb237d1d553cb7d3a88d27b8a3afcf07b89`.

## Apply

From the root of a clean RoughRAX checkout:

```bash
patch -p1 < /path/to/roughrax-hm-c4.patch
uv sync --extra dev
uv run pytest -q tests/test_hm_sigrk.py
```

The ZIP is also an overlay: copy its `roughrax`, `tests`, and `docs` directories
into the repository root. If the earlier `HMSigRK3` prototype is already present,
replace its `roughrax/_solver/hm_sigrk.py` and update the two `__init__.py` files
with the versions in this overlay.

## Use

```python
import diffrax
import jax.numpy as jnp
from georax import Euclidean
from roughrax import HMSigRK3C4, RoughTerm, SignatureInterpolation


def vector_field(y):
    return jnp.stack([jnp.cos(y), jnp.sin(y)])


fine_ts = jnp.linspace(0.0, 1.0, 257)
fine_xs = jnp.stack(
    [jnp.sin(3.0 * fine_ts), jnp.cos(2.0 * fine_ts)],
    axis=-1,
)
coarse_ts = fine_ts[::32]
control = SignatureInterpolation(
    diffrax.LinearInterpolation(ts=fine_ts, ys=fine_xs),
    coarse_ts,
    depth=3,
    solution="stratonovich",
)
term = RoughTerm(vector_field, control, Euclidean())

solution = diffrax.diffeqsolve(
    term,
    HMSigRK3C4(),
    t0=coarse_ts[0],
    t1=coarse_ts[-1],
    dt0=None,
    y0=jnp.asarray(1.0),
    stepsize_controller=diffrax.StepTo(coarse_ts),
    saveat=diffrax.SaveAt(ts=coarse_ts),
    max_steps=coarse_ts.shape[0] + 4,
)
```

The default is:

```python
HMSigRK3C4(substeps=2, rho_scale=1.0)
```

`substeps=4` is the conservative accuracy setting. In the independent validation
included with this overlay, its endpoint RMSE stayed between `0.95x` and `1.17x`
the Log-ODE RMSE over 8, 16, 32, and 64 rough steps. The two-substep default used
half as many evaluation points and stayed between `0.95x` and `1.45x`.

For a two-dimensional driver:

- one completion substep has 48 vector-field evaluation points;
- those points are arranged in three parallel dependency layers;
- the default therefore uses 96 points in six sequential batched layers;
- `substeps=4` uses 192 points in twelve sequential batched layers.

## Benchmark

```bash
uv run python docs/examples/hm_c4_vs_log_ode.py \
    --num-paths 8 \
    --fine-exponent 12 \
    --coarse-exponents 4 5 6 7 8 \
    --hm-c4-substeps 2 \
    --include-c4-four-substeps \
    --logode-inner tsit5 \
    --repeats 5
```

This writes:

```text
docs/examples/outputs/hm_c4_vs_log_ode.csv
docs/examples/outputs/hm_c4_vs_log_ode.png
```

The supplied independent development validation is in:

```text
docs/examples/outputs/hm_sigrk3_c4_independent_validation.csv
docs/examples/outputs/hm_sigrk3_c4_independent_validation.png
```

## Construction

Let the depth-three log-signature be `L = L1 + L2 + L3`. Each internal substep
forms the tensor-signature completion

```text
exp_tensor(L / substeps)
```

through level four. A fixed three-layer tableau generates every labelled stage
forest with at most three vertices. Its per-step update weights are then obtained
from a cached linear moment solve, so every rooted-tree B-series coefficient
through degree four equals the coefficient of the completed signature exactly.
The residual of the checked-in two-dimensional moment system is approximately
`1.1e-14` in float64.

Repeating the same log increment is algebraically coherent because

```text
exp(L / m)^m = exp(L).
```

The repeated degree-four tableau suppresses its first unmatched degree-five
error and converges toward the same truncated Log-ODE flow, without constructing
Lie-bracket vector fields or using automatic differentiation, opposite probes,
or derivative quotients.

## Scope

`HMSigRK3C4` currently supports:

- Euclidean state spaces;
- geometric/Stratonovich controls;
- depth-three or deeper method-1 Lyndon log-signatures;
- array-valued states;
- leading- or trailing-axis stacks of level-one vector fields;
- fixed rough steps that do not cross signature knots.

It does not currently support branched Itô data, manifold retractions, pre-lifted
vector fields, or an embedded adaptive error estimate.
