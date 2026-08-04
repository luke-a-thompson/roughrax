# HMSigRK3 overlay for RoughRAX

This overlay targets RoughRAX `main` at commit
`dc07013c5b9aea1ffe441872ac378d6dc315d533`.

## Install into an existing checkout

From the root of your RoughRAX checkout, either apply the supplied patch:

```bash
patch -p1 < /path/to/roughrax-hm-sigrk3.patch
```

or copy the overlay directories into the checkout:

```bash
cp -R /path/to/roughrax-hm-sigrk3/roughrax .
cp -R /path/to/roughrax-hm-sigrk3/tests .
cp -R /path/to/roughrax-hm-sigrk3/docs .
```

No new dependency is required beyond the current RoughRAX dependencies. The
solver converts the existing method-1 Lyndon-word log-signature to tensor levels
one to three with a cached sparse integer map implemented in JAX; it does not
call a separate signature backend inside each step.

## Run the focused tests

```bash
uv sync --extra dev
uv run pytest -q tests/test_hm_sigrk.py
```

## Minimal usage

```python
import diffrax
import jax.numpy as jnp
from georax import Euclidean
from roughrax import HMSigRK3, RoughTerm, SignatureInterpolation


def vector_field(y):
    return jnp.stack([jnp.cos(y), jnp.sin(y)])


fine_ts = jnp.linspace(0.0, 1.0, 257)
fine_xs = jnp.stack(
    [jnp.sin(3 * fine_ts), jnp.cos(2 * fine_ts)],
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
term = RoughTerm(vector_field, control, Euclidean())
solution = diffrax.diffeqsolve(
    term,
    HMSigRK3(),
    t0=coarse_ts[0],
    t1=coarse_ts[-1],
    dt0=None,
    y0=jnp.asarray(1.0),
    stepsize_controller=diffrax.StepTo(coarse_ts),
    saveat=diffrax.SaveAt(ts=coarse_ts),
    max_steps=coarse_ts.shape[0] + 4,
)
```

## Compare with Log-ODE

```bash
uv run python docs/examples/hm_vs_log_ode.py \
  --num-paths 8 \
  --fine-exponent 12 \
  --coarse-exponents 4 5 6 7 8 \
  --logode-inner tsit5 \
  --repeats 5
```

The script writes:

- `docs/examples/outputs/hm_vs_log_ode.csv`
- `docs/examples/outputs/hm_vs_log_ode.png`

The reported steady-state timing excludes signature materialisation, which is
shared by both solvers. Signature preprocessing time is recorded separately.
The first compiled call is also reported separately from the median warm timing.

## Current scope

`HMSigRK3` is experimental and presently requires:

- a Euclidean state space;
- `solution="stratonovich"`;
- signature depth at least three;
- `diffrax.StepTo(control.ts)`, so every numerical step consumes one stored
  local log-signature.

It has no embedded error estimate and therefore is not an adaptive solver. The
method uses

```text
(d + 1)(d + 2) / 2 + d
```

vector-field evaluations per rough step: 8 for a two-dimensional driver and 13
for a three-dimensional driver.
