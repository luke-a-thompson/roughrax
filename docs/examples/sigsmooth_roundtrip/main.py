"""Check that a reconstructed loopy path has the requested signature."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pysiglib

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roughrax import nonstandard_wong_zakai


def main() -> None:
    degree = 3
    x = np.array(
        [
            [0.0, 0.0],
            [0.7, -0.2],
            [0.4, 0.8],
            [1.2, 0.5],
        ],
        dtype=np.float64,
    )

    sig_x = pysiglib.sig(x, degree)
    x_tilde = nonstandard_wong_zakai(sig_x, x.shape[-1], degree, basepoint=x[0])
    sig_x_tilde = pysiglib.sig(x_tilde, degree)

    error = float(np.max(np.abs(sig_x_tilde - sig_x)))
    np.testing.assert_allclose(sig_x_tilde, sig_x, rtol=1e-10, atol=1e-10)

    print(f"original path points: {len(x)}")
    print(f"loopy path points: {len(x_tilde)}")
    print(f"max signature error: {error:.3e}")


if __name__ == "__main__":
    main()
