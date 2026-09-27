# Changelog

## Unreleased

Changes since `0.2.0`.

### Added

- `Davie`, an explicit rough Taylor solver using full signatures, for Euclidean spaces and manifolds. The control depth determines the truncation degree; no inner ODE solver is required.
- Full geometric tensor signatures and branched Itô signatures, using BCK trees in Euclidean space and MKW ordered forests on manifolds.
- `SignatureInterpolation.from_signatures(...)` for precomputed full signatures, including branched coefficients matched to the target geometry.
- Controlled geometric coefficients in Davie through the same autonomous augmentation used by LogODE, with support for JIT, batching, and differentiation.
- Reverse full-signature increments using the signature group inverse, and geometry-preserving dense output that interpolates linearly in local coordinates.

### Breaking changes

- The previous log-signature `SignatureInterpolation` is now named `LogSignatureInterpolation`. Update imports and constructors for `LogODE`, `LinearMagnus`, and `LinearFer`, including calls to `from_logsignatures(...)`.
- `SignatureInterpolation` now represents full signatures for `Davie`. Its increments require adjacent signature knots; use `diffrax.StepTo(control.ts)`. It does not support fractional or multi-knot increments.
- Solvers reject incompatible control types, and pre-lifted vector fields require log-signature controls. Controlled coefficients continue to require geometric controls.

### Changed

- Updated controlled manifold augmentation to the Georax chart API, with explicit chart selection and coordinate reshaping.
- Simplified Lyndon matrix-basis construction and batched the degree contractions used by `LinearFer`.
- Updated the sphere example to compare GeometricEuler, LogODE, and Davie on one Brownian path, using smooth vector fields, warmed solve timings, reference errors, and SO(3) interpolation for playback.
- Updated the other examples and getting-started documentation to use `LogSignatureInterpolation`.

### Fixed

- Corrected the documented Brownian Itô correction to `-0.5 * dt * Sigma`, matching the correction added by PySigLib to local chain-tree log coefficients.
- Updated branched log-signature preparation to request PySigLib method 0 explicitly.

### Dependencies

- Raised the PySigLib requirement to `>=4.0.0rc2`; checkout installs pin revision `6d663452c7958cd10bdc69baded49f464d177b92` for the upstream MKW factorial fix.
- Matched PySigLib's build-time `jaxlib` dependency to the runtime version and refreshed the lockfile for the updated upstream dependencies.

### Solver limits

Davie supplies no adaptive error estimate. Its dense output preserves geometry but is not a higher-order continuous extension.
