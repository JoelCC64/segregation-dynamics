# Analysis notebooks

## [`differentiable_segregation.ipynb`](differentiable_segregation.ipynb)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JoelCC64/segregation-dynamics/blob/main/notebooks/differentiable_segregation.ipynb)

Phase-2 notebook: the differentiable PyTorch twin of the Fortran reference
model, with the two demonstrations that motivate it:

1. **Sensitivity analysis** — exact ∂(segregation)/∂(K_MIN, R) *through* 100
   steps of dynamics in one backward pass, validated against a
   finite-difference convergence table;
2. **Gradient-based calibration** — recovery of a hidden tolerance threshold
   (K\* = 35, recovered 34.99) from an observed segregation trajectory by
   Adam through the fully unrolled simulation, with gradient checkpointing.

The notebook starts with a **smoke test** (~1 s) that verifies shapes, the
τ → 0 discrete limit, checkpoint exactness and autodiff-vs-finite-difference
agreement before any large run.

Runtimes (verified end-to-end on a laptop CPU): smoke test ~1 s · forward
N=2000, T=200 ~6 s · sensitivity ~40 s · calibration ~70 s. A Colab T4 GPU
(`Runtime → Change runtime type`) makes everything interactive; no code
changes needed — device selection is automatic.

## Planned (Phase 3)

- Phase-diagram sweeps over `K_MIN` and `R` (Fortran ensembles) vs
  gradient-based sensitivity maps (PyTorch) on the same axes.
- Joint calibration of (K_MIN, R) from noisy/partial observations.
- Visualisation of the Fortran `snapshots.dat` output (cluster formation).
