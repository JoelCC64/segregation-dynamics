# Segregation Dynamics — One Model, Two Computational Paradigms

**A continuous-space Schelling-type segregation model, implemented as (1) a
high-performance Fortran 95 forward simulation and (2) a differentiable
PyTorch twin for gradient-based calibration and sensitivity analysis.**

![Fortran](https://img.shields.io/badge/Fortran-95-734f96)
![PyTorch](https://img.shields.io/badge/PyTorch-differentiable-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JoelCC64/segregation-dynamics/blob/main/notebooks/differentiable_segregation.ipynb)

---

## The problem

In 1971, Thomas Schelling showed a result that became a cornerstone of
complex-systems science: even *mild* individual preferences about one's
neighbours can produce *strong* collective segregation. No agent wants a
segregated city, yet segregation emerges. It is a canonical example of
**emergence** — macroscopic order arising from simple local rules — and it
connects directly to the statistical physics of interacting systems:
Schelling-type models have been mapped onto spin dynamics and studied with
the tools of phase transitions and nucleation (Vinković & Kirman 2006;
Gauvin, Vannimenus & Nadal 2009).

This repository studies an **off-lattice (continuous-space) variant** of the
model: agents are points in a 2D domain rather than cells on a grid, which
makes the neighbourhood structure geometric (a Euclidean disk) instead of
combinatorial (nearest lattice sites).

## The model

- Two populations of point agents, `N_A = N_B = 1000` (types 1 and 0), in the
  square `[0, L) × [0, L)` with `L = 100` and **open boundaries** (the
  neighbourhood disk is clipped at the walls — no periodicity).
- **Happiness rule:** an agent is *happy* if it has at least `K_MIN = 30`
  neighbours of its **own type** within Euclidean distance `R = 10`.
  Note this is an *absolute-count* threshold, a variant of the classic
  *fractional* tolerance rule; it couples segregation to local density
  (agents seek same-type *crowds*, not just same-type *majorities*).
- **Dynamics (random relocation kinetics):** at each event, one uniformly
  random unhappy agent teleports to uniformly random locations until it finds
  one where it is happy (up to 40 trials; if none works, it stays at its last
  trial location). This is Schelling's original "move wherever you like"
  kinetics, as opposed to nearest-vacancy or swap kinetics.
- **Stopping criterion:** the simulation ends when every agent is happy, or
  after a fixed budget of relocation events.

**Observables**

| Metric | Definition | Value at t=0 | Value at convergence |
|---|---|---|---|
| Fraction happy | share of agents meeting the happiness rule | 0.48 | 1.00 |
| Segregation index | mean fraction of same-type agents among each agent's neighbours (0.5 = perfectly mixed, → 1 = fully segregated) | 0.498 | 0.629 |

The reference run (seed `20260707`) converges after **1,099 relocation
events** (~0.4 s on a laptop CPU): starting from a statistically perfect
mixture (index 0.498 ≈ 0.5), the population self-organises into measurably
segregated clusters — *without any agent preferring segregation*. That gap
between individual rules and collective outcome is the whole point of the
model.

## Why two paradigms?

This project is deliberately **not** a "legacy Fortran vs. modern Python"
exercise. Fortran remains a first-class tool for scientific simulation, and
each implementation serves a different scientific purpose:

| | **Fortran 95** (`/fortran`) | **PyTorch** (`/pytorch`) |
|---|---|---|
| Role | Reference forward simulation | Differentiable twin of the model |
| Optimised for | Raw speed of the forward dynamics (cell-linked lists, O(1) bookkeeping, zero interpreter overhead) | Gradients *through* the dynamics |
| What it enables | Large ensembles, parameter sweeps, long runs | Autodiff, gradient-based **parameter calibration**, **sensitivity analysis** ∂(outcome)/∂(parameter), integration with ML pipelines |

The forward model answers *"what does this rule set produce?"*; the
differentiable model answers *"which parameters produce this observed
outcome, and how sensitive is the outcome to them?"* — the inverse problem.
Agent-based models are classically hard to calibrate precisely because they
are non-differentiable; re-expressing the model with smooth relaxations in
PyTorch is what unlocks that second family of questions.

## The differentiable twin (Phase 2)

The model's discreteness — hard neighbourhood disks, a Heaviside happiness
threshold, random accept/reject teleports — kills gradients. The PyTorch
implementation ([design document](pytorch/README.md)) replaces each discrete
element with a smooth relaxation:

- **soft neighbourhood**: $\mathbb{1}[d \le R] \to \sigma((R-d)/\tau_d)$;
- **soft happiness gate**: $\mathbb{1}[n \ge K_{\min}] \to \sigma((n-K_{\min})/\tau_h)$;
- **unhappiness-gated gradient flow** instead of random teleports: each agent
  ascends the gradient of its own soft same-type count, gated by $(1-h_i)$.

The rules converge to the discrete ones as $\tau \to 0$ (verified); the
kinetics is deliberately different (local drift vs global teleports) — the
twin is *validated against* the Fortran reference, not a bit-exact replica:
both reach full happiness at reference density, with final hard segregation
0.59 (relaxed drift) vs 0.63 (teleport).

What differentiability delivers (all demonstrated in the
[notebook](notebooks/differentiable_segregation.ipynb), verified end-to-end
on CPU, GPU-ready via Colab):

- **Sensitivity analysis through the dynamics** — one backward pass gives
  $\partial S_T/\partial K_{\min} > 0$ (more demanding agents ⇒ *more*
  segregation: Schelling's insight as a computed number) and
  $\partial S_T/\partial R$, cross-validated against a finite-difference
  convergence table (agreement to $3\times10^{-6}$ at $\delta = 10^{-5}$,
  float64).
- **Gradient-based calibration (inverse problem)** — a hidden tolerance
  threshold $K^\ast = 35$ is recovered to **34.99** from an observed
  segregation trajectory by Adam, backpropagating through the entire
  120-step unrolled simulation with $O(N^2)$ memory (gradient
  checkpointing, verified exact).

## Repository structure

```
.
├── fortran/                  # Reference high-performance implementation
│   ├── segregation.f90       # Documented Fortran 95 code
│   ├── Makefile
│   └── README.md             # Build/run instructions + implementation notes
├── pytorch/
│   ├── segregation_torch.py  # Differentiable twin (soft relaxations, checkpointing)
│   └── README.md             # Design doc: how the discreteness is handled, honestly
├── notebooks/
│   └── differentiable_segregation.ipynb   # Colab-ready demos: sensitivity + calibration
├── LICENSE                   # MIT
└── README.md
```

## Quick start

Requires `gfortran` (any recent version; the code is strict standard
Fortran 95).

```bash
cd fortran
make          # builds ./segregation  (gfortran -O2 -std=f95)
./segregation
```

Output:

```
Converged: all agents happy after 1099 relocation events.
fraction happy                    =  1.000
mean same-type neighbour fraction =  0.629
trial moves = 2304  (0 events found no happy spot)
```

Three data files are produced: `metrics.dat` (time series of the
observables), `snapshots.dat` (periodic full configurations, gnuplot
`index`-friendly), and `final_configuration.dat`. Runs are **reproducible**:
the RNG seed is a compile-time parameter.

**PyTorch twin** — open
[the notebook](notebooks/differentiable_segregation.ipynb) in Colab (badge
above; GPU runtime optional — everything also runs on CPU in ~2 minutes) or
locally:

```bash
pip install torch matplotlib
jupyter notebook notebooks/differentiable_segregation.ipynb
```

The notebook begins with a ~1 s smoke test (shapes, τ→0 discrete limit,
autodiff-vs-finite-differences, checkpoint exactness) before any large run.

## Correctness

The Fortran implementation is verified by:
- clean compilation under `gfortran -std=f95 -pedantic -Wall -Wextra`;
- a full run under runtime bounds checking (`-fcheck=all`);
- an independent brute-force O(N²) recomputation (NumPy) of the final
  state: every agent satisfies the happiness rule, and the segregation
  index matches the Fortran value to 6 decimal places;
- bit-identical metrics across repeated runs (seeded RNG).

## Roadmap

- [x] **Phase 1 — Fortran reference implementation**: cell-linked neighbour
  search, exact incremental bookkeeping, reproducible metrics.
- [x] **Phase 2 — Differentiable PyTorch twin**: vectorised smooth
  relaxation, autodiff through the unrolled dynamics, sensitivity analysis
  and single-parameter calibration demos, gradient checkpointing.
- [ ] **Phase 3 — Inverse problems at scale**: joint calibration of
  (`K_MIN`, `R`) from noisy/partial observations; sensitivity maps along
  trajectories; Fortran ensembles vs. PyTorch gradients on the same axes.

## Background

I am a physics graduate (complex systems, plasma physics) working at the
intersection of computational physics and machine learning. This project sits
exactly on that boundary: a classic statistical-physics-flavoured
agent-based model, treated first with the traditional HPC toolchain and then
with modern differentiable programming — two paradigms, one model, two
different families of scientific questions.

## References

- T. C. Schelling, *Dynamic models of segregation*, J. Math. Sociol. **1**, 143–186 (1971).
- D. Vinković & A. Kirman, *A physical analogue of the Schelling model*, PNAS **103**, 19261–19265 (2006).
- L. Gauvin, J. Vannimenus & J.-P. Nadal, *Phase diagram of a Schelling segregation model*, Eur. Phys. J. B **70**, 293–304 (2009).
- A. Chopra et al., *Differentiable agent-based epidemiology* (GradABM), AAMAS (2023) — the differentiable-ABM trade-off adopted here.

## License

[MIT](LICENSE)
