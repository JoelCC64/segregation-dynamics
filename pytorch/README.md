# Differentiable PyTorch implementation

The differentiable twin of the [Fortran reference model](../fortran). Same
agents, same geometry, same observables — reformulated so that **gradients
flow end-to-end through the simulated dynamics**, enabling the analyses the
forward paradigm cannot easily provide: gradient-based parameter calibration
and sensitivity analysis.

- Model code: [`segregation_torch.py`](segregation_torch.py) (single file, PyTorch only)
- Demonstrations: [`notebooks/differentiable_segregation.ipynb`](../notebooks/differentiable_segregation.ipynb)
  [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JoelCC64/segregation-dynamics/blob/main/notebooks/differentiable_segregation.ipynb)

## The problem: agent-based models are not differentiable

The reference model contains three discrete, gradient-killing elements:

| Where | Discrete object | Why gradients die |
|---|---|---|
| neighbourhood | $\mathbb{1}[d_{ij} \le R]$ | piecewise constant in positions and in $R$ |
| happiness | $\mathbb{1}[n_i \ge K_{\min}]$ | Heaviside step in count and threshold |
| kinetics | unhappy agents teleport at random until happy | discrete accept/reject of random proposals |

## The relaxation

Three substitutions, implemented in `segregation_torch.py`:

**1. Soft neighbourhood.** The hard disk becomes a sigmoid shell:

$$n_i \;=\; \sum_{j\,:\,t_j = t_i} \sigma\!\left(\frac{R - d_{ij}}{\tau_d}\right)$$

**2. Soft happiness gate.**

$$h_i \;=\; \sigma\!\left(\frac{n_i - K_{\min}}{\tau_h}\right) \in (0,1)$$

**3. Unhappiness-gated gradient flow** replaces the random teleports:

$$x_i \;\leftarrow\; \operatorname{clip}\!\Big( x_i + \eta \, (1 - h_i)\, \nabla_{x_i} n_i \Big)$$

Each agent ascends the gradient of *its own* soft same-type count, gated by
its unhappiness; happy agents stand still, exactly as in the discrete model.
The ascent direction has a closed form — attraction toward same-type agents,
dominated by the *marginal neighbours* near the disk edge where $\sigma'$
peaks. Computing the drift analytically (rather than with an inner autograd
call) keeps autodiff free for the *outer* derivatives w.r.t. $K_{\min}$, $R$,
$\tau$, avoiding double-backward.

Every operation is smooth, so any observable at any time is differentiable
w.r.t. any parameter by backprop through the unrolled dynamics. Gradients
w.r.t. $K_{\min}$ flow through the gate; w.r.t. $R$ through both the kernel
and the gate. As $\tau_d, \tau_h \to 0$ the counts and gates converge to the
discrete rules (verified numerically in the smoke test).

## Why this relaxation (and not the alternatives)

- **Gumbel-softmax / straight-through over candidate sites** would preserve
  the teleport kinetics, but a softmax-weighted *mixture of positions* is not
  a valid configuration of this model, and hard sampling with
  straight-through gives biased gradients — a shaky foundation for a
  calibration loop.
- **A density-field (PDE) formulation** is naturally differentiable but
  abandons the agent picture entirely, and with it the direct comparability
  with the Fortran implementation.
- **Gated gradient flow** keeps agents and the utility structure and gives
  *exact gradients of an approximate model* — rather than approximate
  gradients of the exact model. That trade-off is the standard one in
  differentiable agent-based modelling (cf. GradABM).

**What is honestly lost:** the kinetics. Drift is local transport; teleports
are global. Both reach full happiness at the reference density, but the
endpoints differ quantitatively — see the validation table. The right mental
model is a *differentiable twin validated against the reference*, not a
bit-exact replacement.

## Validation (measured, CPU)

| Check | Result |
|---|---|
| $\tau_d \to 0$ vs integer counts | exact (max err 0.0) |
| autodiff vs central finite differences (float64, small system) | agree to $5\times 10^{-10}$ rel. |
| autodiff vs FD through 100 steps at $N{=}2000$ (float64, $\delta{=}10^{-5}$) | agree to $3\times 10^{-6}$ rel. |
| checkpointed vs plain gradients | identical |
| hard fraction happy along relaxed dynamics | $0.52 \to 1.00$ (Fortran: $0.48 \to 1.00$) |
| hard segregation index along relaxed dynamics | $0.50 \to 0.59$ (Fortran teleport: $0.50 \to 0.63$) |
| calibration demo: recover hidden $K^\ast = 35$ from trajectory | $24.0 \to 34.99$, MSE $8\times10^{-10}$ |
| sensitivity sign | $\partial S/\partial K_{\min} > 0$: more demanding agents ⇒ more segregation |

A caveat worth knowing: at long horizons the map $K_{\min} \mapsto S_T$
develops fine-scale structure (clipping kinks, trajectory divergence), so
finite differences with a *large* step disagree with the exact local
gradient even though both are "right" at their own scale. The notebook
demonstrates this with a $\delta$-convergence table; trajectory-matching
losses over moderate horizons are the robust choice for calibration.

## Engineering notes

- **Device-agnostic:** `default_device()` picks CUDA when available; initial
  conditions are drawn on CPU so a seed gives the same configuration on CPU
  and GPU.
- **Memory:** all pairwise tensors are $O(N^2)$. Backprop through $T$ steps
  stores $O(T N^2)$ activations — prohibitive at $N=2000$, $T=100$. With
  `simulate(..., use_checkpoint=True)` activations are recomputed in the
  backward pass: memory drops to $O(N^2)$ total at ~2× forward compute
  (gradients verified identical).
- **No GPU required:** every demo in the notebook completes on CPU in
  ~2 minutes total; a Colab T4 makes it interactive.

## API sketch

```python
from segregation_torch import Config, init_state, simulate, hard_metrics

cfg = Config(n_steps=200)                      # Fortran reference scale
X0, types = init_state(cfg)                    # device-agnostic

k = torch.tensor(30.0, requires_grad=True)     # learnable threshold
out = simulate(X0, types, cfg, k_min=k, use_checkpoint=True)
out["seg"][-1].backward()                      # dS/dK_MIN in k.grad
```
