"""Differentiable continuous-space Schelling-type segregation model (PyTorch).

This is the differentiable twin of the Fortran reference implementation in
``/fortran``: same agents, same geometry, same observables — but every
discrete element of the model is replaced by a smooth relaxation so that
gradients flow end-to-end through the simulated dynamics.

The three discrete elements and their relaxations:

====================================  =========================================
Discrete model (Fortran)              Relaxed twin (this module)
====================================  =========================================
disk membership  1[d_ij <= R]         sigmoid((R - d_ij) / tau_d)
happiness        1[n_same >= K_MIN]   sigmoid((n_same - K_MIN) / tau_h)
unhappy agents teleport to random     unhappy-gated gradient ascent of each
locations until happy                 agent's own soft same-type count
====================================  =========================================

As tau_d, tau_h -> 0 the soft counts and gates converge pointwise to the
discrete rules. The *kinetics* remain different by design (local drift vs
global random teleports); see ``pytorch/README.md`` for the honest discussion
of what is gained and lost in the relaxation.

Conventions
-----------
* N = number of agents. Positions ``X``: shape (N, 2). Types: (N,) long.
* All pairwise tensors are (N, N); memory is O(N^2), so use
  ``use_checkpoint=True`` in :func:`simulate` when backpropagating through
  long trajectories (activations are then recomputed instead of stored:
  memory O(N^2) total instead of O(T * N^2), at ~2x forward compute).
* Device-agnostic: tensors follow the device of the inputs;
  :func:`default_device` picks CUDA when available. Random initial
  conditions are drawn on CPU so a given seed yields the *same* initial
  configuration on CPU and GPU.
"""

import math
from dataclasses import dataclass, replace
from typing import Optional

import torch
from torch.utils.checkpoint import checkpoint

# Agent density of the Fortran reference run: 2000 agents on a 100 x 100 box.
# Keeping this density when changing N preserves the physics (expected
# same-type neighbours in a radius-10 disk ~ 31.4, threshold K_MIN = 30).
REFERENCE_DENSITY = 0.2


def default_device() -> torch.device:
    """CUDA if available (e.g. on Colab with a GPU runtime), else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    """Model + relaxation parameters. Defaults match the Fortran reference."""

    n_a: int = 1000          # type-1 agents
    n_b: int = 1000          # type-0 agents
    box: float = 100.0       # domain side L; agents live in [0, L]^2
    radius: float = 10.0     # neighbourhood radius R
    k_min: float = 30.0      # same-type neighbours required to be "happy"
    tau_d: float = 1.0       # softness of the disk edge (length units)
    tau_h: float = 3.0       # softness of the happiness gate (count units)
    step_size: float = 0.5   # drift step eta (length units per step)
    max_step: float = 2.0    # cap on per-step displacement (stability)
    n_steps: int = 150       # default trajectory length
    seed: int = 20260707     # same seed convention as the Fortran run

    @staticmethod
    def scaled(n_per_type: int, **overrides) -> "Config":
        """A config with fewer/more agents at the *reference density*.

        The box is rescaled so that n / box^2 = REFERENCE_DENSITY, which
        keeps the expected neighbour counts (and hence the meaning of
        ``k_min``) identical to the N = 2000 reference run.
        """
        box = math.sqrt(2 * n_per_type / REFERENCE_DENSITY)
        return replace(Config(), n_a=n_per_type, n_b=n_per_type, box=box,
                       **overrides)


def init_state(cfg: Config, device: Optional[torch.device] = None,
               dtype: torch.dtype = torch.float32):
    """Random initial condition: uniform positions, first n_a agents type 1.

    Returns
    -------
    X     : (N, 2) float tensor of positions in [0, box)^2
    types : (N,)   long tensor of agent types (1 / 0)
    """
    device = default_device() if device is None else device
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    n = cfg.n_a + cfg.n_b
    X = torch.rand(n, 2, generator=gen, dtype=dtype) * cfg.box   # (N, 2)
    types = torch.cat([torch.ones(cfg.n_a, dtype=torch.long),
                       torch.zeros(cfg.n_b, dtype=torch.long)])  # (N,)
    return X.to(device), types.to(device)


def _pairwise(X: torch.Tensor, eps: float = 1e-9):
    """Pairwise displacements and distances.

    diff[i, j] = x_i - x_j, shape (N, N, 2);  d[i, j] = |x_i - x_j|, (N, N).
    The +eps inside the sqrt keeps the gradient finite on the diagonal
    (d = 0), where it would otherwise be NaN; diagonal terms are masked out
    of every sum, and finite * 0 = 0, so no NaN can leak into the graph.
    """
    diff = X[:, None, :] - X[None, :, :]                 # (N, N, 2)
    d = torch.sqrt((diff * diff).sum(-1) + eps)          # (N, N)
    return diff, d


def soft_counts(X: torch.Tensor, types: torch.Tensor, radius, tau_d,
                eps: float = 1e-9):
    """Soft neighbour counts within the (relaxed) radius-R disk.

    ``radius`` may be a python float or a 0-dim tensor (possibly with
    ``requires_grad=True`` — it then enters the autodiff graph).

    Returns ``n_same``, ``n_total``: (N,) soft counts. In the limit
    tau_d -> 0 they converge to the integer counts of the Fortran code.
    """
    _, d = _pairwise(X, eps)
    kernel = torch.sigmoid((radius - d) / tau_d)          # (N, N) soft disk
    not_self = ~torch.eye(X.shape[0], dtype=torch.bool, device=X.device)
    kernel = kernel * not_self
    same = types[:, None] == types[None, :]               # (N, N) bool
    n_same = (kernel * same).sum(dim=1)                   # (N,)
    n_total = kernel.sum(dim=1)                           # (N,)
    return n_same, n_total


def soft_happiness(n_same: torch.Tensor, k_min, tau_h) -> torch.Tensor:
    """Soft happiness gate h in (0, 1); -> Heaviside(n_same - k_min) as tau_h -> 0."""
    return torch.sigmoid((n_same - k_min) / tau_h)


def soft_metrics(n_same, n_total, k_min, tau_h, eps: float = 1e-9):
    """Differentiable observables of a configuration.

    frac_happy : mean soft happiness.
    seg_index  : mean same-type fraction among each agent's neighbours,
                 weighted by w = n_total / (n_total + 1) so that agents with
                 (softly) zero neighbours drop out — this reduces to the
                 Fortran metric ("skip isolated agents") whenever everyone
                 has neighbours, which is the case at the reference density.
    """
    h = soft_happiness(n_same, k_min, tau_h)
    w = n_total / (n_total + 1.0)
    seg = (w * n_same / (n_total + eps)).sum() / (w.sum() + eps)
    return h.mean(), seg


@torch.no_grad()
def hard_metrics(X: torch.Tensor, types: torch.Tensor, radius, k_min):
    """The *discrete* observables (exact Fortran definitions), for comparison.

    Hard disk membership, hard threshold, isolated agents skipped. Used to
    evaluate configurations produced by the relaxed dynamics against the
    Fortran reference values (0.498 mixed -> 0.629 converged).
    Returns (frac_happy, seg_index) as python floats.
    """
    radius = float(radius)
    k_min = float(k_min)
    diff = X[:, None, :] - X[None, :, :]
    d2 = (diff * diff).sum(-1)
    inside = d2 <= radius * radius
    inside.fill_diagonal_(False)
    same = types[:, None] == types[None, :]
    n_same = (inside & same).sum(dim=1)
    n_tot = inside.sum(dim=1)
    frac_happy = (n_same.float() >= k_min).float().mean().item()
    has_neigh = n_tot > 0
    seg = (n_same[has_neigh].float() / n_tot[has_neigh].float()).mean().item()
    return frac_happy, seg


def relaxation_step(X, types, radius, k_min, tau_d, tau_h,
                    step_size, max_step, box, eps: float = 1e-9):
    """One relaxed relocation step (the differentiable analogue of a sweep
    of Fortran relocation events).

    Each agent ascends the gradient of ITS OWN soft same-type count, gated
    by its soft unhappiness:

        x_i  <-  x_i + eta * (1 - h_i) * grad_{x_i} n_same_i

    The ascent direction has the closed form

        grad_{x_i} n_same_i = sum_j same_ij * s_ij (1 - s_ij) / tau_d
                                       * (x_j - x_i) / d_ij

    with s_ij = sigmoid((R - d_ij)/tau_d): attraction toward same-type
    agents, dominated by those near the disk edge (s' peaks at d ~ R) —
    the "marginal neighbours" an agent is about to gain or lose. Computing
    the drift analytically keeps autograd free for the OUTER derivatives
    (w.r.t. k_min, radius, ...), avoiding a double-backward.

    All quantities stay in the autodiff graph: gradients w.r.t. ``k_min``
    flow through the gate (1 - h_i), gradients w.r.t. ``radius`` through
    both s_ij and the gate.
    """
    n = X.shape[0]
    diff, d = _pairwise(X, eps)                            # (N,N,2), (N,N)
    s = torch.sigmoid((radius - d) / tau_d)                # (N, N)
    not_self = ~torch.eye(n, dtype=torch.bool, device=X.device)
    same = (types[:, None] == types[None, :]) & not_self   # (N, N) bool

    n_same = (s * same).sum(dim=1)                         # (N,)
    h = torch.sigmoid((n_same - k_min) / tau_h)            # (N,)

    # w_ij = d(soft count)/d(distance) magnitude, masked to same-type pairs
    w = (s * (1.0 - s) / tau_d) * same                     # (N, N)
    # sum_j w_ij * (x_j - x_i)/d_ij  =  -sum_j w_ij * diff_ij / d_ij
    pull = -(w[:, :, None] * diff / d[:, :, None]).sum(dim=1)   # (N, 2)

    drift = step_size * (1.0 - h)[:, None] * pull          # (N, 2)

    # Cap the per-step displacement (relaxation stability, akin to a CFL
    # condition). clamp(max=1) keeps the rescaling differentiable a.e.
    norm = drift.norm(dim=1, keepdim=True)                 # (N, 1)
    drift = drift * torch.clamp(max_step / (norm + eps), max=1.0)

    # Stay inside the domain. Note: clamp has zero gradient outside the
    # box, so agents pinned to a wall stop feeling parameter gradients —
    # acceptable here (few agents, box mostly empty near walls; README).
    return (X + drift).clamp(0.0, box)


def simulate(X0, types, cfg: Config, k_min=None, radius=None,
             n_steps: Optional[int] = None, record_every: int = 1,
             use_checkpoint: bool = False):
    """Unroll the relaxed dynamics for ``n_steps`` starting from ``X0``.

    ``k_min`` / ``radius`` may be 0-dim tensors with ``requires_grad=True``:
    they then influence every step, and gradients of any recorded
    observable w.r.t. them are available by ``backward()`` — this is what
    enables calibration and sensitivity analysis.

    With ``use_checkpoint=True`` each step's (N, N) activations are
    recomputed during backward instead of stored, so memory stays O(N^2)
    rather than O(T * N^2). Use it whenever you backpropagate through more
    than a handful of steps at N ~ 1000+.

    Returns a dict:
      ``X``          final positions (N, 2), differentiable;
      ``seg``        (T_rec,) soft segregation index at recorded steps;
      ``frac_happy`` (T_rec,) soft happy fraction at recorded steps;
      ``steps``      list of recorded step indices (ints).
    """
    device, dtype = X0.device, X0.dtype
    if k_min is None:
        k_min = torch.tensor(float(cfg.k_min), device=device, dtype=dtype)
    if radius is None:
        radius = torch.tensor(float(cfg.radius), device=device, dtype=dtype)
    if n_steps is None:
        n_steps = cfg.n_steps

    def one_step(X, k_min, radius):
        return relaxation_step(X, types, radius, k_min, cfg.tau_d, cfg.tau_h,
                               cfg.step_size, cfg.max_step, cfg.box)

    def observe(X, k_min, radius):
        n_same, n_total = soft_counts(X, types, radius, cfg.tau_d)
        fh, seg = soft_metrics(n_same, n_total, k_min, cfg.tau_h)
        return fh, seg

    def maybe_ckpt(fn, *args):
        if use_checkpoint and torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    X = X0
    seg_rec, happy_rec, steps = [], [], []

    fh, seg = maybe_ckpt(observe, X, k_min, radius)
    happy_rec.append(fh); seg_rec.append(seg); steps.append(0)

    for t in range(1, n_steps + 1):
        X = maybe_ckpt(one_step, X, k_min, radius)
        if t % record_every == 0 or t == n_steps:
            fh, seg = maybe_ckpt(observe, X, k_min, radius)
            happy_rec.append(fh); seg_rec.append(seg); steps.append(t)

    return {"X": X,
            "seg": torch.stack(seg_rec),
            "frac_happy": torch.stack(happy_rec),
            "steps": steps}
