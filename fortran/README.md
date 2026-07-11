# Fortran reference implementation

High-performance forward simulation of the continuous-space Schelling-type
segregation model, in strict standard **Fortran 95**. This is the reference
implementation of the project: the differentiable PyTorch twin
([`../pytorch`](../pytorch)) is validated against its output.

## Build & run

```bash
make            # gfortran -O2 -std=f95  ->  ./segregation
./segregation
```

Other targets: `make debug` (bounds checking + backtraces, `-fcheck=all`)
and `make clean`.

## Parameters

All parameters are compile-time constants at the top of
[`segregation.f90`](segregation.f90):

| Parameter | Value | Meaning |
|---|---|---|
| `N_A`, `N_B` | 1000, 1000 | agents per type |
| `L` | 100.0 | domain side, `[0,L) × [0,L)`, open boundaries |
| `R` | 10.0 | neighbourhood radius (Euclidean disk) |
| `K_MIN` | 30 | same-type neighbours required to be happy |
| `MAX_MOVES` | 10000 | relocation-event budget (stops earlier on convergence) |
| `MAX_TRIES` | 40 | random trial locations per relocation event |
| `SEED` | 20260707 | RNG seed — runs are fully reproducible |
| `M` | 10 | cells per side of the neighbour-search grid (must keep `L/M ≥ R`; asserted at startup) |

## Output files

| File | Content |
|---|---|
| `metrics.dat` | time series: event, unhappy count, fraction happy, segregation index |
| `snapshots.dat` | full configuration `x y type` every 1,000 events, blocks separated by two blank lines (gnuplot `index` convention) |
| `final_configuration.dat` | configuration at convergence |

Quick look with gnuplot:

```gnuplot
plot 'final_configuration.dat' u 1:2:($3+1) w points pt 7 ps 0.5 lc variable
```

## Implementation notes

- **Neighbour search** uses cell-linked lists (the standard
  molecular-dynamics structure) on an `M × M` grid; the 3×3 block search is
  exact because the cell size satisfies `L/M ≥ R`, asserted at startup.
- **Incremental bookkeeping:** after each relocation event only the agents
  within radius `R` of the departure and arrival points are re-evaluated —
  a teleport cannot change any other agent's neighbour count. The unhappy
  set is maintained exactly with O(1) insert/remove via an inverse index map.
- **Reproducibility:** the RNG is seeded deterministically from the `SEED`
  parameter; repeated runs produce bit-identical metrics.
- **Verification:** compiles clean under
  `gfortran -std=f95 -pedantic -Wall -Wextra` and runs clean under
  `-fcheck=all`; the final state is cross-checked by an independent
  brute-force O(N²) recomputation of every agent's happiness and of the
  segregation index (agreement to 6 decimal places).
