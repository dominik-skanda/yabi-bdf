# YABI-BDF numerical baseline (T30 decomposition — behaviour-preservation anchor)

Captured 2026-11-04, before the T30 `BDFIntegrator` decomposition refactor. **These values must not differ after the
refactor.** The refactor is purely structural (separation of responsibilities); any numeric drift is a regression.

## Provenance — comparison against the initial commit (`cb357e3`)

The initial commit was built in isolation and its outputs compared against this baseline. **They differ**, and the
difference is fully explained (no unexplained regression):

| anchor | initial `cb357e3` | baseline (`main`) |
|---|---|---|
| `yabi_basic` error | `4.32592239807406e-10` | `6.53219478508049e-10` |
| `13_hires` y1 | `0.000737133157726` | `0.000737133286353` |
| `05_robertson` | `y1=0.107300429907 …` | identical |
| `12_brusselator` | `y1(5) = 0.938299 …` | identical |
| `15_oregonator` | `t=1 => (112559.4758, …)` | identical |

The change is introduced by the **first commit after the initial one**, `8f5492c`
`fix(step-control): reduce BDF order on rejected steps (DASSL/LSODE-style)` — an intentional correctness fix. Every
commit in the history was built and run: there are exactly two value regimes, the switch occurs at `8f5492c`, and no
later commit (through `main`) changes either anchor.

**Do not read the smaller initial error as “better”.** The initial `yabi_basic` error (`4.33e-10`) is *smaller* than the
baseline (`6.53e-10`), but both are ~`1e-10` absolute against `y(2) ≈ 0.368` (≈0.6 ppm) — at the requested tolerance
floor. A different sequence of accepted steps/orders simply lands closer by coincidence; global error is not minimised
by the controller. The current algorithm is the correct one (order reduction on rejection per DASSL/LSODE), so the
baseline is `main` from `8f5492c` onward, **not** `cb357e3`. Testbenches `31`/`32` did not exist at the initial commit,
and `28` was a different (framework-only) program, so they have no initial-commit counterpart.

## Test suites

Tests are organised in three tiers below `tests/` and selected with ctest labels:

| tier | label | contents |
|---|---|---|
| unit | `unit` | isolated component contracts (config, replay tape, Nordsieck state, step controller, Newton, sensitivity, adjoint, dense/CSC solver, sparsity, CscMatrix, CppAD RHS sparsity, Krylov solver) |
| component | `component` | cross-component integration (`native_sparse`, `matrix_free_newton`, plus `suitesparse_sparse` when SuiteSparse is available) |
| system | `system` | mathematical benchmarks and regressions (`tests/system/`) |

- `ctest --test-dir build` (CppAD): **58/58 pass** — 12 unit, 3 component (including
  `suitesparse_sparse` when SuiteSparse is available), 40 system, 3 example.
- `ctest --test-dir build-pure` (no CppAD): **21/21 pass** — 11 unit, 7 pure system,
  3 example.

## Key numeric anchors (representative outputs, verbatim)

`yabi_basic` (decay `y'=-y`, t=2):
```
y(2) = 0.367879440518223
exact = 0.367879441171442
error = 6.53219478508049e-10
```

`05_stiff_robertson_autonomous` (t=1e4):
```
y1=0.107300429907 y2=4.800167041e-07 y3=0.892699090076 sum=1
```

`12_brusselator_autonomous` (t=5):
```
y1(5) = 0.938299, y2(5) = 4.52728
```

`13_hires_autonomous` (t=321.812):
```
y = (0.000737133286353,0.000144248972571,5.88876753153e-05,0.00117565512229,0.00238641718911,0.00623915986836,0.00285004111321,0.00284995888679)
```

`15_oregonator_autonomous`:
```
t=1 => y=(112559.4758,0.05952720969,6712.895565) OK
t=5 => y=(1.001753339,571.3421806,23962.69912) OK
t=10 => y=(1.000670524,1492.35853,10713.98688) OK
```

`28_second_order_sensitivity`:
```
forward(1) = 4 (exact 4) rel=1.55578e-09
forward(2) = 8 (exact 8) rel=5.84036e-09
reverse(2) = [order1 4, order2 8] rel=(1.55578e-09,9.95208e-09)
```

`31_dense_output` (interior points of the last step):
```
theta=-0.50 t=2.997174190003 y=0.049927955809049 exact=0.049927956130660 err=3.216e-10 OK
theta=-0.25 t=2.998587095001 y=0.049857462162821 exact=0.049857462484014 err=3.212e-10 OK
```

`32_fixed_order` (per-order convergence slopes):
```
order 5: slope 5.229   order 6: slope 5.983
```
