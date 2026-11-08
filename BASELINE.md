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
`fix(step-control): reduce BDF order on rejected steps` — an intentional correctness fix. Every
commit in the history was built and run: there are exactly two value regimes, the switch occurs at `8f5492c`, and no
later commit (through `main`) changes either anchor.

**Do not read the smaller initial error as “better”.** The initial `yabi_basic` error (`4.33e-10`) is *smaller* than the
baseline (`6.53e-10`), but both are ~`1e-10` absolute against `y(2) ≈ 0.368` (≈0.6 ppm) — at the requested tolerance
floor. A different sequence of accepted steps/orders simply lands closer by coincidence; global error is not minimised
by the controller. The current algorithm is the correct one (order reduction on rejection), so the
baseline is `main` from `8f5492c` onward, **not** `cb357e3`. Testbenches `31`/`32` did not exist at the initial commit,
and `28` was a different (framework-only) program, so they have no initial-commit counterpart.

## Test suites

Tests are organised in three tiers below `tests/` and selected with ctest labels:

| tier | label | contents |
|---|---|---|
| unit | `unit` | isolated component contracts (config, replay tape, Nordsieck state, step controller, Newton, sensitivity, adjoint, dense/CSC solver, sparsity, CscMatrix, CppAD RHS sparsity/batch `J·V`, `UserODERHS` sparsity/`J·v`/batch `J·V`, Krylov solver, pluggable preconditioner/forcing, sensitivity linearization) |
| component | `component` | cross-component integration (`native_sparse`, `matrix_free_newton`, `matrix_free_pure`, `matrix_free_jfree`, plus `suitesparse_sparse` when SuiteSparse is available) |
| system | `system` | mathematical benchmarks and regressions (`tests/system/`) |

- `ctest --test-dir build` (CppAD): **64/64 pass** — 15 unit, 5 component (including
  `suitesparse_sparse` when SuiteSparse is available), 41 system, 3 example.
- `ctest --test-dir build-pure` (no CppAD): **26/26 pass** — 14 unit, 2 component,
  7 pure system, 3 example.

T32c (pluggable/J-free preconditioners and forcing) and T32d (batch/sparse/matrix-free sensitivity interfaces)
are additive: every pre-existing numeric anchor below is unchanged, `yabi_basic` remains
`6.53219478508049e-10`, and the sensitivity replay values are identical.

T29 (RAII memory management) and the arbitrary-order (3+) forward/reverse sensitivity work are likewise
behaviour-preserving for the *nominal* trajectory: every anchor below except `28_second_order_sensitivity`
is bit-identical after both. `28` itself was extended from second order to orders 1–4 and its `forward(2)`
now runs the full-Taylor replay, so its printed `forward(2)` relative error moved from `5.84e-09` to
`9.95e-09` (the value `8` is unchanged, still ≈1e-9).

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

`28_second_order_sensitivity` (orders 1–4):
```
forward(1) = 4 (exact 4) rel=1.55578e-09
forward(2) = 8 (exact 8) rel=9.95208e-09
forward(3) = 24 (exact 24) rel=4.2256e-08
forward(4) = 96 (exact 96) rel=1.32408e-07
reverse(4) = [order1 4 order2 8 order3 24 order4 96]
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

## Additive test/evidence + opt-in damped Newton (T41–T45) — anchors unchanged

The test-coverage and evidence effort (T41–T45: unit/component/system tiers, the CWI stiff-chemistry benchmarks
POLLU/BEAM/EMEP/MEDAKZO, the telemetry harness, the EMEP robustness investigation, and the damped-Newton feature) is
**behaviour-preserving for the defaults**.  It added `tests/`, `benchmarks/`, and CI wiring, plus one opt-in feature:
a damped Newton (backtracking line search) behind `setNewtonLineSearch(bool)` — a new config flag that is **off by
default** and gated to the plain, non-taped state solve, so the historical full-step behaviour is reproduced
bit-for-bit when it is not enabled.  Consequently every anchor above remains **byte-identical**, including `yabi_basic`
error `6.53219478508049e-10`.  The EMEP benchmark (stiffness ratio ~1e14) originally needed a conservative
*configuration* (`setFixedOrder(4)` + fresh Jacobian per iteration + `setNewtonLineSearch(true)`); that was a
work-around for a one-character boundary bug in the minimal-step acceptance (`h < 4·eps·|t|` vs the floor's exact
`h = 4·eps·|t|`).  Fixing the test to `h <= 4·eps·|t|` lets EMEP match its reference with defaults (max order 5, no
line search, no fixed order) — see README "Extreme-stiffness behaviour (EMEP)" — and is likewise a no-op on
well-behaved problems, so it leaves every anchor above untouched.

