# YABI test suite

Every test lives below `tests/` and is registered with CTest. The suite is
split into three tiers; each tier is a CTest label, so it can be run on its own
in CI and during development.

| directory | label | scope |
|---|---|---|
| `tests/unit/` | `unit` | one executable per component; isolated contracts, no full integration run |
| `tests/component/` | `component` | cross-component behaviour (native sparse assembly, matrix-free Newton–Krylov, SuiteSparse symbolic reuse) |
| `tests/system/` | `system` | mathematical benchmarks and regressions on canonical ODE problems |

The self-checking programs in `examples/` are registered under the `example`
label by the top-level `CMakeLists.txt`.

## Running

```sh
ctest --test-dir build --output-on-failure          # every tier
ctest --test-dir build -L unit --output-on-failure  # unit only
ctest --test-dir build -L component --output-on-failure
ctest --test-dir build -L system --output-on-failure
ctest --test-dir build -L example --output-on-failure
```

Build both configurations to exercise the whole matrix:

```sh
cmake -S . -B build                          # CppAD ON
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure

cmake -S . -B build-pure -DYABI_USE_CPPAD=OFF
cmake --build build-pure -j"$(nproc)"
ctest --test-dir build-pure --output-on-failure
```

## Unit tier

Each file under `tests/unit/` is an independent executable with a `main()`. It
links directly against the header-only component under test and checks public
behaviour, boundary cases, and failure paths. `test_support.hpp` provides the
shared `check` / `near` / `finish` reporting helpers; `engine_fixture.hpp`
provides the dummy dependencies shared by the Newton, sensitivity, and adjoint
engines.

Current coverage:

- `test_config_statistics` — `IntegratorConfig`, `IntegratorStatistics`
- `test_replay_tape` — `ReplayTapeStore` allocation, stability, release
- `test_nordsieck_state` — history sizing, `l_i`, predictor, corrector, dense output
- `test_step_controller` — error-weight modes
- `test_newton_solver` — owned work arrays and row/column scaling
- `test_sensitivity_engine` — forward solution, tape reset, replay buffers
- `test_adjoint_engine` — adjoint storage lifetime and `release()`
- `test_linear_solver` — dense LAPACK path and the CSC reference path
- `test_sparsity_pattern` — column-oriented pattern equality and shape
- `test_csc_matrix` — CSC construction, diagonal insertion, scaling, validation
- `test_cppad_rhs_sparsity` — CppAD sparse Jacobian detection, values, `J·v`,
  and multi-direction batch `J·V` (CppAD builds only)
- `test_user_ode_rhs` — pure/analytic sparsity pattern validation, Jacobian
  extraction order, analytic/default `J·v`, batch `J·V`, clone preservation,
  invalid input rejection (both configurations)
- `test_krylov_solver` — GMRES/FGMRES/BiCGSTAB, preconditioners, direct
  fallback, and dense delegation (both configurations)
- `test_preconditioner` — `Preconditioner`/`FunctionPreconditioner` contract,
  user-preconditioner precedence, assembled vs J-free update, diagonal
  extraction, forcing context and Eisenstat–Walker clamping (both
  configurations)
- `test_sensitivity_linearization` — batched `J·V`, `applyBlock`, and the
  `SensitivityLinearization` adapter (both configurations)

The CppAD RHS test is excluded automatically when `YABI_USE_CPPAD=OFF`; all
other unit tests run in both configurations.

## Component tier

- `native_sparse.cpp` — CppAD sparse Jacobian → native CSC → sparse Newton path;
  asserts bit-identical results against the dense path.
- `matrix_free_newton.cpp` — HIRES state solve through GMRES/FGMRES/BiCGSTAB and
  the `J·v` operator, with Jacobi and exact-CSC preconditioners; asserts the
  direct trajectory is reproduced with no CSC fallback, that
  `operatorApplications() > 0`, and that forward sensitivities still follow the
  explicit `J`/`H` path (CppAD builds).
- `matrix_free_pure.cpp` — the same matrix-free solve driven by `UserODERHS`
  with an analytic pattern and `J·v`; runs in both configurations, asserts a
  `< 1e-9` match to the dense trajectory, a non-zero operator count, and a
  one-time diagnostic (not a silent downgrade) when the pattern is missing.
- `matrix_free_jfree.cpp` — the same Brusselator through `UserODERHS` with **no
  sparsity pattern** and a `requiresAssembledMatrix() == false` preconditioner;
  asserts the dense trajectory is matched, the operator/preconditioner ran, the
  update saw a null matrix and live operator, and no missing-pattern warning was
  logged (both configurations).
- `suitesparse_sparse.cpp` — UMFPACK/KLU numeric factorization with symbolic
  reuse; registered only when both libraries are found.

## System tier

`tests/system/` contains the mathematical testbench. Numbered `*.cpp` and
`manual_*.cpp` programs build CppAD tapes and are registered only in CppAD
builds; `pure_*.cpp` use `user_ode.hpp` and run in both configurations. See
`tests/system/README.md` for the problem list and measured convergence slopes.

## Adding a test

1. Put it in the tier that matches its scope.
2. For a unit test, add `tests/unit/test_<component>.cpp`; the glob in
   `tests/CMakeLists.txt` picks it up automatically.
3. Give every test a CTest label. System/component tests use the
   `yabi_register_test(source label)` helper.
4. Keep failure output on stderr and return non-zero from `main()` on failure.
