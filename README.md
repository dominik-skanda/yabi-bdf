# YABI — Yet Another BDF Integrator

> **Note on origin:** This is a complete rewrite of the BDF integrator developed
> by Dominik Skanda (based on his dissertation work at University of Freiburg).
> The mathematical formulation follows BDF methods with sensitivity generation as presented in the [dissertation by Dominik Skanda](https://freidok.uni-freiburg.de/data/8787).  
> It supports dense (LAPACK), sparse (UMFPACK/KLU via SuiteSparse), and matrix-free Newton–Krylov
> (GMRES/FGMRES/BiCGSTAB via Jacobian–vector products) linear solvers through a pluggable `LinearSolver` interface.
> The rewrite was carried out using **OpenCode** with the **Big Pickle** model and
> **OpenRouter** free models. All algorithmic components (Nordsieck predictor,
> simplified Newton, matrix scaling/conditioning strategy, corrector acceptance,
> and sensitivity replay) were reimplemented to match the mathematical structure
> of the original approach while exposing a clean modern interface.

YABI is a variable-order, variable-step BDF integrator for stiff ordinary
differential equations. It provides forward, adjoint (reverse) and simultaneous
sensitivity analysis through CppAD automatic differentiation.

- **Orders 1–6**, variable order and variable step size.
- **Three right-hand-side forms**: autonomous `f(y)`, time dependent `f(t, y)`,
  and time/parameter dependent `f(t, y, p)`.
- **Sensitivities**: forward mode, adjoint/reverse mode, and simultaneous
  (variational) sensitivities.
- **Stiff problems**: implicit corrector with Newton iteration, Jacobian reuse,
  and LAPACK linear solves.
- **C++17**, BLAS/LAPACK, and an external CppAD installation.

## Quick Verification

Prove tolerance adherence, first-order convergence, and second-order sensitivity
in one command. This regenerates the work–precision diagram
(`work_precision_diagram.png`) and prints the slope summary plus the
second-order sensitivity output locally.

```sh
cmake -S . -B build && cmake --build build -j
python3 reproduce_benchmarks.py
```

`reproduce_benchmarks.py` runs two self-checking testbench binaries and parses
their output:

- `build/yabi_test_11_tolerance_and_order` — the work–precision proof (tolerance
  adherence + first-order step convergence).
- `build/yabi_test_28_second_order_sensitivity` — exact second-order forward and
  reverse sensitivities.

Plotting requires `matplotlib` (the build and the two binaries themselves do
not):

```sh
pip install matplotlib
```

## Dependencies

| Dependency | Notes |
|---|---|
| CppAD | Optional. For automatic differentiation / examples; can be disabled or not required in some builds. Must provide `ADFun::capacity_order` (2013 or newer) if used. |
| BLAS | Fortran `dgemm_`, `dgemv_`, `dger_`, `daxpy_`, `dscal_`, `dnrm2_`. |
| LAPACK | Fortran `dgetrf_`, `dgetrs_` (for dense linear solver). |
| SuiteSparse (UMFPACK/KLU) | Optional, for sparse linear solvers via the pluggable `LinearSolver` interface. |
| C++17 compiler | GCC or Clang. |
| CMake ≥ 3.16 | Build only. |
| matplotlib | Optional. Only for `reproduce_benchmarks.py`'s work–precision plot (and `pyplot/plot_all.py`). |

CppAD is distributed under EPL-2.0 with a GPL-2.0-or-later secondary license.
BLAS and LAPACK are BSD-style. YABI itself is Apache-2.0.

On Debian/Ubuntu:

```sh
sudo apt install build-essential liblapack-dev libblas-dev cmake libcppad-dev
```

## Layout

```
yabi-bdf/
  include/yabi/
    bdf_integrator.hpp        public class declaration
    linear_algebra.hpp        BLAS/LAPACK bindings
    detail/
      core.ipp                construction, reset, configuration, results
      step_control.ipp        step size, predictor, order control, output
      newton.ipp              Newton iteration and corrector acceptance
      sensitivity_setup.ipp   sensitivity storage and tape lifetime
      sensitivity_forward.ipp forward-mode replay
      sensitivity_reverse.ipp reverse/adjoint replay
  src/
    bdf_integrator.cpp        explicit instantiation for double
  examples/
  tests/
    unit/                     isolated component unit tests
    component/                cross-component integration tests
    system/                   mathematical benchmarks and regressions
```

The `.ipp` files contain the member definitions of the class template. They
are implementation details: include `<yabi/bdf_integrator.hpp>`, not the
`.ipp` files.

## Build

```sh
cmake -S . -B build [-DCPPAD_ROOT=/path/to/cppad]
cmake --build build -j
ctest --test-dir build --output-on-failure
```

If CppAD is not installed in standard paths:

```sh
cmake -S . -B build -DCPPAD_ROOT=/path/to/cppad
cmake --build build -j$(nproc)
ctest --test-dir build
```

When CppAD is installed system-wide, `-DCPPAD_ROOT` / `CPPAD_ROOT` can be
omitted. CMake first tries `pkg-config`, then looks for `cppad/cppad.hpp` and
`libcppad_lib`.

## Usage

### User ODE formulation
- **CppAD / automatic differentiation:** Pass an `ADFun<double>` (tape) describing the RHS — sensitivities/Jacobians can be handled via AD/Internal Numerical Differentiation.
- **Analytical derivatives:** Use the user-ODE interfaces in `include/yabi/user_ode.hpp` and the manual sensitivity examples (`examples/manual_sensitivity_demo.cpp`, `examples/manual_sensitivity_pure.cpp`) to provide analytic Jacobians/sensitivity RHS without requiring CppAD.
- **Custom sparse solvers:** Implement `yabi::LinearSolver` (`include/yabi/linear_solver.hpp`) to plug your own sparse direct/iterative solver; set it via `integrator.setLinearSolver(...)`. UMFPACK/KLU are provided via SuiteSparse; dense LAPACK is the default.

```cpp
#include <yabi/bdf_integrator.hpp>

CppAD::vector<CppAD::AD<double>> x(1);
x[0] = 1.0;                       // y(0)
CppAD::Independent(x);

CppAD::vector<CppAD::AD<double>> f(1);
f[0] = -0.5 * x[0];

CppAD::ADFun<double> rhs;
rhs.Dependent(x, f);
rhs.optimize();

yabi::BDFIntegrator<double> integrator(rhs);

double y0[1]       = {1.0};
double abs_error[1] = {1e-12};
double y[1];

integrator.resetIntegration(y0, 0.1, 1e-10, abs_error,
                            false, false, nullptr, nullptr, nullptr, nullptr, 0);
integrator.setAlgorithmSettings(5, std::numeric_limits<double>::epsilon(), true, 1);
integrator.performIntegration(2.0, 100000);
integrator.getSolutionVector(y);
```

Dense output: `getSolutionAt(t, y)` evaluates the continuous extension at any `t`
inside the last accepted step `[t_end - h, t_end]` (it returns `false` outside):

```cpp
double h = integrator.getCurrentStepSize();   // last step size
integrator.getSolutionAt(2.0 - 0.5 * h, y);   // interpolate inside the last step
```

The seven trailing sensitivity-seed arguments can be replaced by a
`yabi::IntegrationOptions` struct (defaults reproduce the `false, false,
nullptr, …, 0` tail):

```cpp
yabi::IntegrationOptions<double> opts;             // taping=false, direct=false,
integrator.resetIntegration(y0, 0.1, 1e-10, abs_error, opts);  // seeds=nullptr, numOfderivatives=0
```

For sensitivity, fill the fields and pass the same struct to any of the three
overloads (autonomous / time / time+parameter):

```cpp
opts.taping = false;
opts.direct = false;
opts.y_seed_list              = yseed_ptr.data();   // double** (per-direction state seeds)
opts.p_seed_list              = pseed_ptr.data();   // double** (per-direction parameter seeds)
opts.relative_error_derivatives = rel_err.data();   // double*  (per-direction rel. tol)
opts.absolut_error_derivatives   = abs_ptr.data();  // double** (per-direction abs. tol)
opts.numOfderivatives         = ndir;
integrator.resetIntegration(y0, t0, p, max_step, 1e-10, abs_error, opts);
```

The RHS type is inferred from the dimensions of the CppAD tape:

| Tape domain | Tape range | Meaning |
|---|---|---|
| `n` | `n` | autonomous: `f(y)` |
| `n + 1` | `n` | time dependent: `f(t, y)` |
| `n + 1 + p` | `n` | time and parameter dependent: `f(t, y, p)` |

`setAlgorithmSettings(max_order, machine_precision, hit_end_time,
error_weight_index)` controls the maximum BDF order (1–6), the precision used
for the final-step test, whether the integrator must land exactly on the end
time, and which error-weight norm to use. All four arguments are validated
before any state is changed.

`max_order` is a **cap**, not a forced order: the controller chooses the order
actually used. Use `getCurrentOrder()` to read it. Likewise `max_StepLength` caps
the step but is not an accuracy knob. Control accuracy with `rtol`/`atol`; because the
controller changes order as the step cap changes, the error is not monotone in
`max_StepLength` (measured examples in `tests/system/README.md`).

To **force** a fixed order (e.g. for per-order convergence measurement), use
`setFixedOrder(q)`:

```cpp
integrator.setFixedOrder(4);        // pin the order to 4 (ramps up 1..4, then stays)
integrator.getCurrentOrder();       // read the order actually used
```

### Step-controller adaptation (stability)

The step controller is the original 2011 BDF algorithm with one small, deliberate
addition for stability: on a rejected step it estimates the local truncation error at the
next-lower order (from the highest Nordsieck term) and reduces the order when that gives a
smaller or already-acceptable error. This DASSL/LSODE-style order reduction lets the
integrator recover from stiff-oscillatory problems — e.g. the Oregonator's relaxation
oscillations, where high-order BDF is unstable at *any* step size. It is a **no-op for
well-behaved problems** (they accept on the first attempt, so no order reduction ever fires),
and each reduction is recorded in the sensitivity tape so forward/reverse replays reconstruct
the correct order sequence.

## Sensitivity API

### Forward mode

Enable taping in `resetIntegration`, integrate to the output time, then call
`forward`:

```cpp
double y_seed[2] = {1.0, 0.0};    // d/dy0
double p_seed[2] = {0.0, 0.0};
double direction[2];

integrator.forward(1, y_seed, p_seed, false);
integrator.getForwardSolutionVector(direction);
```

For a parameter derivative, set `y_seed` to zero and the corresponding entry
of `p_seed` to one. Calling `forward(..., true)` at the same output time resets
the forward replay so that a new direction can be requested.

### Reverse / adjoint mode

Integrate to the output time with taping enabled, then seed the adjoint with
the weights of the scalar output of interest:

```cpp
double weight[2] = {1.0, 0.0};
double* adj_species[2];
double* adj_parameter[2];

integrator.reverse(1, weight);
integrator.getReverseSolutionVector(adj_species, adj_parameter);
```

`adj_species[i][k]` is the k-th order adjoint of state `i`;
`adj_parameter[i][k]` is the corresponding parameter adjoint.

### Simultaneous (variational) sensitivities

Pass seed directions and error controls directly to `resetIntegration`:

```cpp
double* y_seeds[ndir];
double* p_seeds[ndir];
double* abs_seeds[ndir];
double rel_seed = 1e-9;
// fill y_seeds[d], p_seeds[d] and abs_seeds[d] ...

integrator.resetIntegration(y0, t0, p, max_step, rel_error, abs_error,
                            false, false, y_seeds, p_seeds, &rel_seed,
                            abs_seeds, ndir);
integrator.performIntegration(t_end, max_steps);
integrator.getSimultaneousDerivativesVector(solution_derivatives);
```

`examples/sensitivity_demo.cpp` demonstrates all three modes and checks them
against central finite differences.

### Second-order (forward / reverse)

`forward` and `reverse` take an `order` argument. Order `q` is computed
incrementally: CppAD's `Forward(q)` needs the lower Taylor orders first, so to
obtain the second-order directional derivative you run order 1 first:

```cpp
integrator.forward(1, seed, pseed, true);
integrator.forward(1, seed, pseed, false);
integrator.forward(2, seed, pseed, true);
integrator.forward(2, seed, pseed, false);
integrator.getForwardSolutionVector(dir2);   // d^2 y / d seed^2

integrator.reverse(2, weights);              // 2nd-order adjoint
integrator.getReverseSolutionVector(adj_sp, adj_p);   // adj_sp[i][1] = 2nd-order
```

A direct `forward(2)` without a prior `forward(1)` is not valid. The
`tests/system/28_second_order_sensitivity.cpp` driver verifies second order against
the exact analytical solution of `y' = y²`.

## Tests

The tests are split into three ctest tiers below `tests/`:

- `tests/unit/` — one focused executable per component (`IntegratorConfig`,
  `ReplayTape`, `NordsieckState`, `StepController`, `NewtonSolver`,
  `SensitivityEngine`, `AdjointEngine`, `DenseLUSolver`/CSC reference path,
  `SparsityPattern`, `CscMatrix`, CppAD RHS sparsity/`J·v`, and the Krylov
  solver). These are fast, dependency-light contract tests.
- `tests/component/` — cross-component integration: native CSC assembly and
  the CppAD sparse Jacobian path (`native_sparse`), matrix-free Newton–Krylov
  against the direct HIRES trajectory (`matrix_free_newton`), plus symbolic
  reuse with UMFPACK/KLU (`suitesparse_sparse`, registered only when
  SuiteSparse is available).
- `tests/system/` — the mathematical benchmark suite: linear problems,
  Robertson, Van der Pol, Brusselator, HIRES, Oregonator, Prothero–Robinson,
  damped pendulum, E5, large sparse/network systems, forward/reverse/
  simultaneous sensitivities, second order, dense output, fixed order, and
  failure/tolerance regressions.

Every test checks its result against an independent reference solution.
Run a single tier with `ctest --test-dir build -L unit`, `-L component`, or
`-L system`; `ctest` also registers the three self-checking examples under the
`example` label (`example_basic`, `example_nonautonomous`,
`example_sensitivity_demo`).

See `tests/system/README.md` for the measured convergence slopes; the per-order
convergence test for orders 2–6 is now expressible via `setFixedOrder` +
`getCurrentOrder` (`tests/system/32_fixed_order.cpp`).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).


## API Quick Start

### Basic integration (autonomous)
```cpp
#include <yabi/bdf_integrator.hpp>
#include <cppad/cppad.hpp>

int main() {
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(1);
    x[0] = 1.0;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(1);
    f[0] = -x[0];
    rhs.Dependent(x, f);
    rhs.optimize();

    yabi::BDFIntegrator<double> integrator(rhs);
    double y0[1] = {1.0}, abs_err[1] = {1e-10};
    integrator.resetIntegration(y0, 2.0, 1e-10, abs_err, false, false, NULL,NULL,NULL,NULL,0);
    integrator.setAlgorithmSettings(5, 1e-16, true, 1);
    integrator.performIntegration(2.0, 100000);
    double y[1]; integrator.getSolutionVector(y);
    return 0;
}
```

### Forward/Reverse/Simultaneous sensitivities
See `examples/sensitivity_demo.cpp` for full usage of `forward()`, `reverse()`, and simultaneous seeds via `resetIntegration` with seed arrays.

### Pluggable linear solvers (sparse)
```cpp
#include <yabi/bdf_integrator.hpp>
#include <yabi/detail/umfpack_linear_solver.ipp>

yabi::BDFIntegrator<double> integrator(rhs);
integrator.setLinearSolver(new yabi::UmfpackLinearSolver<double>()); // takes ownership
```
Defaults to dense LAPACK. Supports UMFPACK/KLU via SuiteSparse. See `examples/heat1d_umfpack_literature.cpp`.

### Matrix-free Newton–Krylov (state solve)
```cpp
#include <yabi/bdf_integrator.hpp>
#include <yabi/matrix_free_solver.hpp>

auto *solver = new yabi::GmresLinearSolver<double>();
solver->setPreconditioner(yabi::KrylovPreconditioner::Jacobi); // default
solver->setTolerance(1e-12, 0.0, 200, 30);                     // rtol, atol, max it, restart
integrator.setLinearSolver(solver); // takes ownership; auto-enables matrix-free assembly
```
`FgmresLinearSolver` and `BicgstabLinearSolver` are drop-in alternatives.
The Krylov iteration drives the **state** Newton solve through the RHS
Jacobian–vector product (`AbstractRHS::jacobianVectorProduct`, one CppAD
`Forward(1)` sweep for AD tapes) and never sees a stored dense `J`. The scaled
CSC `P` is still assembled for the Newton scaling and the preconditioner
(`Jacobi` or the exact `Direct` solve). The path requires an RHS sparsity
pattern; without one the historical dense solve is used.

Sensitivities are deliberately unchanged: forward/reverse/simultaneous
sensitivity still uses the explicit `J`/`H` blocks and is not matrix-free.
`tests/component/matrix_free_newton.cpp` verifies all three Krylov methods on
HIRES against the direct trajectory and checks that the sensitivity result is
unchanged.

### Pure/manual ODE interface (no CppAD dependency in user code)
Users provide ODE as `std::function<void(double,const double*,int,double*)>` via `include/yabi/user_ode.hpp`. See `tests/system/manual_sensitivity_pure.cpp`.

## Tests
```sh
ctest --test-dir build --output-on-failure   # all tiers
ctest --test-dir build -L unit               # isolated component contracts
ctest --test-dir build -L component          # native sparse, matrix-free, SuiteSparse
ctest --test-dir build -L system             # mathematical benchmarks
./run_tests.sh                                # verbose full run
```

## Detailed usage

### 1) Providing the ODE (user function)
- **Via CppAD tape (AD):** Build `CppAD::ADFun<double>` from RHS `f(x)` (or `f(t,x)`) — see Usage example. CppAD optional.
- **Via analytical derivatives / manual interfaces:** Define RHS and provide analytic Jacobian/derivative routines through `include/yabi/user_ode.hpp`. See `examples/manual_sensitivity_pure.cpp` and `examples/manual_sensitivity_demo.cpp` (no CppAD tape required in pure form).

### 2) Analytical derivatives
- Supply analytic Jacobian `df/dy` and/or derivative RHS for sensitivity equations as appropriate for your formulation.
- Use the user-ODE interface; see `include/yabi/user_ode.hpp` for the API and `examples/manual_sensitivity_pure.cpp` for a complete example.

### 3) Pluggable linear solvers (dense + sparse)
**Interface:** `include/yabi/linear_solver.hpp` defines the abstract solver used by the corrector/linear solves. Implement `solve(A, b, x)` (and setup/factorization as needed).

**Built-in:**
- `DenseLUSolver<double>` (default, LAPACK `dgetrf`/`dgetrs`)
- `UmfpackLinearSolver<double>` (SuiteSparse UMFPACK)
- `KluLinearSolver<double>` (SuiteSparse KLU)
- `GmresLinearSolver<double>` / `FgmresLinearSolver<double>` / `BicgstabLinearSolver<double>`
  (matrix-free state Newton solve via `J·v`; see above)

**Connect your own solver (complete example):**
```cpp
#include <yabi/linear_solver.hpp>
#include <yabi/bdf_integrator.hpp>

class MySparseSolver : public yabi::LinearSolver<double> {
public:
  bool factorize(const yabi::SparseMatrix<double>& A) override { /* build symbolic/numeric */ return true; }
  bool solve(const yabi::SparseMatrix<double>& A, const double* b, double* x) override { /* solve */ return true; }
};

yabi::BDFIntegrator<double> integrator(rhs);
integrator.setLinearSolver(new MySparseSolver()); // integrator takes ownership
```
See `include/yabi/detail/dense_linear_solver.ipp`, `umfpack_linear_solver.ipp`, `klu_linear_solver.ipp` for reference implementations. Any sparse solver (MUMPS, SuperLU, Pardiso, Cholmod, etc.) can be plugged this way.

**Complete examples:** `examples/heat1d_umfpack_literature.cpp`, `examples/heat1d_dense_literature.cpp`, `examples/manual_sensitivity_pure.cpp`, `examples/manual_sensitivity_demo.cpp`.
