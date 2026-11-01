BDF Integrator Testbench - Canonical Tests
==========================================

This testbench implements canonical tests for BDF integrators including:
- Linear test problems
- Stiff test problems from literature (Hairer & Wanner test suite style)
- Sensitivity analysis tests (forward, reverse/adjoint, simultaneous)
- Failure-path regression (invalid error weight / singular Newton matrix)
- Tolerance adherence: global error must track the requested `rtol` across
  eight decades, and tightening `rtol` must actually improve the result
- First-order step convergence with `MAXORDER = 1`: halving `max_StepLength`
  halves the global error (observed slope 0.96)
- Mass conservation checks where applicable

What is deliberately NOT claimed: a per-order convergence test for orders 2-6.
`setAlgorithmSettings(MAXORDER, ...)` only caps the order; the controller still
chooses it, and the chosen order is not exposed by the public API. `MAXORDER=1`
is the only order that is genuinely pinned, so it is the only slope asserted
(test `11_tolerance_and_order.cpp`). Error versus `max_StepLength` is therefore
not monotone in general: at `rtol = 1e-4` the relative error measured over
`t <= 3` was 2.2e-4 at `max_step = 0.05`, 2.0e-6 at 0.025, and 2.3e-3 at 0.0125.
All of those stay inside the requested error envelope; the wobble is the order
controller, not a loss of convergence.

Run with: make testbench (or compile individually)

The three examples are also self-checking and are registered in ctest as
`example_basic`, `example_nonautonomous` and `example_sensitivity_demo`.

Includes canonical tests (Brusselator, Robertson, HIRES, Van der Pol, Oregonator, Prothero–Robinson, damped pendulum, E5, etc.) plus sensitivities.

## Literature references
- Hairer E., Wanner G. (1996), Solving Ordinary Differential Equations II, Springer. (Brusselator, Robertson, HIRES, Van der Pol, Oregonator, E5, Prothero–Robinson, etc.)
- Enright W.H., Hull T.E. (1976), Comparing numerical methods for the solutions of systems of ODEs. BIT 15, 10–48.
- SUNDIALS/CVODE test suite (Roberson, Van der Pol, etc.)
- Shampine L.F., Reichelt M.W. (1997), The MATLAB ODE suite.
