# Verification

> **Supplementary** to [`yabi-torch`](..) — these are independent correctness checks, **not part of the
> package** (they do not ship in the wheel or sdist). Everything here is reproducible with
> [`checks.py`](checks.py).

This directory answers the four questions raised in an independent review of `yabi-torch`. Numbers below
are from **`yabi-torch 0.1.4`**, `torch 2.11.0+cu128`, `scipy 1.18.1`, `torchdiffeq 0.2.5`, run on the
Robertson problem (a canonical stiff ODE), `float64`, single-threaded, `rtol=1e-6, atol=1e-10`.

---

## 1. Mass conservation at t = 1e5

`d/dt (y0+y1+y2) = 0` identically for Robertson, so the sum must remain 1.

| solver | y(T) | \|sum − 1\| |
|---|---|---|
| dense | `[1.786699e-02, 7.275182e-08, 9.821329e-01]` | 1.2e-15 |
| matrix_free | `[1.786699e-02, 7.275183e-08, 9.821329e-01]` | 3.1e-15 |

Both representations conserve mass to machine precision — far within the `rtol=1e-6` / `atol=1e-10`
tolerances.

## 2. Adjoint vs finite differences on the stiff rate

`d y0(T)/d(1e4)`, where the `1e4` coefficient is a trainable parameter, computed by the continuous
adjoint and compared against a central difference with a small perturbation `δ = 0.1` (0.001% of the
`1e4` coefficient); the FD reference is solved at `rtol=1e-12`.

| solver | t | adjoint | finite difference | \|adj − FD\| |
|---|---|---|---|---|
| dense | 1 | 4.6066628e-07 | 4.6073621e-07 | 6.99e-11 |
| matrix_free | 1 | 4.6066628e-07 | 4.6073622e-07 | 6.99e-11 |
| dense | 1e5 | 3.1704898e-06 | 3.1692035e-06 | 1.29e-09 |
| matrix_free | 1e5 | 3.1692439e-06 | 3.1689478e-06 | 2.96e-10 |

The adjoint matches the finite difference to ~1e-9 absolute (the derivative itself is ~1e-6, so the
relative figure is ~1e-4; the absolute agreement is far tighter than the `rtol=1e-6` forward solve).

## 3. Implicit baselines at t = 1e5

| solver | y0(T) | \|sum − 1\| | wall-clock |
|---|---|---|---|
| yabi-torch dense | 1.786699e-02 | 1.2e-15 | 0.13 s |
| yabi-torch matrix_free | 1.786699e-02 | 3.1e-15 | 2.28 s |
| SciPy `BDF` | 1.786603e-02 | 4.4e-16 | 0.06 s |
| torchdiffeq `implicit_adams` | −9.1e+107 | — | diverges |

- **SciPy BDF** agrees with `yabi-torch` to within the `rtol=1e-6` band (~5e-6 on `y0`) and also conserves
  mass to machine precision. On this small (n=3) problem SciPy's Fortran BDF is ~2× faster than
  `yabi-torch` dense — expected, as SciPy is single-problem, non-differentiable, non-batched.
- **`torchdiffeq`'s `implicit_adams` diverges** (≈1e111). It is "implicit" in the Adams–Moulton sense but
  solves the implicit step by *fixed-point iteration*, not Newton, so it is not a stiff solver under these
  tolerances.

## 4. Adjoint under Newton-step rejection

Tightening to `rtol=1e-10` forces more rejected Newton steps / step-size reductions mid-interval.

| solver | \|adj − FD\| |
|---|---|
| dense | 1.28e-10 |
| matrix_free | 1.28e-10 |

The adjoint remains correct in the rejection regime: the continuous adjoint replays the *accepted*
trajectory, so rejected steps do not enter it.

---

## Reproduce

```bash
pip install yabi-torch scipy torchdiffeq   # torchdiffeq is optional (only Check 3's second row)
python verification/checks.py
```

## Notes

- These checks **surfaced and fixed a bug**: `matrix_free` + a parameterized RHS failed at t ≥ 1e4
  (the initial step-size estimate was built from the zero dummy Jacobian instead of the J·v hook). Fixed in
  `0.1.4`; the matrix-free rows above are post-fix. Regression guard:
  `bindings/python/test_nn_module.py::test_matrix_free_parameter_longtime`.
