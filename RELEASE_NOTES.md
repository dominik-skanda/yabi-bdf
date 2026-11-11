# yabi-torch 0.1.4

`pip install yabi-torch` (CPU). GPU: `pip install <the 0.1.4+cu130 wheel from this release's assets>`.

A batched, differentiable ODE solver for PyTorch — YABI's variable-order BDF
integrator + continuous adjoint, wrapped as a torch-native `solve_ivp`.

## What's new in 0.1.4

- **Fixed a matrix-free bug (T58):** `matrix_free` + a parameterized `nn.Module` RHS now integrates
  stiff problems to t=1e5 (it previously failed with "too much newton steps rejected" because the
  initial step-size estimate was built from the zero dummy Jacobian instead of the J·v hook).
  Regression-guarded by `test_nn_module.py::test_matrix_free_parameter_longtime`.

## Independent review (T57) — measured

- **Mass conservation** on Robertson at t=1e5 is machine-precision (`|sum−1| ≈ 1e-15`, dense + matrix-free).
- **Stiff-rate adjoint** `d y0/d(1e4)` matches central differences to ~1e-9 absolute (dense + matrix-free).
- **Implicit baseline:** SciPy `BDF` agrees to tolerance (conserves to 4e-16); `torchdiffeq`'s
  `implicit_adams` diverges to 1e111 (it uses fixed-point iteration, not Newton — not a stiff solver).
- **Adjoint under Newton-step rejection** (`rtol=1e-10`) still matches finite differences (~1e-10).

## GPU (CUDA 13.0)

The CUDA wheels are published as **release assets** (each ~743 MB — too large for PyPI's ~100 MB
per-file limit). They bundle the CUDA 13.0 runtime (cuSOLVER/cuBLAS/cuSPARSE/cudart/nvJitLink). Install:

```bash
pip install https://github.com/dominik-skanda/yabi-bdf/releases/download/v0.1.4/yabi_torch-0.1.4+cu130-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
```

(adjust `cp310` to your Python; needs a CUDA 13 driver and a Turing (sm_75) or newer GPU).

## Honest limits

Alpha; `float64` only; stiff systems only (on non-stiff problems an explicit RK is the right tool);
Linux x86_64 wheels (cp310–313) + sdist.
