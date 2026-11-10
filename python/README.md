# yabi-torch

A batched, **differentiable** ODE solver for PyTorch — YABI's variable-order BDF integrator with the
continuous adjoint, wrapped as a torch-native `solve_ivp`.

```bash
pip install yabi-torch
```

```python
import torch
from yabi_torch import solve_ivp

# Robertson — a classic stiff ODE (mass-conserving chemical kinetics)
def f(y):
    return torch.stack([
        -0.04 * y[0] + 1e4 * y[1] * y[2],
         0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2,
         3e7 * y[1] ** 2,
    ])

y0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
yT = solve_ivp(f, y0, t_span=(0.0, 1e5), rtol=1e-6, atol=1e-10)

yT.sum().backward()          # y0.grad = d(sum y(T)) / d y0, via the continuous adjoint
print(yT)                    # [0.0179, 7.3e-8, 0.9821]  (mass conserved)
```

## What it is

`yabi_torch.solve_ivp` solves the autonomous initial-value problem `y' = f(y)`, `y(t0) = y0` with YABI's
**BDF integrator** (orders 1–6, variable step, Newton iteration) — the kind of solver that handles *stiff*
systems where explicit methods like `torchdiffeq`'s RK need impractically small steps. It is differentiable
in `y0` **and in the model parameters `θ`** through the **continuous adjoint**, so a neural ODE can be
*trained* (`loss.backward()` fills both `y0.grad` and the network's `.grad`).

- `f` is a `torch.nn.Module` or any callable `y -> f(y)` on a `float64` tensor — **no hand-written Jacobian**
  (it is computed automatically with `torch.autograd`).
- `y0` may be a single `(n,)` or a batch `(batch, n)`; the gradient is the adjoint per trajectory.

## API

```python
solve_ivp(f, y0, t_span=(0.0, 1.0), rtol=1e-8, atol=1e-10,
          jac=None, solver="dense", threads=1, device="cpu")
```

| option | meaning |
|---|---|
| `f` | the vector field `y -> f(y)` (`torch.nn.Module` or callable) |
| `y0` | `(n,)` or `(batch, n)` `torch.float64` tensor |
| `t_span` | `(t0, t1)` — `f` is autonomous, so only `t1 - t0` matters |
| `jac` | optional analytic Jacobian, `numpy (n,) -> (n*n,)` column-major (else automatic) |
| `solver` | `"dense"` (full Jacobian, dense Newton) or `"matrix_free"` (J·v via `torch.autograd`, GMRES) |
| `threads` | trajectories solved concurrently (default 1 = serial; bit-identical to serial) |
| `device` | `"cpu"` (default) or `"cuda"` (f/Jacobian on GPU + cuSOLVER Newton; needs a CUDA-enabled build) |

## Features

- **Stiff-capable** — BDF + Newton, not an explicit RK.
- **Automatic Jacobian** — `torch.autograd`; pass `jac` only if you want the analytic path.
- **Dense and matrix-free** — both Jacobian representations behind one API.
- **Batched + differentiable** — solve a batch and backprop through it.
- **Parallel batch** — `threads=N` is bit-identical to serial.
- **GPU** — `device="cuda"` offloads the dense Newton LU to cuSOLVER and (on a GPU with compute capability ≥ 7.5)
  evaluates `f` on-device.  **This requires a CUDA-enabled build (`YABI_USE_CUDA=ON`)** — the PyPI wheels are
  CPU-only, so `device="cuda"` needs a source build. A CUDA wheel is planned.

## Requirements

- Python ≥ 3.10, NumPy, PyTorch ≥ 2.0.
- Installing from source (the sdist) needs a C++17 compiler, CMake ≥ 3.16, and BLAS/LAPACK.

## License

Apache-2.0. The integrator core is [YABI](https://github.com/dominik-skanda/yabi-bdf).
