# yabi-torch 0.1.3

`pip install yabi-torch`

A batched, differentiable ODE solver for PyTorch — YABI's variable-order BDF
integrator + continuous adjoint, wrapped as a torch-native `solve_ivp`.

## What's new in 0.1.3

The solve is now differentiable in the model parameters too, so a neural ODE can
be *trained*: `loss.backward()` fills both `y0.grad` and every network `.grad`
through the continuous adjoint's augmented (λ, μ) state.

```python
import torch
from yabi_torch import solve_ivp

net = torch.nn.Sequential(nn.Linear(2, 8, dtype=torch.float64), nn.Tanh(),
                          nn.Linear(8, 2, dtype=torch.float64))
y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
loss = ((solve_ivp(net, y0, t_span=(0.0, 1.0)) - target) ** 2).sum()
loss.backward()   # fills every net parameter's .grad
```

Two core bugs fixed on the way (accuracy-first): the matrix-free adjoint used a
dummy zero Jacobian (it now supplies Jᵀλ directly), and the matrix-free operator
passed a full-domain direction where a state-sized `v` was expected.

## Why a BDF

Explicit solvers like `torchdiffeq`'s `dopri5` are the wrong tool for *stiff*
systems (chemical kinetics, circuits, biological networks). On Robertson, the
canonical stiff ODE, yabi-torch is **~25× faster at t=1 and ~380× at t=10**.
Reproduce with `benchmarks/torchdiffeq_comparison.py`.

- Automatic Jacobian via `torch.autograd`; `solver="dense"` or `"matrix_free"`.
- Batched `(batch, n)` + parallel `threads=N` (bit-identical to serial).
- GPU: `device="cuda"` offloads the Newton LU to cuSOLVER.

## Honest limits

Alpha; `float64` only; stiff systems only (on non-stiff problems an explicit RK
is the right tool); Linux x86_64 wheels (cp310–313) + sdist.
