"""YABI torch ODE solver — a batched, differentiable `solve_ivp`.

```python
import torch
from yabi_torch import solve_ivp

y0 = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float64, requires_grad=True)
yT = solve_ivp(f, jac, y0, t_span=(0.0, 2.0), rtol=1e-8, atol=1e-10)
yT.backward(torch.ones_like(yT))   # y0.grad = d sum(yT) / d y0 via the adjoint
```

`y0` may be `(n,)` or `(batch, n)`; the batch is solved trajectory-by-trajectory
(each an independent `BDFIntegrator`) and the gradient is the continuous adjoint.
The ODE is autonomous (`f(y)`, no `t`), so the span is shifted to start at 0.
"""
import numpy as np
import torch
import yabi


def solve_ivp(f, jac, y0, t_span=(0.0, 1.0), rtol=1e-8, atol=1e-10):
    """Solve y' = f(y), y(t0) = y0 to y(t1), differentiable in y0.

    Args:
        f: callable y -> RHS (length-n array-like).
        jac: callable y -> column-major Jacobian (length-n*n array-like).
        y0: torch float64 tensor, shape (n,) or (batch, n), requires_grad.
        t_span: (t0, t1).  f is autonomous, so only t1 - t0 matters.
        rtol, atol: relative/absolute tolerances (scalars).

    Returns:
        y(t1) as a torch float64 tensor with the same batch shape, carrying a
        grad_fn whose backward is the continuous adjoint.
    """
    t0, t1 = float(t_span[0]), float(t_span[1])
    t_end = t1 - t0
    if t_end <= 0:
        raise ValueError("t_span must be increasing")
    n = y0.shape[-1]
    if y0.dtype != torch.float64:
        raise TypeError("y0 must be torch.float64 (YABI is double precision)")

    class _Solve(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y0):
            y0_np = y0.detach().numpy()
            single = y0_np.ndim == 1
            if single:
                y0_np = y0_np[None, :]
            batch = y0_np.shape[0]
            atol_np = np.full(n, atol)
            integrators = []
            yT = np.empty((batch, n))
            for b in range(batch):
                it = yabi.BDFIntegrator(f, jac, n)
                assert it.reset_integration(y0_np[b], t_end, rtol, atol_np)
                it.set_algorithm_settings(5, np.finfo(np.float64).eps, True, 1)
                assert it.integrate(t_end)
                yb = np.zeros(n)
                assert it.get_solution(yb)
                yT[b] = yb
                integrators.append(it)
            ctx.integrators = integrators
            ctx.n = n
            out = torch.as_tensor(yT, dtype=y0.dtype)
            return out[0] if single else out

        @staticmethod
        def backward(ctx, grad_yT):
            grad_np = grad_yT.detach().numpy()
            single = grad_np.ndim == 1
            if single:
                grad_np = grad_np[None, :]
            adj = np.empty((grad_np.shape[0], ctx.n))
            for b in range(grad_np.shape[0]):
                adj_b = np.zeros(ctx.n)
                assert ctx.integrators[b].continuous_adjoint(grad_np[b], adj_b)
                adj[b] = adj_b
            out = torch.as_tensor(adj, dtype=grad_yT.dtype)
            return out[0] if single else out

    return _Solve.apply(y0)
