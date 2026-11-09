"""YABI torch ODE solver — a batched, differentiable `solve_ivp`.

```python
import torch
from yabi_torch import solve_ivp

y0 = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float64, requires_grad=True)
yT = solve_ivp(f, y0, t_span=(0.0, 2.0), rtol=1e-8, atol=1e-10)   # automatic Jacobian
yT.backward(torch.ones_like(yT))   # y0.grad = d sum(yT) / d y0 via the adjoint
```

`f` is a `torch.nn.Module` (or any callable `y -> f(y)` on a float64 tensor).  The
Jacobian is **automatic** (via `torch.autograd.functional.jacobian`) unless you pass
`jac` (an analytic column-major Jacobian callable, `numpy (n,) -> (n*n,)`).  `y0` may
be `(n,)` or `(batch, n)`; the batch is solved trajectory-by-trajectory and the
gradient is the continuous adjoint.  The ODE is autonomous (`f(y)`, no `t`), so the
span is shifted to start at 0.
"""
import numpy as np
import torch
import yabi


def _wrap_rhs(f):
    """f: torch callable y -> f(y).  Return a numpy-in/out RHS for the binding."""
    def f_np(y_np):
        y_t = torch.as_tensor(y_np, dtype=torch.float64)
        return f(y_t).detach().numpy()
    return f_np


def _automatic_jacobian(f, n):
    """Return a numpy-in/out column-major Jacobian via torch.autograd."""
    def jac_np(y_np):
        y_t = torch.tensor(y_np, dtype=torch.float64, requires_grad=True)
        # enable_grad: inside a custom autograd.Function.forward grad mode is
        # off, so the derivative here must re-enable it explicitly.
        with torch.enable_grad():
            J = torch.autograd.functional.jacobian(f, y_t)  # (n, n), J[i,j] = df_i/dy_j
        return J.detach().numpy().T.reshape(-1)             # column-major (n*n)
    return jac_np


def _automatic_jvp(f):
    """Return a numpy-in/out Jacobian-vector product J·v via torch.autograd (one JVP).

    Uses forward-mode JVP (functional.jvp): torch.autograd.grad would give the
    vector-Jacobian product Jᵀv, which is the *transposed* operator and wrong for
    the matrix-free Newton (it needs J·v)."""
    def jv_np(y_np, v_np):
        y_t = torch.tensor(y_np, dtype=torch.float64, requires_grad=True)
        v_t = torch.tensor(v_np, dtype=torch.float64)
        # enable_grad: inside a custom autograd.Function.forward grad mode is
        # off, so the JVP must re-enable it explicitly.
        with torch.enable_grad():
            _, jvp = torch.autograd.functional.jvp(f, y_t, v_t)
        return jvp.detach().numpy()
    return jv_np


def solve_ivp(f, y0, t_span=(0.0, 1.0), rtol=1e-8, atol=1e-10, jac=None, solver="dense"):
    """Solve y' = f(y), y(t0) = y0 to y(t1), differentiable in y0.

    Args:
        f: torch callable y -> f(y) (a torch.nn.Module or function).
        y0: torch float64 tensor, shape (n,) or (batch, n), requires_grad.
        t_span: (t0, t1).  f is autonomous, so only t1 - t0 matters.
        rtol, atol: relative/absolute tolerances (scalars).
        jac: optional analytic Jacobian, numpy (n,) -> (n*n,) column-major.  If
            None, the Jacobian is computed automatically with torch.autograd.
        solver: "dense" (full Jacobian, dense Newton) or "matrix_free" (J·v via
            torch.autograd, GMRES Newton — no full Jacobian).

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

    f_np = _wrap_rhs(f)
    if solver == "matrix_free":
        jv_np = _automatic_jvp(f)

        def make():
            return yabi.matrix_free_integrator(f_np, jv_np, n)
    elif solver == "dense":
        jac_np = jac if jac is not None else _automatic_jacobian(f, n)

        def make():
            return yabi.BDFIntegrator(f_np, jac_np, n)
    else:
        raise ValueError("solver must be 'dense' or 'matrix_free'")

    class _Solve(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y0_batch):
            y0_np = y0_batch.detach().numpy()  # (batch, n)
            batch = y0_np.shape[0]
            atol_np = np.full(n, atol)
            integrators = []
            yT = np.empty((batch, n))
            for b in range(batch):
                it = make()
                assert it.reset_integration(y0_np[b], t_end, rtol, atol_np)
                it.set_algorithm_settings(5, np.finfo(np.float64).eps, True, 1)
                assert it.integrate(t_end)
                yb = np.zeros(n)
                assert it.get_solution(yb)
                yT[b] = yb
                integrators.append(it)
            ctx.integrators = integrators
            ctx.n = n
            return torch.as_tensor(yT, dtype=y0_batch.dtype)

        @staticmethod
        def backward(ctx, grad_yT):
            grad_np = grad_yT.detach().numpy()  # (batch, n)
            adj = np.empty((grad_np.shape[0], ctx.n))
            for b in range(grad_np.shape[0]):
                adj_b = np.zeros(ctx.n)
                assert ctx.integrators[b].continuous_adjoint(grad_np[b], adj_b)
                adj[b] = adj_b
            return torch.as_tensor(adj, dtype=grad_yT.dtype)

    single = y0.dim() == 1
    y0_batch = y0 if not single else y0.unsqueeze(0)
    out = _Solve.apply(y0_batch)  # (batch, n)
    return out[0] if single else out
