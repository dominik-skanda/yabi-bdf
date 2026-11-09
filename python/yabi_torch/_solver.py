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

from concurrent.futures import ThreadPoolExecutor


def _wrap_rhs(f, device="cpu"):
    """f: torch callable y -> f(y).  Return a numpy-in/out RHS for the binding.

    With device="cuda" the state is copied to the GPU per evaluation and f runs
    on-device; the result is copied back to the host array."""
    def f_np(y_np):
        y_t = torch.as_tensor(y_np, dtype=torch.float64, device=device)
        return f(y_t).detach().cpu().numpy()
    return f_np


def _automatic_jacobian(f, n, device="cpu"):
    """Return a numpy-in/out column-major Jacobian via torch.autograd."""
    def jac_np(y_np):
        y_t = torch.tensor(y_np, dtype=torch.float64, requires_grad=True, device=device)
        # enable_grad: inside a custom autograd.Function.forward grad mode is
        # off, so the derivative here must re-enable it explicitly.
        with torch.enable_grad():
            J = torch.autograd.functional.jacobian(f, y_t)  # (n, n), J[i,j] = df_i/dy_j
        return J.detach().cpu().numpy().T.reshape(-1)       # column-major (n*n)
    return jac_np


def _automatic_jvp(f, device="cpu"):
    """Return a numpy-in/out Jacobian-vector product J·v via torch.autograd (one JVP).

    Uses forward-mode JVP (functional.jvp): torch.autograd.grad would give the
    vector-Jacobian product Jᵀv, which is the *transposed* operator and wrong for
    the matrix-free Newton (it needs J·v)."""
    def jv_np(y_np, v_np):
        y_t = torch.tensor(y_np, dtype=torch.float64, requires_grad=True, device=device)
        v_t = torch.tensor(v_np, dtype=torch.float64, device=device)
        # enable_grad: inside a custom autograd.Function.forward grad mode is
        # off, so the JVP must re-enable it explicitly.
        with torch.enable_grad():
            _, jvp = torch.autograd.functional.jvp(f, y_t, v_t)
        return jvp.detach().cpu().numpy()
    return jv_np


def solve_ivp(f, y0, t_span=(0.0, 1.0), rtol=1e-8, atol=1e-10, jac=None, solver="dense",
              threads=1, device="cpu"):
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
        threads: number of trajectories solved concurrently (default 1 = serial).
            Independent trajectories are solved on a thread pool; the result is
            bit-identical to serial.
        device: "cpu" (default) or "cuda".  "cuda" evaluates f and the Jacobian/JVP
            on the GPU (f must already be on the device) and, for the dense path,
            offloads the Newton LU to cuSOLVER (requires a CUDA build of the yabi
            module).  The state and result stay host-side.

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
    threads = max(1, int(threads))
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but torch CUDA is not available")
        # The f-on-GPU path needs torch CUDA kernels; torch's cu12x wheels dropped
        # pre-Turing GPUs (min sm_75, CC 7.5), so fail clearly rather than with the
        # opaque "no kernel image" runtime error.
        cc = torch.cuda.get_device_capability(0)
        if (cc[0], cc[1]) < (7, 5):
            raise RuntimeError(
                f"device='cuda' needs a GPU with compute capability >= 7.5 "
                f"(this device is {cc[0]}.{cc[1]}); the f-on-GPU path is unsupported "
                f"on this GPU.  The cuSOLVER LU offload alone is available via "
                f"yabi.cuda_integrator.")
        if solver == "dense" and not hasattr(yabi, "cuda_integrator"):
            raise RuntimeError(
                "device='cuda' (dense) requires the yabi module to be built with CUDA "
                "(YABI_USE_CUDA=ON)")

    f_np = _wrap_rhs(f, device)
    if solver == "matrix_free":
        jv_np = _automatic_jvp(f, device)

        def make():
            return yabi.matrix_free_integrator(f_np, jv_np, n)
    elif solver == "dense":
        jac_np = jac if jac is not None else _automatic_jacobian(f, n, device)
        if device == "cuda":
            def make():
                return yabi.cuda_integrator(f_np, jac_np, n)
        else:
            def make():
                return yabi.BDFIntegrator(f_np, jac_np, n)
    else:
        raise ValueError("solver must be 'dense' or 'matrix_free'")

    class _Solve(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y0_batch):
            y0_np = y0_batch.detach().numpy()  # (batch, n)
            batch = y0_np.shape[0]

            def solve_one(b):
                it = make()
                assert it.reset_integration(y0_np[b], t_end, rtol, np.full(n, atol))
                it.set_algorithm_settings(5, np.finfo(np.float64).eps, True, 1)
                assert it.integrate(t_end)
                yb = np.zeros(n)
                assert it.get_solution(yb)
                return it, yb

            if threads > 1 and batch > 1:
                with ThreadPoolExecutor(max_workers=threads) as ex:
                    results = list(ex.map(solve_one, range(batch)))
                integrators = [r[0] for r in results]
                yT = np.stack([r[1] for r in results])
            else:
                integrators, rows = [], []
                for b in range(batch):
                    it, yb = solve_one(b)
                    integrators.append(it)
                    rows.append(yb)
                yT = np.stack(rows)

            ctx.integrators = integrators
            ctx.n = n
            ctx.threads = threads
            return torch.as_tensor(yT, dtype=y0_batch.dtype)

        @staticmethod
        def backward(ctx, grad_yT):
            # Copy (not view): grad_yT.detach().numpy() shares memory with the
            # torch gradient tensor, and the adjoint's nested autograd can reuse
            # that buffer while it is still being read as the seed, corrupting
            # the batched gradient (single-trajectory happens not to trigger it).
            grad_np = np.array(grad_yT.detach().numpy())  # (batch, n)
            batch = grad_np.shape[0]

            def adjoint_one(b):
                adj_b = np.zeros(ctx.n)
                assert ctx.integrators[b].continuous_adjoint(grad_np[b], adj_b)
                return adj_b

            if ctx.threads > 1 and batch > 1:
                with ThreadPoolExecutor(max_workers=ctx.threads) as ex:
                    adj_rows = list(ex.map(adjoint_one, range(batch)))
                adj = np.stack(adj_rows)
            else:
                adj = np.stack([adjoint_one(b) for b in range(batch)])
            return torch.as_tensor(adj, dtype=grad_yT.dtype)

    single = y0.dim() == 1
    y0_batch = y0 if not single else y0.unsqueeze(0)
    out = _Solve.apply(y0_batch)  # (batch, n)
    return out[0] if single else out
