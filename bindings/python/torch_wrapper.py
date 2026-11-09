"""YABI inside `torch.autograd.Function` — differentiate through an ODE solve
w.r.t. the initial condition via YABI's continuous adjoint.

Usage:

    solve = make_ode_solver(f, jac, n)          # f(y)->f, jac(y)->J (column-major)
    y0 = torch.tensor([...], requires_grad=True, dtype=torch.float64)
    yT = solve(y0, t_end=2.0, rtol=1e-10)       # forward solve
    loss = (yT * w).sum()                        # scalar objective
    loss.backward()                              # fills y0.grad = d loss / d y0

Run `python torch_wrapper.py` (with torch installed) to execute the gradient
check below against `torch.autograd.gradcheck`.
"""
import numpy as np
import yabi


def make_ode_solver(f, jac, n):
    """Return a callable that solves y' = f(y), y(0) = y0 to y(t_end), with a
    gradient w.r.t. y0 computed by YABI's continuous adjoint."""
    # torch is imported lazily so this file stays importable without torch.
    import torch

    class _YabiOdeSolve(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y0, t_end, rtol):
            y0_np = y0.detach().numpy()
            atol = np.full(n, rtol)
            it = yabi.BDFIntegrator(f, jac, n)
            assert it.reset_integration(y0_np, float(t_end), float(rtol), atol)
            it.set_algorithm_settings(5, np.finfo(np.float64).eps, True, 1)
            assert it.integrate(float(t_end))
            y = np.zeros(n)
            assert it.get_solution(y)
            ctx.it = it  # keep the integrator (and its recorded trajectory) alive
            ctx.n = n
            return torch.as_tensor(y, dtype=y0.dtype)

        @staticmethod
        def backward(ctx, grad_yT):
            grad_np = grad_yT.detach().numpy()
            adj = np.zeros(ctx.n)
            assert ctx.it.continuous_adjoint(grad_np, adj)
            # grad_y0; t_end and rtol are non-differentiable inputs.
            return torch.as_tensor(adj, dtype=grad_yT.dtype), None, None

    return _YabiOdeSolve.apply


if __name__ == "__main__":
    import torch

    # Decay chain: y0' = -k0 y0,  yi' = k_{i-1} y_{i-1} - ki yi.
    k = np.array([0.1, 0.2, 0.3, 0.4])
    n = 4

    def f(y):
        out = np.zeros(n)
        out[0] = -k[0] * y[0]
        for i in range(1, n):
            out[i] = k[i - 1] * y[i - 1] - k[i] * y[i]
        return out

    def jac(y):
        J = np.zeros(n * n)
        for i in range(n):
            J[i + n * i] = -k[i]
        for i in range(1, n):
            J[i + n * (i - 1)] = k[i - 1]
        return J

    solve = make_ode_solver(f, jac, n)

    def wrapped(y0):
        return solve(y0, torch.as_tensor(2.0), torch.as_tensor(1e-10))

    y0 = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float64, requires_grad=True)
    print("torch.autograd.gradcheck:", torch.autograd.gradcheck(wrapped, (y0,)))
