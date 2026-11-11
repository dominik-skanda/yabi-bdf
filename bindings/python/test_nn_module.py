"""T52 — solve_ivp with a torch.nn.Module f (no hand-written Jacobian).

The Jacobian is automatic (torch.autograd): the dense path forms the full n×n
Jacobian, the matrix-free path forms only J·v (one JVP per Krylov iteration).
References: the exact matrix exponential for a linear module, and cross-mode
agreement for a nonlinear MLP.
"""
import numpy as np
import torch
import torch.nn as nn

import yabi_torch

# ---- linear module: y' = A y, reference y(T) = expm(A T) y0 ----
A = torch.tensor([[0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0],
                  [-2.0, -3.0, -1.0]], dtype=torch.float64)  # stable, well-conditioned


class LinearODE(nn.Module):
    def __init__(self, A):
        super().__init__()
        self.register_buffer("A", A)

    def forward(self, y):
        return self.A @ y


def linear_jac(y_np):  # analytic column-major Jacobian of LinearODE
    return A.T.detach().numpy().reshape(-1).copy()


def test_linear_dense_matches_expm():
    m = LinearODE(A)
    y0 = torch.tensor([1.0, 0.5, -0.25], dtype=torch.float64)
    T = 2.0
    yT = yabi_torch.solve_ivp(m, y0, t_span=(0.0, T))
    ref = torch.linalg.matrix_exp(A * T) @ y0
    assert torch.allclose(yT, ref, atol=1e-5, rtol=1e-5), f"{yT} vs {ref}"
    print("linear dense == expm: OK")


def test_linear_matrix_free_matches_dense():
    m = LinearODE(A)
    y0 = torch.tensor([1.0, 0.5, -0.25], dtype=torch.float64)
    T = 2.0
    dense = yabi_torch.solve_ivp(m, y0, t_span=(0.0, T))
    mfree = yabi_torch.solve_ivp(m, y0, t_span=(0.0, T), solver="matrix_free")
    ref = torch.linalg.matrix_exp(A * T) @ y0
    assert torch.allclose(dense, ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(mfree, ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(dense, mfree, atol=1e-6, rtol=1e-6)
    print("linear matrix-free == dense == expm: OK")


def test_linear_analytic_matches_automatic():
    m = LinearODE(A)
    y0 = torch.tensor([1.0, 0.5, -0.25], dtype=torch.float64)
    auto = yabi_torch.solve_ivp(m, y0, t_span=(0.0, 2.0))
    anal = yabi_torch.solve_ivp(m, y0, t_span=(0.0, 2.0), jac=linear_jac)
    assert torch.allclose(auto, anal, atol=1e-10)
    print("linear analytic == automatic: OK")


def test_jvp_is_autograd():
    m = LinearODE(A)
    jv = yabi_torch._automatic_jvp(m)
    y = np.array([1.0, 0.5, -0.25])
    v = np.array([0.3, -1.0, 0.7])
    got = jv(y, v)
    want = (A @ torch.tensor(v, dtype=torch.float64)).detach().numpy()
    assert np.allclose(got, want, atol=1e-12), f"J·v {got} vs A·v {want}"
    print("J·v == A·v (autograd): OK")


def test_nn_module_gradcheck():
    m = LinearODE(A)
    y0 = torch.tensor([1.0, 0.5, -0.25], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda y: yabi_torch.solve_ivp(m, y, t_span=(0.0, 2.0)), (y0,))
    print("nn.Module gradcheck (dense): OK")


# ---- nonlinear MLP: y' = tanh(W y + b); cross-mode agreement + gradcheck ----
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 8, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(8, 3, dtype=torch.float64),
        )

    def forward(self, y):
        return self.net(y)


def test_mlp_modes_agree():
    torch.manual_seed(0)
    m = MLP()
    y0 = torch.tensor([0.3, -0.1, 0.2], dtype=torch.float64)
    dense = yabi_torch.solve_ivp(m, y0, t_span=(0.0, 1.0))
    mfree = yabi_torch.solve_ivp(m, y0, t_span=(0.0, 1.0), solver="matrix_free")
    assert torch.allclose(dense, mfree, atol=1e-5, rtol=1e-5), f"{dense} vs {mfree}"
    print("MLP matrix-free == dense: OK")


def test_mlp_gradcheck():
    torch.manual_seed(0)
    m = MLP()
    y0 = torch.tensor([0.3, -0.1, 0.2], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda y: yabi_torch.solve_ivp(m, y, t_span=(0.0, 1.0)), (y0,))
    print("MLP gradcheck (dense): OK")


def _fd_param_grad(m, y0, solver, eps=1e-6):
    """Central-difference gradient of sum(solve_ivp(m, y0)) w.r.t. each parameter."""
    def loss(mm):
        return yabi_torch.solve_ivp(mm, y0, t_span=(0.0, 1.0), solver=solver).sum()

    grads = []
    for p in m.parameters():
        fd = torch.zeros_like(p)
        flat = p.detach().flatten().clone()
        for i in range(flat.numel()):
            o = flat[i].item()
            flat[i] = o + eps
            p.data.copy_(flat.reshape(p.shape)); lp = loss(m).item()
            flat[i] = o - eps
            p.data.copy_(flat.reshape(p.shape)); lm = loss(m).item()
            flat[i] = o
            p.data.copy_(flat.reshape(p.shape))
            fd.view(-1)[i] = (lp - lm) / (2 * eps)
        grads.append(fd)
    return grads


def test_parameter_gradient():
    # The parameter adjoint (d loss / d theta) must match central differences,
    # for a *fixed* y0 (no requires_grad) — the neural-ODE training case.
    torch.manual_seed(0)
    m = torch.nn.Sequential(nn.Linear(2, 3, dtype=torch.float64), nn.Tanh(),
                            nn.Linear(3, 2, dtype=torch.float64))
    y0 = torch.tensor([0.7, -0.4], dtype=torch.float64)
    for solver in ("dense", "matrix_free"):
        for p in m.parameters():
            p.grad = None
        yabi_torch.solve_ivp(m, y0, t_span=(0.0, 1.0), solver=solver).sum().backward()
        adj = [p.grad.clone() for p in m.parameters()]
        fd = _fd_param_grad(m, y0, solver)
        for a, f in zip(adj, fd):
            err = float((a - f).abs().max())
            assert err < 1e-4, f"{solver}: adjoint-FD max error {err:.3e}"
        print(f"parameter gradient ({solver}) == FD: OK")


def test_neural_ode_training():
    # End-to-end: train an MLP to reproduce a linear oscillator, one SGD step.
    torch.manual_seed(1)
    A = torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64)
    m = torch.nn.Sequential(nn.Linear(2, 8, dtype=torch.float64), nn.Tanh(),
                            nn.Linear(8, 2, dtype=torch.float64))
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    y0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    target = torch.linalg.matrix_exp(A * 1.0) @ y0

    def loss():
        return ((yabi_torch.solve_ivp(m, y0, t_span=(0.0, 1.0)) - target) ** 2).sum()

    l0 = loss().item()
    opt.zero_grad()
    loss().backward()
    opt.step()
    l1 = loss().item()
    assert l1 < l0, f"loss did not decrease: {l0:.4e} -> {l1:.4e}"
    print(f"neural ODE training: loss {l0:.4e} -> {l1:.4e} (one SGD step)")


def test_matrix_free_parameter_longtime():
    # T58 regression guard: matrix-free + a parameterized RHS must integrate the
    # stiff Robertson problem to t=1e5 (it used to fail with "too much newton
    # steps rejected" because the matrix-free Forward(1,·) built the Jacobian
    # from the zero dummy Jacobian, corrupting the initial step-size estimate).
    class Robertson(nn.Module):
        def __init__(self, k=1e4):
            super().__init__()
            self.k = nn.Parameter(torch.tensor(float(k), dtype=torch.float64))

        def forward(self, y):
            return torch.stack([-0.04 * y[0] + self.k * y[1] * y[2],
                                0.04 * y[0] - self.k * y[1] * y[2] - 3e7 * y[1] ** 2,
                                3e7 * y[1] ** 2])

    m = Robertson()
    y0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    dense = yabi_torch.solve_ivp(m, y0, t_span=(0.0, 1e5), rtol=1e-6, atol=1e-10)
    mfree = yabi_torch.solve_ivp(m, y0, t_span=(0.0, 1e5), rtol=1e-6, atol=1e-10,
                                 solver="matrix_free")
    assert torch.allclose(dense, mfree, atol=1e-6, rtol=1e-6), f"{dense} vs {mfree}"
    # mass conservation still holds to machine precision
    assert abs(mfree.sum().item() - 1.0) < 1e-12
    print("matrix-free + parameterized RHS at t=1e5 == dense: OK")


if __name__ == "__main__":
    test_linear_dense_matches_expm()
    test_linear_matrix_free_matches_dense()
    test_linear_analytic_matches_automatic()
    test_jvp_is_autograd()
    test_nn_module_gradcheck()
    test_mlp_modes_agree()
    test_mlp_gradcheck()
    test_parameter_gradient()
    test_neural_ode_training()
    test_matrix_free_parameter_longtime()
    print("PASS: nn.Module f — automatic Jacobian (dense + matrix-free), y0 + parameter adjoint")
