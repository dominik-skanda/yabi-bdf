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


if __name__ == "__main__":
    test_linear_dense_matches_expm()
    test_linear_matrix_free_matches_dense()
    test_linear_analytic_matches_automatic()
    test_jvp_is_autograd()
    test_nn_module_gradcheck()
    test_mlp_modes_agree()
    test_mlp_gradcheck()
    print("PASS: nn.Module f — automatic Jacobian (dense + matrix-free), adjoint gradcheck")
