"""T51/T52 — batched differentiable solve_ivp: gradcheck, batch, stiff, analytic
vs automatic Jacobian, and dense vs matrix-free."""
import numpy as np
import torch

import yabi_torch

# ---- decay chain (n=4) ----
k = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)


def decay_f(y):  # torch (4,) -> (4,) — functional (no in-place, so autograd works)
    neg = -k * y                                   # -k_i * y_i
    pos = torch.cat([torch.zeros(1, dtype=torch.float64), k[:-1] * y[:-1]])  # k_{i-1} * y_{i-1}
    return neg + pos


def decay_jac(y_np):  # numpy (4,) -> (16,) column-major
    J = np.zeros(16)
    for i in range(4):
        J[i + 4 * i] = -k[i].item()
    for i in range(1, 4):
        J[i + 4 * (i - 1)] = k[i - 1].item()
    return J


def test_gradcheck():
    y0 = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda y: yabi_torch.solve_ivp(decay_f, y, t_span=(0.0, 2.0)), (y0,))
    print("gradcheck (dense, automatic): OK")


def test_analytic_matches_automatic():
    y0 = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float64)
    auto = yabi_torch.solve_ivp(decay_f, y0, t_span=(0.0, 2.0))
    anal = yabi_torch.solve_ivp(decay_f, y0, t_span=(0.0, 2.0), jac=decay_jac)
    assert torch.allclose(auto, anal, atol=1e-8)
    print("analytic == automatic Jacobian: OK")


def test_matrix_free_matches_dense():
    y0 = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float64)
    dense = yabi_torch.solve_ivp(decay_f, y0, t_span=(0.0, 2.0))
    mfree = yabi_torch.solve_ivp(decay_f, y0, t_span=(0.0, 2.0), solver="matrix_free")
    assert torch.allclose(dense, mfree, atol=1e-7)
    print("matrix-free == dense: OK")


def test_batch():
    single = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float64)
    batch = torch.stack([single, single * 2.0, single * 0.5])
    y_single = yabi_torch.solve_ivp(decay_f, single, t_span=(0.0, 2.0))
    y_batch = yabi_torch.solve_ivp(decay_f, batch, t_span=(0.0, 2.0))
    assert y_batch.shape == (3, 4)
    assert torch.allclose(y_batch[0], y_single, atol=1e-8)
    assert torch.allclose(y_batch[1], 2.0 * y_single, atol=1e-8)  # linear ODE
    assert torch.allclose(y_batch[2], 0.5 * y_single, atol=1e-8)
    print("batch: OK")


def test_batch_backward():
    # The batched adjoint (backward) must match the single-trajectory adjoint.
    # Guards the buffer-aliasing fix: grad_yT.detach().numpy() shares memory
    # with the torch gradient tensor, and the adjoint's nested autograd reused
    # it mid-read, corrupting the batched gradient.
    single = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float64)
    batch = torch.stack([single, single * 1.1, single * 0.9]).requires_grad_()

    s = single.clone().requires_grad_()
    ys = yabi_torch.solve_ivp(decay_f, s, t_span=(0.0, 2.0))
    ys.sum().backward()

    yb = yabi_torch.solve_ivp(decay_f, batch, t_span=(0.0, 2.0))
    yb.sum().backward()

    # linear ODE: d(sum y)/dy0 is constant across trajectories.
    assert torch.allclose(batch.grad[0], s.grad, atol=1e-12), \
        f"{batch.grad[0]} vs {s.grad}"
    assert torch.allclose(batch.grad[1], s.grad, atol=1e-12)
    assert torch.allclose(batch.grad[2], s.grad, atol=1e-12)
    print("batch backward == single adjoint: OK")


def test_stiff():
    def f(y):  # Robertson (classic stiff)
        return torch.stack([-0.04 * y[0] + 1e4 * y[1] * y[2],
                            0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1],
                            3e7 * y[1] * y[1]])

    y0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    yT = yabi_torch.solve_ivp(f, y0, t_span=(0.0, 1e5), rtol=1e-6, atol=1e-10)
    assert abs(yT.sum().item() - 1.0) < 1e-6, f"mass drift: {yT.sum().item()}"
    print("stiff (Robertson): OK, y(T) =", yT.detach().numpy())


def test_stiff_matrix_free():
    # Same Robertson, matrix-free.  Guards the J·v orientation: feeding the
    # transposed operator (Jᵀv) would break Newton convergence on this strongly
    # non-symmetric Jacobian, so this test fails on the transpose bug.
    def f(y):
        return torch.stack([-0.04 * y[0] + 1e4 * y[1] * y[2],
                            0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1],
                            3e7 * y[1] * y[1]])

    y0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    yT = yabi_torch.solve_ivp(f, y0, t_span=(0.0, 1e5), rtol=1e-6, atol=1e-10,
                              solver="matrix_free")
    assert abs(yT.sum().item() - 1.0) < 1e-6, f"mass drift: {yT.sum().item()}"
    print("stiff (Robertson, matrix-free): OK, y(T) =", yT.detach().numpy())


if __name__ == "__main__":
    test_gradcheck()
    test_analytic_matches_automatic()
    test_matrix_free_matches_dense()
    test_batch()
    test_batch_backward()
    test_stiff()
    test_stiff_matrix_free()
    print("PASS: batched differentiable solve_ivp (dense + matrix-free, analytic + automatic)")
