"""T51 — batched, differentiable solve_ivp (torch gradcheck + batch + stiff)."""
import numpy as np
import torch

import yabi_torch

# ---- decay chain (n=4) for the gradcheck + batch ----
k = np.array([0.1, 0.2, 0.3, 0.4])


def decay_f(y):
    out = np.zeros(4)
    out[0] = -k[0] * y[0]
    for i in range(1, 4):
        out[i] = k[i - 1] * y[i - 1] - k[i] * y[i]
    return out


def decay_jac(y):
    J = np.zeros(16)
    for i in range(4):
        J[i + 4 * i] = -k[i]
    for i in range(1, 4):
        J[i + 4 * (i - 1)] = k[i - 1]
    return J


def test_gradcheck():
    def wrapped(y0):
        return yabi_torch.solve_ivp(decay_f, decay_jac, y0, t_span=(0.0, 2.0),
                                    rtol=1e-8, atol=1e-10)

    y0 = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(wrapped, (y0,))
    print("gradcheck: OK")


def test_batch():
    single = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float64)
    batch = torch.stack([single, single * 2.0, single * 0.5])

    y_single = yabi_torch.solve_ivp(decay_f, decay_jac, single, t_span=(0.0, 2.0))
    y_batch = yabi_torch.solve_ivp(decay_f, decay_jac, batch, t_span=(0.0, 2.0))

    assert y_batch.shape == (3, 4)
    assert torch.allclose(y_batch[0], y_single, atol=1e-8)
    assert torch.allclose(y_batch[1], 2.0 * y_single, atol=1e-8)  # linear ODE
    assert torch.allclose(y_batch[2], 0.5 * y_single, atol=1e-8)
    print("batch: OK")


def test_stiff():
    # Robertson (classic stiff): y0+y1+y2 == 1 (mass conservation) to t=1e5.
    def f(y):
        return np.array([-0.04 * y[0] + 1e4 * y[1] * y[2],
                         0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] * y[1],
                         3e7 * y[1] * y[1]])

    def jac(y):
        return np.array([-0.04, 0.04, 0.0,
                         1e4 * y[2], -1e4 * y[2] - 6e7 * y[1], 6e7 * y[1],
                         1e4 * y[1], -1e4 * y[1], 0.0])  # column-major

    y0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    yT = yabi_torch.solve_ivp(f, jac, y0, t_span=(0.0, 1e5), rtol=1e-6, atol=1e-10)
    assert abs(yT.sum().item() - 1.0) < 1e-6, f"mass drift: {yT.sum().item()}"
    print("stiff (Robertson): OK, y(T) =", yT.detach().numpy())


if __name__ == "__main__":
    test_gradcheck()
    test_batch()
    test_stiff()
    print("PASS: batched differentiable solve_ivp")
