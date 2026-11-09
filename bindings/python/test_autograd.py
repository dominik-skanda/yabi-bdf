"""T38 — continuous-adjoint Python binding, checked against central differences.
No torch/jax required: this proves the *binding* is correct."""
import numpy as np
import yabi

n = 4
k = np.array([0.1, 0.2, 0.3, 0.4])
T = 2.0


def f(y):
    out = np.zeros(n)
    out[0] = -k[0] * y[0]
    for i in range(1, n):
        out[i] = k[i - 1] * y[i - 1] - k[i] * y[i]
    return out


def jac(y):
    J = np.zeros(n * n)  # column-major J[i + n*j] = df_i/dy_j
    for i in range(n):
        J[i + n * i] = -k[i]
    for i in range(1, n):
        J[i + n * (i - 1)] = k[i - 1]
    return J


def integrate(y0):
    it = yabi.BDFIntegrator(f, jac, n)
    atol = np.full(n, 1e-10)
    assert it.reset_integration(y0, 0.1, 1e-10, atol)
    it.set_algorithm_settings(5, np.finfo(np.float64).eps, True, 1)
    assert it.integrate(T)
    y = np.zeros(n)
    assert it.get_solution(y)
    return it, y


y0 = np.array([1.0, 0.5, 0.25, 0.125])

# Central-difference Jacobian dy(T)/dy0 (column j = d y(T) / d y0[j]).
eps = 1e-6
J_fd = np.zeros((n, n))
for j in range(n):
    yp, ym = y0.copy(), y0.copy()
    yp[j] += eps
    ym[j] -= eps
    _, yp_T = integrate(yp)
    _, ym_T = integrate(ym)
    J_fd[:, j] = (yp_T - ym_T) / (2.0 * eps)

# Continuous adjoint (T25): adj = d(w . y(T)) / d y0 = J_fd^T w.
it, _ = integrate(y0)
J_adj = np.zeros((n, n))  # row j = d(e_j . y(T)) / d y0 = J_fd[j, :]
for j in range(n):
    w = np.zeros(n)
    w[j] = 1.0
    adj = np.zeros(n)
    assert it.continuous_adjoint(w, adj)
    J_adj[j, :] = adj

adj_err = np.max(np.abs(J_adj - J_fd))
print("continuous adjoint vs FD :", adj_err)

assert adj_err < 1e-6, f"adjoint mismatch:\n{J_adj}\nvs\n{J_fd}"
print("PASS: continuous-adjoint binding agrees with central differences")
