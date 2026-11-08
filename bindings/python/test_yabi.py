"""T37 — YABI Python bindings smoke test (Brusselator, zero-copy NumPy)."""
import numpy as np
import yabi

A, B = 1.0, 3.0


def f(y):
    return np.array([A + y[0] * y[0] * y[1] - (B + 1.0) * y[0],
                     B * y[0] - y[0] * y[0] * y[1]])


def jac(y):
    # column-major J[i + n*j] = df_i/dy_j
    J = np.empty(4)
    J[0] = 2.0 * y[0] * y[1] - (B + 1.0)  # df0/dy0
    J[1] = B - 2.0 * y[0] * y[1]           # df1/dy0
    J[2] = y[0] * y[0]                     # df0/dy1
    J[3] = -y[0] * y[0]                    # df1/dy1
    return J


it = yabi.BDFIntegrator(f, jac, 2)

# Zero-copy round-trip: y0 and atol are NumPy buffers; the binding hands their
# raw pointers to YABI (no copy), and get_solution writes into y_out's buffer.
y0 = np.array([1.0, 2.0])
atol = np.array([1e-10, 1e-10])
assert it.reset_integration(y0, 5.0, 1e-10, atol), "reset_integration failed"
it.set_algorithm_settings(5, np.finfo(np.float64).eps, True, 1)
assert it.integrate(5.0), "integration failed"

y_out = np.zeros(2)
assert it.get_solution(y_out), "get_solution failed"

# Reference y(5) from the C++ Brusselator test (BASELINE: 0.938299, 4.52728).
ref = np.array([0.938299, 4.52728])
err = np.max(np.abs(y_out - ref) / (1.0 + np.abs(ref)))
print("Brusselator y(5) =", y_out)
print("max relative error vs reference =", err)
assert err < 1e-3, "Brusselator mismatch: " + str(y_out)
print("PASS: yabi Python bindings (zero-copy NumPy round-trip)")
