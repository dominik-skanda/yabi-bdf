"""T53 — parallel batch: `threads=N` is bit-identical to `threads=1`.

Independent trajectories are solved on a thread pool (the binding releases the
GIL during the C++ solve and re-acquires it only for the Python f/jac/jv
callbacks).  The result must match serial exactly, for both the dense and
matrix-free paths, in the forward and the adjoint backward.
"""
import time

import numpy as np
import torch

import yabi_torch

k = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)


def decay_f(y):  # torch (4,) -> (4,), functional
    neg = -k * y
    pos = torch.cat([torch.zeros(1, dtype=torch.float64), k[:-1] * y[:-1]])
    return neg + pos


def make_batch(batch):
    single = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float64)
    # distinct (non-scaled) initial conditions so it is a real batch
    rows = [single * (1.0 + 0.13 * i) + 0.01 * i for i in range(batch)]
    return torch.stack(rows)


def test_parallel_equals_serial_dense():
    batch = make_batch(8)
    serial = yabi_torch.solve_ivp(decay_f, batch, t_span=(0.0, 2.0), threads=1)
    par = yabi_torch.solve_ivp(decay_f, batch, t_span=(0.0, 2.0), threads=4)
    assert torch.equal(serial, par), "dense: threads=4 != threads=1"
    print("dense: threads=4 == threads=1 (bit-identical): OK")


def test_parallel_equals_serial_matrix_free():
    batch = make_batch(8)
    serial = yabi_torch.solve_ivp(decay_f, batch, t_span=(0.0, 2.0),
                                  solver="matrix_free", threads=1)
    par = yabi_torch.solve_ivp(decay_f, batch, t_span=(0.0, 2.0),
                               solver="matrix_free", threads=4)
    assert torch.equal(serial, par), "matrix_free: threads=4 != threads=1"
    print("matrix_free: threads=4 == threads=1 (bit-identical): OK")


def test_parallel_backward_matches_serial():
    b_ser = make_batch(5).requires_grad_()
    b_par = b_ser.detach().clone().requires_grad_()
    y_ser = yabi_torch.solve_ivp(decay_f, b_ser, t_span=(0.0, 2.0), threads=1)
    y_par = yabi_torch.solve_ivp(decay_f, b_par, t_span=(0.0, 2.0), threads=3)
    y_ser.sum().backward()
    y_par.sum().backward()
    assert torch.equal(b_ser.grad, b_par.grad), "backward: threads=3 != threads=1"
    print("backward: threads=3 == threads=1 (bit-identical): OK")


def test_parallel_gradcheck():
    batch = make_batch(3).requires_grad_()

    def wrapped(y):
        return yabi_torch.solve_ivp(decay_f, y, t_span=(0.0, 2.0), threads=2).sum()

    assert torch.autograd.gradcheck(wrapped, (batch,))
    print("gradcheck with threads=2: OK")


def test_scaling_informal():
    # Correctness is the contract; this just reports the wall-clock so the
    # parallel path is observed (speedup is environment/thread-count dependent).
    batch = make_batch(16)
    t0 = time.perf_counter()
    yabi_torch.solve_ivp(decay_f, batch, t_span=(0.0, 2.0), threads=1)
    t1 = time.perf_counter()
    yabi_torch.solve_ivp(decay_f, batch, t_span=(0.0, 2.0), threads=8)
    t2 = time.perf_counter()
    print(f"scaling: threads=1 {t1 - t0:.3f}s, threads=8 {t2 - t1:.3f}s")


if __name__ == "__main__":
    test_parallel_equals_serial_dense()
    test_parallel_equals_serial_matrix_free()
    test_parallel_backward_matches_serial()
    test_parallel_gradcheck()
    test_scaling_informal()
    print("PASS: parallel batch (threads=N) bit-identical to serial")
