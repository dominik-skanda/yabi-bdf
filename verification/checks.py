#!/usr/bin/env python3
"""Independent verification of yabi-torch — the four reviewer checks.

Supplementary to the `yabi-torch` package (not part of it).

Checks (Robertson, float64, single-threaded, rtol=1e-6 / atol=1e-10):
  1. Mass conservation at t=1e5 (dense + matrix_free) — sum must stay 1.
  2. Adjoint d y0(T)/d(1e4 coefficient) vs central differences.
  3. Implicit baselines at t=1e5: SciPy BDF and torchdiffeq `implicit_adams`.
  4. Adjoint under Newton-step rejection (rtol=1e-10).

Run:
    pip install yabi-torch scipy torchdiffeq   # torchdiffeq optional (Check 3 row)
    python verification/checks.py

Everything is deterministic for a given machine; wall-clock is informational only.
"""
import signal
import time

import numpy as np
import torch
import torch.nn as nn

from yabi_torch import solve_ivp
from scipy.integrate import solve_ivp as scipy_solve_ivp

try:
    from torchdiffeq import odeint as td_odeint
    HAVE_TORCHDIFFEQ = True
except ImportError:
    HAVE_TORCHDIFFEQ = False

import importlib.metadata as _md

torch.set_num_threads(1)

RTOL, ATOL = 1e-6, 1e-10
T_CONS = 1e5
Y0 = [1.0, 0.0, 0.0]


def robertson_np(t, y):
    return np.array([-0.04 * y[0] + 1e4 * y[1] * y[2],
                     0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2,
                     3e7 * y[1] ** 2])


def robertson_td(t, y):
    return torch.stack([-0.04 * y[0] + 1e4 * y[1] * y[2],
                        0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2,
                        3e7 * y[1] ** 2])


class RobertsonRate(nn.Module):
    """Robertson with the stiff 1e4 coefficient as a trainable parameter."""

    def __init__(self, k=1e4):
        super().__init__()
        self.k = nn.Parameter(torch.tensor(float(k), dtype=torch.float64))

    def forward(self, y):
        return torch.stack([-0.04 * y[0] + self.k * y[1] * y[2],
                            0.04 * y[0] - self.k * y[1] * y[2] - 3e7 * y[1] ** 2,
                            3e7 * y[1] ** 2])


def mass(y):
    return y[0] + y[1] + y[2]


def yabi_solve(f, solver, rtol=RTOL, atol=ATOL, t_end=T_CONS):
    y0 = torch.tensor(Y0, dtype=torch.float64)
    t0 = time.perf_counter()
    yT = solve_ivp(f, y0, t_span=(0.0, t_end), rtol=rtol, atol=atol, solver=solver)
    return time.perf_counter() - t0, yT.detach().numpy()


class _TimedOut(Exception):
    pass


def scipy_bdf(t_end=T_CONS):
    t0 = time.perf_counter()
    sol = scipy_solve_ivp(robertson_np, [0.0, t_end], Y0, method="BDF",
                          rtol=RTOL, atol=ATOL, t_eval=[t_end])
    return time.perf_counter() - t0, sol.y[:, -1]


def td_implicit_adams(t_end=T_CONS):
    if not HAVE_TORCHDIFFEQ:
        return None, None

    def h(signum, frame):
        raise _TimedOut()

    y0 = torch.tensor(Y0, dtype=torch.float64)
    tt = torch.tensor([0.0, t_end], dtype=torch.float64)
    signal.signal(signal.SIGALRM, h)
    signal.alarm(180)
    t0 = time.perf_counter()
    try:
        yT = td_odeint(robertson_td, y0, tt, method="implicit_adams",
                       rtol=RTOL, atol=ATOL)[-1]
        dt = time.perf_counter() - t0
        signal.alarm(0)
        return dt, yT.detach().numpy()
    except Exception:
        signal.alarm(0)
        return None, None


def adjoint_fd(solver, t_end, rtol=RTOL, rtol_fd=1e-12, delta=0.1):
    """Adjoint d y0(T)/dk vs central difference (tight forward reference)."""
    m = RobertsonRate(1e4)
    y0 = torch.tensor(Y0, dtype=torch.float64)

    def g(mm, r):
        return solve_ivp(mm, y0, t_span=(0.0, t_end), rtol=r, atol=ATOL,
                         solver=solver)[0].item()

    for p in m.parameters():
        p.grad = None
    yT = solve_ivp(m, y0, t_span=(0.0, t_end), rtol=rtol, atol=ATOL, solver=solver)[0]
    yT.backward()
    adj = m.k.grad.item()

    k0 = m.k.item()
    m.k.data = torch.tensor(k0 + delta, dtype=torch.float64)
    lp = g(m, rtol_fd)
    m.k.data = torch.tensor(k0 - delta, dtype=torch.float64)
    lm = g(m, rtol_fd)
    m.k.data = torch.tensor(k0, dtype=torch.float64)
    fd = (lp - lm) / (2 * delta)
    return adj, fd


def _ver(name):
    try:
        return _md.version(name)
    except Exception:
        return "?"


def main():
    print(f"yabi-torch {_ver('yabi-torch')}, torch {torch.__version__}, "
          f"scipy {_ver('scipy')}, torchdiffeq {_ver('torchdiffeq')} "
          f"({'installed' if HAVE_TORCHDIFFEQ else 'NOT installed'})")
    print(f"rtol={RTOL}, atol={ATOL}, float64\n")

    # Check 1: mass conservation
    print("=== Check 1: mass conservation (Robertson, t=1e5) ===")
    for solver in ("dense", "matrix_free"):
        dt, yT = yabi_solve(lambda y: robertson_td(0.0, y), solver)
        print(f"  {solver:12s}: y(T)=[{yT[0]:.8e} {yT[1]:.8e} {yT[2]:.8e}]  "
              f"|sum-1|={abs(mass(yT)-1):.2e}  {dt:.3f}s")

    # Check 2: adjoint vs FD on the 1e4 rate
    print("\n=== Check 2: adjoint d y0(T)/d(1e4) vs central differences ===")
    for solver, t_end in (("dense", 1.0), ("matrix_free", 1.0), ("dense", T_CONS),
                          ("matrix_free", T_CONS)):
        try:
            adj, fd = adjoint_fd(solver, t_end)
            print(f"  {solver:12s} t={t_end:>7.0f}: adjoint={adj:+.8e}  FD={fd:+.8e}  "
                  f"|adj-FD|={abs(adj-fd):.2e}  rel={abs(adj-fd)/abs(fd):.2e}")
        except Exception as e:
            print(f"  {solver:12s} t={t_end:>7.0f}: FAILS ({type(e).__name__})")

    # Check 3: implicit baselines
    print("\n=== Check 3: implicit baselines (same rtol/atol/dtype, t=1e5) ===")
    dt, yT = scipy_bdf()
    print(f"  {'scipy BDF':24s}: y(T)=[{yT[0]:.8e} {yT[1]:.8e} {yT[2]:.8e}]  "
          f"|sum-1|={abs(mass(yT)-1):.2e}  {dt:.3f}s")
    if not HAVE_TORCHDIFFEQ:
        print(f"  {'torchdiffeq implicit_adams':24s}: skipped (torchdiffeq not installed)")
    else:
        dt, yT = td_implicit_adams()
        if yT is None:
            print(f"  {'torchdiffeq implicit_adams':24s}: did not converge within 180s")
        else:
            print(f"  {'torchdiffeq implicit_adams':24s}: y(T)=[{yT[0]:.8e} {yT[1]:.8e} {yT[2]:.8e}]  "
                  f"|sum-1|={abs(mass(yT)-1):.2e}  {dt:.3f}s")

    # Check 4: adjoint under rejection (tight rtol)
    print("\n=== Check 4: adjoint under Newton-step rejection (rtol=1e-10, t=1e5) ===")
    for solver in ("dense", "matrix_free"):
        try:
            adj, fd = adjoint_fd(solver, T_CONS, rtol=1e-10)
            print(f"  {solver:12s}: adjoint={adj:+.8e}  FD={fd:+.8e}  "
                  f"|adj-FD|={abs(adj-fd):.2e}  rel={abs(adj-fd)/abs(fd):.2e}")
        except Exception as e:
            print(f"  {solver:12s}: FAILS ({type(e).__name__})")


if __name__ == "__main__":
    main()
