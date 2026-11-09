#!/usr/bin/env python3
"""
torchdiffeq_comparison.py — the reproducible source of the announcement numbers.

Compares `yabi_torch.solve_ivp` (YABI's BDF) against `torchdiffeq`'s `odeint`
with the explicit `dopri5` method, on the two cases that motivate the project:

  * Robertson — the canonical *stiff* ODE.  An explicit RK is stability-limited
    to a step size on the fast transient's timescale, so it crawls; BDF steps on
    the slow timescale once the transient decays.  This is where YABI wins.
  * A neural-ODE MLP — *non-stiff*.  Here an explicit RK needs no Newton/LU, so
    `dopri5` is the right tool and is faster.  Reported honestly: this is a
    stiff-systems solver, not a general-purpose one.

Everything is double precision and single-threaded (fair for both).  Wall-clock
is the metric; correctness is checked by comparing the two solutions.

Usage:
    pip install torch torchdiffeq       # torchdiffeq is an optional dev dep
    python3 benchmarks/torchdiffeq_comparison.py

The numbers below were measured on Linux x86_64, torch 2.11.0, torchdiffeq 0.2.5.
"""

import time

import torch

from yabi_torch import solve_ivp

RTOL = 1e-6
ATOL = 1e-10


def robertson(t, y):
    """Robertson chemical kinetics (stiff): y' = f(y), autonomous."""
    return torch.stack(
        [
            -0.04 * y[0] + 1e4 * y[1] * y[2],
            0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2,
            3e7 * y[1] ** 2,
        ]
    )


def _mlp(n, width=8, seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(n, width, dtype=torch.float64),
        torch.nn.Tanh(),
        torch.nn.Linear(width, n, dtype=torch.float64),
    )


def _time_yabi(f, y0, t_end, solver="dense"):
    t0 = time.perf_counter()
    yT = solve_ivp(f, y0, t_span=(0.0, t_end), rtol=RTOL, atol=ATOL, solver=solver)
    return time.perf_counter() - t0, yT


def _time_dopri5(func, y0, t_end):
    from torchdiffeq import odeint

    t = torch.tensor([0.0, t_end], dtype=torch.float64)
    t0 = time.perf_counter()
    yT = odeint(func, y0, t, method="dopri5", rtol=RTOL, atol=ATOL)[-1]
    return time.perf_counter() - t0, yT


def bench_robertson(t_end):
    y0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    dt_yabi, y_yabi = _time_yabi(lambda y: robertson(0.0, y), y0, t_end)
    dt_td, y_td = _time_dopri5(robertson, y0, t_end)
    return dt_yabi, dt_td, float((y_yabi - y_td).abs().max())


def bench_mlp(n, t_end=1.0, width=8):
    net = _mlp(n, width)
    y0 = torch.randn(n, dtype=torch.float64)
    dt_yabi, y_yabi = _time_yabi(net, y0, t_end)
    dt_td, y_td = _time_dopri5(lambda t, y: net(y), y0, t_end)
    return dt_yabi, dt_td, float((y_yabi - y_td).abs().max())


def main():
    try:
        import torchdiffeq  # noqa: F401
    except ImportError:
        raise SystemExit(
            "torchdiffeq is not installed — `pip install torchdiffeq` (optional dev dep)."
        )

    torch.set_num_threads(1)
    print(f"rtol={RTOL}, atol={ATOL}, torch {torch.__version__}, "
          f"torchdiffeq {torchdiffeq.__version__}, single thread\n")

    print("=== Robertson (stiff): YABI BDF vs dopri5 ===")
    for t_end in (1.0, 10.0):
        dy, dt_, err = bench_robertson(t_end)
        print(f"  t={t_end:>3}: yabi {dy:8.3f}s | dopri5 {dt_:8.3f}s | "
              f"speedup {dt_/dy:6.1f}x | max|dy| = {err:.2e}")

    print("\n=== MLP (non-stiff): YABI BDF vs dopri5 ===")
    for n in (3, 10, 50):
        dy, dt_, err = bench_mlp(n)
        print(f"  n={n:>3}: yabi {dy:8.3f}s | dopri5 {dt_:8.3f}s | "
              f"speedup {dt_/dy:6.2f}x | max|dy| = {err:.2e}")


if __name__ == "__main__":
    main()
