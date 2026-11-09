"""T54 — GPU.

Two paths, tested separately:
  * the cuSOLVER **LU offload** (yabi.cuda_integrator) — verified on this machine
    (bit-identical to the CPU LAPACK solve); works on any CUDA GPU.
  * the **f-on-GPU** path (solve_ivp(..., device="cuda")) — implemented but needs a
    compute-capability >= 7.0 GPU (torch's cu12x wheels dropped Pascal/CC 6.x), so
    it is skipped here and ready for a newer GPU.
"""
import numpy as np
import torch

import yabi
import yabi_torch

k = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)


def decay_f(y):  # torch (4,) -> (4,), functional
    neg = -k * y
    pos = torch.cat([torch.zeros(1, dtype=torch.float64), k[:-1] * y[:-1]])
    return neg + pos


def _have_cuda_module():
    if not torch.cuda.is_available():
        print("SKIP: torch CUDA not available")
        return False
    if not hasattr(yabi, "cuda_integrator"):
        print("SKIP: yabi module built without CUDA (YABI_USE_CUDA=OFF)")
        return False
    return True


def test_cuda_lu_offload():
    """cuSOLVER LU offload == CPU LAPACK (verified bit-identical on this machine)."""
    if not _have_cuda_module():
        return
    n = 4
    f_np = yabi_torch._wrap_rhs(decay_f, "cpu")
    jac_np = yabi_torch._automatic_jacobian(decay_f, n, "cpu")
    y0 = np.array([1.0, 0.5, 0.25, 0.125])

    def solve(make):
        it = make()
        it.reset_integration(y0, 2.0, 1e-8, np.full(n, 1e-10))
        it.set_algorithm_settings(5, np.finfo(np.float64).eps, True, 1)
        assert it.integrate(2.0)
        yb = np.zeros(n)
        assert it.get_solution(yb)
        return yb

    gpu = solve(lambda: yabi.cuda_integrator(f_np, jac_np, n))
    cpu = solve(lambda: yabi.BDFIntegrator(f_np, jac_np, n))
    assert np.array_equal(gpu, cpu), f"{gpu} vs {cpu}"
    print("cuSOLVER LU offload == CPU (bit-identical): OK")


def test_cuda_f_on_gpu():
    """f evaluated on-device (device='cuda') == CPU.  Needs CC >= 7.0."""
    if not _have_cuda_module():
        return
    cc = torch.cuda.get_device_capability(0)
    if (cc[0], cc[1]) < (7, 5):
        print(f"SKIP: f-on-GPU needs CC >= 7.5 (device is {cc[0]}.{cc[1]})")
        return

    kc = k.cuda()
    def f(y):
        neg = -kc * y
        pos = torch.cat([torch.zeros(1, dtype=torch.float64, device=y.device),
                         kc[:-1] * y[:-1]])
        return neg + pos

    y0 = torch.tensor([1.0, 0.5, 0.25, 0.125], dtype=torch.float64)
    cpu = yabi_torch.solve_ivp(decay_f, y0, t_span=(0.0, 2.0))
    gpu = yabi_torch.solve_ivp(f, y0, t_span=(0.0, 2.0), device="cuda")
    assert torch.allclose(gpu, cpu, atol=1e-7, rtol=1e-7), f"{gpu} vs {cpu}"

    mfree = yabi_torch.solve_ivp(f, y0, t_span=(0.0, 2.0), solver="matrix_free",
                                 device="cuda")
    assert torch.allclose(mfree, cpu, atol=1e-6, rtol=1e-6), f"{mfree} vs {cpu}"
    print("f-on-GPU (dense + matrix-free) == CPU: OK")


if __name__ == "__main__":
    test_cuda_lu_offload()
    test_cuda_f_on_gpu()
    print("PASS: GPU paths (LU offload verified; f-on-GPU gated on CC >= 7.5)")
