/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// T36e — GPU sparse iterative Krylov on a large-sparse literature benchmark:
// the 1D heat equation by the method of lines (central differences).  Same
// problem as test 40, but with n=500 interior points (a 500x500 tridiagonal
// system, stiffness ~ n^2) solved with the CudaKrylovSolver (cuSPARSE SpMV +
// cuBLAS BiCGStab), and verified against the analytic solution
//   u(x,t) = sin(pi x) exp(-pi^2 t).
// Built only when YABI_USE_CUDA=ON (registered separately, not by the numeric
// system glob).

#include <yabi/bdf_integrator.hpp>
#include <yabi/cppad_rhs.hpp>
#include <yabi/cuda_krylov_solver.hpp>

#include <cppad/cppad.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {
constexpr double kPi = 3.14159265358979323846;
}

int main() {
    const std::size_t n = 500; // interior points
    const double dx = 1.0 / (double)(n + 1);
    const double dx2 = dx * dx;
    const double t_end = 0.05;

    // RHS: u_i' = (u_{i-1} - 2 u_i + u_{i+1}) / dx^2, u_0 = u_{n+1} = 0.
    CppAD::vector<CppAD::AD<double>> x(n);
    for (std::size_t i = 0; i < n; ++i)
        x[i] = std::sin(kPi * (double)(i + 1) * dx);
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(n);
    for (std::size_t i = 0; i < n; ++i) {
        CppAD::AD<double> left = (i > 0) ? x[i - 1] : CppAD::AD<double>(0.0);
        CppAD::AD<double> right = (i + 1 < n) ? x[i + 1] : CppAD::AD<double>(0.0);
        f[i] = (left - 2.0 * x[i] + right) / dx2;
    }
    CppAD::ADFun<double> fun;
    fun.Dependent(x, f);
    fun.optimize();

    yabi::BDFIntegrator<double> it(fun);
    auto *solver = new yabi::CudaKrylovSolver<double>();
    solver->setTolerance(1e-10, 0.0, 200, 30);
    // Inexact Newton (Eisenstat-Walker): loosen the Krylov tolerance as the
    // Newton residual shrinks, so the linear solve stays achievable on a stiff
    // system (otherwise the fixed relative tolerance becomes unachievably tight).
    solver->setForcingTerm(yabi::eisenstatWalkerForcing<double>());
    it.setLinearSolver(solver);

    std::vector<double> y0(n), abs_err(n);
    for (std::size_t i = 0; i < n; ++i) {
        y0[i] = std::sin(kPi * (double)(i + 1) * dx);
        abs_err[i] = 1e-10;
    }
    if (!it.resetIntegration(y0.data(), 0.02, 1e-10, abs_err.data(), false, false, NULL, NULL, NULL,
                             NULL, 0) ||
        !it.setAlgorithmSettings(5, 1e-16, true, 1) ||
        !it.performIntegration(t_end, 1000000)) {
        std::cerr << "FAIL: integration\n";
        return 1;
    }
    std::vector<double> y(n);
    if (!it.getSolutionVector(y.data())) {
        std::cerr << "FAIL: getSolutionVector\n";
        return 1;
    }

    const double decay = std::exp(-kPi * kPi * t_end);
    double max_rel = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double exact = std::sin(kPi * (double)(i + 1) * dx) * decay;
        max_rel = std::max(max_rel, std::fabs(y[i] - exact) / (1.0 + std::fabs(exact)));
    }
    std::cout << "n=" << n << " max relative error = " << max_rel
              << " (O(dx^2) spatial truncation)\n";

    if (max_rel < 1e-5) {
        std::cout << "PASS: GPU Krylov solves the method-of-lines heat equation to the analytic "
                     "solution\n";
        return 0;
    }
    std::cerr << "FAIL: heat equation diverged from the analytic solution\n";
    return 1;
}
