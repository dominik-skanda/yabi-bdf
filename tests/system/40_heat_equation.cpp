/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// T43 — large-sparse stiff benchmark: the 1D heat equation discretized by the
// method of lines (central differences).  u_t = u_xx on [0,1], Dirichlet
// u(0)=u(1)=0, u(x,0)=sin(pi x), with the analytic solution
//   u(x,t) = sin(pi x) exp(-pi^2 t).
// The Jacobian is tridiagonal (sparse) and the system is stiff (stiffness ~ n^2),
// so this is the "large sparse" case the sparse/matrix-free/external solvers were
// built for.  Verified against the analytic solution at a fixed time.

#include <yabi/bdf_integrator.hpp>

#include <cppad/cppad.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

} // namespace

int main() {
    const std::size_t n = 200;   // interior points
    const double dx = 1.0 / (double)(n + 1);
    const double dx2 = dx * dx;
    const double t_end = 0.05;

    // RHS: u_i' = (u_{i-1} - 2 u_i + u_{i+1}) / dx^2, with u_0 = u_{n+1} = 0.
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

    // The residual error vs the analytic PDE solution is the O(dx^2) *spatial*
    // discretization of u_xx (central differences), not a time-integration
    // error: at rtol=1e-10 the BDF time error is negligible, so the result
    // matches the analytic solution to within the spatial truncation ~(pi dx)^2/6.
    const double decay = std::exp(-kPi * kPi * t_end);
    double max_rel = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double exact = std::sin(kPi * (double)(i + 1) * dx) * decay;
        max_rel = std::max(max_rel, std::fabs(y[i] - exact) / (1.0 + std::fabs(exact)));
    }
    std::cout << "n=" << n << " max relative error = " << max_rel
              << " (O(dx^2) spatial truncation)\n";

    if (max_rel < 1e-5) {
        std::cout << "PASS: method-of-lines heat equation matches the analytic solution\n";
        return 0;
    }
    std::cerr << "FAIL: heat equation diverged from the analytic solution\n";
    return 1;
}
