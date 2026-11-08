/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// T43 — Medical Akzo Nobel problem (CWI "Test Set for IVP Solvers", medakzo):
// a 400-dimensional stiff reaction-diffusion system (antibody/tissue
// penetration) with a moving front — the boundary antibody supply phi switches
// 2 -> 0 at t = 5.  Reference solution at t = 20 transcribed from medakzo.f
// (Cray C90, rtol/atol = 1e-10).  The RHS is non-autonomous only through the
// boundary phi(t); the tape carries the time as an extra variable.

#include <yabi/bdf_integrator.hpp>

#include <cppad/cppad.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

constexpr int N = 200; // spatial points (dimension = 2N)
constexpr std::size_t dim = 2 * N;
constexpr double k = 100.0, c = 4.0;
constexpr double dzeta = 1.0 / N;
constexpr double dzeta2 = dzeta * dzeta;

void coeff(double zeta, double &alpha, double &beta) {
    const double dum = (zeta - 1.0) * (zeta - 1.0) / c;
    alpha = 2.0 * (zeta - 1.0) * dum / c;
    beta = dum * dum;
}

// Reference (CWI medakzo.f, t=20), 0-based state indices:
//   [78]=antibody@40, [132]=antibody@67, [170]=antibody@86,
//   [198]=antibody@100, [199]=tissue@100.
const double ref[5] = {2.339942217046434e-4, 3.576835958481664e-4, 3.085949832350532e-4,
                       1.173741304462833e-4, 6.190822732534586e-6};

} // namespace

int main() {
    // Tape domain = state (dim) + time (1).
    CppAD::vector<CppAD::AD<double>> x(dim + 1);
    for (std::size_t p = 0; p < N; ++p) {
        x[2 * p] = 0.0;
        x[2 * p + 1] = 1.0;
    }
    x[dim] = 0.0; // time
    CppAD::Independent(x);

    CppAD::AD<double> t = x[dim];
    CppAD::AD<double> ph = CppAD::CondExpLt(t, CppAD::AD<double>(5.0), CppAD::AD<double>(2.0),
                                           CppAD::AD<double>(0.0));

    CppAD::vector<CppAD::AD<double>> f(dim);
    for (std::size_t p = 0; p < N; ++p) {
        const std::size_t i = 2 * p;
        const double zeta = (double)(p + 1) * dzeta;
        double alpha, beta;
        coeff(zeta, alpha, beta);
        CppAD::AD<double> kprod = k * x[i] * x[i + 1];
        f[i + 1] = -kprod;
        if (p == 0) {
            CppAD::AD<double> yl = ph, yr = x[i + 2];
            f[i] = (yl - 2.0 * x[i] + yr) * beta / dzeta2 + alpha * (yr - yl) / (2.0 * dzeta) -
                   kprod;
        } else if (p == N - 1) {
            f[i] = -kprod;
        } else {
            CppAD::AD<double> yl = x[i - 2], yr = x[i + 2];
            f[i] = (yl - 2.0 * x[i] + yr) * beta / dzeta2 + alpha * (yr - yl) / (2.0 * dzeta) -
                   kprod;
        }
    }
    CppAD::ADFun<double> fun;
    fun.Dependent(x, f);
    fun.optimize();

    yabi::BDFIntegrator<double> it(fun);
    std::vector<double> y0(dim, 0.0), ab(dim, 1e-10);
    for (std::size_t p = 0; p < N; ++p)
        y0[2 * p + 1] = 1.0; // tissue = 1

    if (!it.resetIntegration(y0.data(), 0.0, 1.0, 1e-8, ab.data(), false, false, NULL, NULL, NULL,
                             NULL, 0) ||
        !it.setAlgorithmSettings(5, 1e-16, true, 1) || !it.performIntegration(20.0, 1000000)) {
        std::cerr << "FAIL: MEDAKZO integration did not converge\n";
        return 1;
    }
    std::vector<double> y(dim);
    it.getSolutionVector(y.data());

    // Verify the five reference components (relative tolerance 1e-5).
    double max_rel = 0.0;
    const int idx[5] = {78, 132, 170, 198, 199};
    for (int j = 0; j < 5; ++j)
        max_rel = std::max(max_rel, std::fabs(y[idx[j]] - ref[j]) / (1.0 + std::fabs(ref[j])));
    std::cout << "MEDAKZO max relative error vs CWI reference = " << max_rel << "\n";

    if (max_rel < 1e-5) {
        std::cout << "PASS: MEDAKZO matches the CWI reference solution\n";
        return 0;
    }
    std::cerr << "FAIL: MEDAKZO diverged from the CWI reference\n";
    return 1;
}
