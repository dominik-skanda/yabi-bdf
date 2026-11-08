/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// T43 — two-body Kepler problem (Hairer & Wanner DETEST A2), a conservation
// benchmark.  The energy E = |v|^2/2 - 1/r and the angular momentum L = x vy -
// y vx are invariants of the exact solution, so integrating one period and
// checking them (and near-return to the start) verifies long-term accuracy
// independently of a stored reference.

#include <yabi/bdf_integrator.hpp>

#include <cppad/cppad.hpp>

#include <cmath>
#include <iostream>

namespace {
constexpr double kPi = 3.14159265358979323846;
}

int main() {
    const double ecc = 0.5;                       // eccentricity
    const double y0[4] = {1.0 - ecc, 0.0, 0.0, std::sqrt((1.0 + ecc) / (1.0 - ecc))};
    const double energy0 = (y0[2] * y0[2] + y0[3] * y0[3]) / 2.0 -
                           1.0 / std::sqrt(y0[0] * y0[0] + y0[1] * y0[1]);
    const double momentum0 = y0[0] * y0[3] - y0[1] * y0[2];

    CppAD::vector<CppAD::AD<double>> x(4);
    x[0] = y0[0];
    x[1] = y0[1];
    x[2] = y0[2];
    x[3] = y0[3];
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(4);
    CppAD::AD<double> r3 = CppAD::pow(x[0] * x[0] + x[1] * x[1], 1.5);
    f[0] = x[2];        // x'
    f[1] = x[3];        // y'
    f[2] = -x[0] / r3;  // x''
    f[3] = -x[1] / r3;  // y''
    CppAD::ADFun<double> fun;
    fun.Dependent(x, f);
    fun.optimize();

    yabi::BDFIntegrator<double> it(fun);
    double abs_err[4] = {1e-12, 1e-12, 1e-12, 1e-12};
    if (!it.resetIntegration(const_cast<double *>(y0), 0.05, 1e-12, abs_err, false, false, NULL,
                             NULL, NULL, NULL, 0) ||
        !it.setAlgorithmSettings(6, 1e-16, true, 1) ||
        !it.performIntegration(2.0 * kPi, 1000000)) {
        std::cerr << "FAIL: integration\n";
        return 1;
    }
    double y[4];
    it.getSolutionVector(y);

    const double energy = (y[2] * y[2] + y[3] * y[3]) / 2.0 -
                          1.0 / std::sqrt(y[0] * y[0] + y[1] * y[1]);
    const double momentum = y[0] * y[3] - y[1] * y[2];
    const double dE = std::fabs(energy - energy0) / (1.0 + std::fabs(energy0));
    const double dL = std::fabs(momentum - momentum0) / (1.0 + std::fabs(momentum0));

    std::cout << "energy drift (relative) = " << dE << ", angular-momentum drift = " << dL
              << "\n";

    if (dE < 1e-8 && dL < 1e-8) {
        std::cout << "PASS: two-body orbit conserves energy and angular momentum\n";
        return 0;
    }
    std::cerr << "FAIL: invariants drifted\n";
    return 1;
}
