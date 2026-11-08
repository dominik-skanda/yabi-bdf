/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// T49 step 1 — HIRES on the pure/analytic path (UserODERHS + hand-written
// Jacobian).  Same problem as system test 13 (Hairer & Wanner §IV.4), but with
// an analytic J so the performance path (no CppAD auto-diff) is measurable.
// Verifies the analytic trajectory matches the CppAD reference to ~1e-9.

#include <yabi/bdf_integrator.hpp>
#include <yabi/user_ode_rhs.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <vector>

namespace {

constexpr int n = 8;

// HIRES right-hand side, verbatim from hir_equation.f (Hairer & Wanner §IV.4).
void hires_ode(const double *y, double *f) {
    f[0] = -1.71 * y[0] + 0.43 * y[1] + 8.32 * y[2] + 0.0007;
    f[1] = 1.71 * y[0] - 8.75 * y[1];
    f[2] = -10.03 * y[2] + 0.43 * y[3] + 0.035 * y[4];
    f[3] = 8.32 * y[1] + 1.71 * y[2] - 1.12 * y[3];
    f[4] = -1.745 * y[4] + 0.43 * y[5] + 0.43 * y[6];
    f[5] = -280.0 * y[5] * y[7] + 0.69 * y[3] + 1.71 * y[4] - 0.43 * y[5] + 0.69 * y[6];
    f[6] = 280.0 * y[5] * y[7] - 1.81 * y[6];
    f[7] = -f[6];
}

// Analytic Jacobian J[i + n*j] = df_i/dy_j (column-major), derived by hand from
// hires_ode above.  Only the 280*y5*y7 nonlinearity contributes y-dependent
// entries (columns 5 and 7); everything else is constant.
void hires_jacobian(const double *y, double *J) {
    for (int i = 0; i < n * n; ++i)
        J[i] = 0.0;
    // column 0 (df/dy0)
    J[0 + n * 0] = -1.71;
    J[1 + n * 0] = 1.71;
    // column 1 (df/dy1)
    J[0 + n * 1] = 0.43;
    J[1 + n * 1] = -8.75;
    J[3 + n * 1] = 8.32;
    // column 2 (df/dy2)
    J[0 + n * 2] = 8.32;
    J[2 + n * 2] = -10.03;
    J[3 + n * 2] = 1.71;
    // column 3 (df/dy3)
    J[2 + n * 3] = 0.43;
    J[3 + n * 3] = -1.12;
    J[5 + n * 3] = 0.69;
    // column 4 (df/dy4)
    J[2 + n * 4] = 0.035;
    J[4 + n * 4] = -1.745;
    J[5 + n * 4] = 1.71;
    // column 5 (df/dy5)
    J[4 + n * 5] = 0.43;
    J[5 + n * 5] = -280.0 * y[7] - 0.43;
    J[6 + n * 5] = 280.0 * y[7];
    J[7 + n * 5] = -280.0 * y[7];
    // column 6 (df/dy6)
    J[4 + n * 6] = 0.43;
    J[5 + n * 6] = 0.69;
    J[6 + n * 6] = -1.81;
    J[7 + n * 6] = 1.81;
    // column 7 (df/dy7)
    J[5 + n * 7] = -280.0 * y[5];
    J[6 + n * 7] = 280.0 * y[5];
    J[7 + n * 7] = -280.0 * y[5];
}

} // namespace

int main() {
    // Reference at t = 321.812 from the CppAD HIRES test (testbench 13).
    const double ref[8] = {0.000737133286353, 0.000144248972571, 5.88876753153e-05,
                           0.00117565512229,  0.00238641718911,  0.00623915986836,
                           0.00285004111321,  0.00284995888679};

    yabi::UserODERHS<double> rhs(n, hires_ode, hires_jacobian);
    yabi::BDFIntegrator<double> integrator(rhs);

    double y0[n] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057};
    double abs_err[n] = {1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10};
    double y[n];

    if (!integrator.resetIntegration(y0, 321.812, 1e-10, abs_err, false, false, NULL, NULL, NULL,
                                     NULL, 0)) {
        std::cerr << "FAIL: reset" << std::endl;
        return 1;
    }
    if (!integrator.setAlgorithmSettings(5, std::numeric_limits<double>::epsilon(), true, 1)) {
        std::cerr << "FAIL: settings" << std::endl;
        return 1;
    }
    if (!integrator.performIntegration(321.812, 10000000)) {
        std::cerr << "FAIL: integration did not converge" << std::endl;
        return 1;
    }
    if (!integrator.getSolutionVector(y)) {
        std::cerr << "FAIL: getSolution" << std::endl;
        return 1;
    }

    double max_rel = 0.0;
    for (int i = 0; i < n; ++i)
        max_rel = std::max(max_rel, std::fabs(y[i] - ref[i]) / (1.0 + std::fabs(ref[i])));
    std::cout << "HIRES (analytic) max relative error vs CppAD reference = " << max_rel << "\n";

    if (max_rel < 1e-9) {
        std::cout << "PASS: analytic HIRES matches the CppAD reference\n";
        return 0;
    }
    std::cerr << "FAIL: analytic HIRES diverged from the reference\n";
    return 1;
}
