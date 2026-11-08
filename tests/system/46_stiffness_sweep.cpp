/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// T43 — stiffness-ratio stress sweep.  Prothero-Robinson (Hairer & Wanner §IV.1,
// the canonical stiffness probe from Enright-Hull STDTST class E):
//   y' = lambda * (y - sin(t)) + cos(t),  y(0) = 1,
// with exact solution  y(t) = sin(t) + e^{lambda t}.
// The stiffness ratio is |lambda|; sweeping lambda from -1e1 to -1e8 (seven
// orders of magnitude) asserts the BDF method stays finite and accurate across
// the whole range, making the "stable on long/stiff" claim quantitative.

#include <yabi/bdf_integrator.hpp>

#include <cppad/cppad.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>

namespace {

int failures = 0;

} // namespace

int main() {
    std::cout << "T43 stiffness-ratio stress sweep (Prothero-Robinson)\n";
    const double t_end = 1.0;
    const double lambdas[] = {-1e1, -1e2, -1e3, -1e4, -1e5, -1e6, -1e7, -1e8};

    for (double lambda : lambdas) {
        // Tape domain = state(1) + time(1): f = lambda*(y - sin(t)) + cos(t).
        CppAD::vector<CppAD::AD<double>> x(2);
        x[0] = 1.0; // y
        x[1] = 0.0; // t
        CppAD::Independent(x);
        CppAD::vector<CppAD::AD<double>> f(1);
        f[0] = lambda * (x[0] - CppAD::sin(x[1])) + CppAD::cos(x[1]);
        CppAD::ADFun<double> fun;
        fun.Dependent(x, f);
        fun.optimize();

        yabi::BDFIntegrator<double> it(fun);
        double y0[1] = {1.0};
        double atol[1] = {1e-11};
        if (!it.resetIntegration(y0, 0.0, 0.1, 1e-10, atol, false, false, NULL, NULL, NULL, NULL,
                                 0) ||
            !it.setAlgorithmSettings(5, 1e-16, true, 1) ||
            !it.performIntegration(t_end, 10000000)) {
            std::cerr << "  lambda=" << lambda << " FAILED to converge\n";
            ++failures;
            continue;
        }
        double y[1];
        it.getSolutionVector(y);

        const double exact = std::sin(t_end) + std::exp(lambda * t_end);
        const double err = std::fabs(y[0] - exact) / (1.0 + std::fabs(exact));
        std::cout << "  lambda=" << lambda << " y(1)=" << y[0] << " exact=" << exact
                  << " rel_err=" << err << "\n";
        if (!(err < 1e-6) || !std::isfinite(y[0])) {
            std::cerr << "  lambda=" << lambda << " FAILED (err=" << err << ")\n";
            ++failures;
        }
    }

    if (failures) {
        std::cerr << failures << " stiffness-sweep case(s) failed\n";
        return 1;
    }
    std::cout << "PASS: finite + accurate across stiffness ratios 1e1..1e8\n";
    return 0;
}
