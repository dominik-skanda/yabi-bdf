/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// T45 — damped Newton (line search) soundness: on a problem whose Newton
// iteration converges without overshooting, the opt-in line search must not
// change the solution.  It is a *mathematical* no-op (the full step, lambda = 1,
// is accepted on the first try); the one extra residual evaluation it performs
// reorders floating-point, so the two runs agree to the last few ULP but are not
// bit-identical.  That last-ULP difference is exactly why the feature is opt-in
// (off by default) rather than a default behaviour change.

#include <yabi/bdf_integrator.hpp>

#include <cppad/cppad.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>

namespace {

int failures = 0;

// Robertson (stiff), integrated with the line search off and on; both must agree
// to floating-point roundoff (the line search must not perturb the solution).
bool integrateRobertson(bool line_search, double y[3]) {
    CppAD::vector<CppAD::AD<double>> x(3);
    x[0] = 1.0;
    x[1] = 0.0;
    x[2] = 0.0;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(3);
    f[0] = -0.04 * x[0] + 1.0e4 * x[1] * x[2];
    f[1] = 0.04 * x[0] - 1.0e4 * x[1] * x[2] - 3.0e7 * x[1] * x[1];
    f[2] = 3.0e7 * x[1] * x[1];
    CppAD::ADFun<double> fun;
    fun.Dependent(x, f);
    fun.optimize();

    yabi::BDFIntegrator<double> it(fun);
    double y0[3] = {1.0, 0.0, 0.0};
    double atol[3] = {1e-12, 1e-18, 1e-12};
    if (line_search)
        it.setNewtonLineSearch(true, 6);
    if (!it.resetIntegration(y0, 100.0, 1e-10, atol, false, false, NULL, NULL, NULL, NULL, 0) ||
        !it.setAlgorithmSettings(5, 1e-16, true, 1) || !it.performIntegration(1e4, 10000000))
        return false;
    return it.getSolutionVector(y);
}

} // namespace

int main() {
    std::cout << "T45 damped-Newton line search: no-op on non-overshooting problems\n";
    double a[3], b[3];
    if (!integrateRobertson(false, a) || !integrateRobertson(true, b)) {
        std::cerr << "FAIL: integration\n";
        return 1;
    }
    for (int i = 0; i < 3; ++i) {
        double d = std::fabs(a[i] - b[i]) / (1.0 + std::fabs(a[i]));
        if (!(d < 1e-14)) {
            std::cerr << "FAIL: component " << i << " differs beyond roundoff (" << a[i] << " vs "
                      << b[i] << ", rel " << d << ")\n";
            ++failures;
        }
    }
    std::cout << "y (line search off) = (" << a[0] << ", " << a[1] << ", " << a[2] << ")\n";
    std::cout << "y (line search on)  = (" << b[0] << ", " << b[1] << ", " << b[2] << ")\n";

    if (failures) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "PASS: line search does not perturb non-overshooting problems (roundoff only)\n";
    return 0;
}
