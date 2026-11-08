/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// T42 — RHS-backend agreement: the same ODE integrated through CppADRHS (AD)
// and UserODERHS (analytic Jacobian) must produce the same solution, verified
// against the exact solution of the coupled system y1'=y1^2, y2'=y1*y2.

#include <yabi/bdf_integrator.hpp>
#include <yabi/cppad_rhs.hpp>
#include <yabi/user_ode_rhs.hpp>

#include <cppad/cppad.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const char *what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << std::endl;
        ++failures;
    }
}

const double a = 2.0, b = 3.0, T = 0.2;

CppAD::ADFun<double> makeTape() {
    CppAD::vector<CppAD::AD<double>> x(2);
    x[0] = a;
    x[1] = b;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(2);
    f[0] = x[0] * x[0];
    f[1] = x[0] * x[1];
    CppAD::ADFun<double> fun;
    fun.Dependent(x, f);
    fun.optimize();
    return fun;
}

// Analytic f + Jacobian for y1'=y1^2, y2'=y1*y2.
void ode(const double *y, double *f) {
    f[0] = y[0] * y[0];
    f[1] = y[0] * y[1];
}
void jacobian(const double *y, double *J) {
    J[0] = 2.0 * y[0]; // df0/dy0
    J[1] = y[1];       // df1/dy0
    J[2] = 0.0;        // df0/dy1
    J[3] = y[0];       // df1/dy1
}

bool integrate(const yabi::AbstractRHS<double> &rhs, std::vector<double> &y) {
    yabi::BDFIntegrator<double> it(rhs);
    double y0[2] = {a, b}, abs_err[2] = {1e-12, 1e-12};
    if (!it.resetIntegration(y0, 0.0, NULL, 0.01, 1e-12, abs_err, false, false, NULL, NULL, NULL,
                             NULL, 0) ||
        !it.performIntegration(T, 100000))
        return false;
    y.resize(2);
    return it.getSolutionVector(y.data());
}

} // namespace

int main() {
    std::cout << "T42 RHS-backend agreement (CppAD vs analytic)\n";

    std::vector<double> y_cppad, y_pure;
    {
        CppAD::ADFun<double> fun = makeTape();
        yabi::CppADRHS<double> rhs(fun);
        check(integrate(rhs, y_cppad), "CppAD integration");
    }
    {
        yabi::UserODERHS<double> rhs(2, ode, jacobian);
        check(integrate(rhs, y_pure), "UserODERHS integration");
    }

    // Exact: y1 = a/(1-aT), y2 = b/(1-aT).  The integrator's global error is
    // ~1e-9 (BDF local-error control), so use a relative tolerance.
    const double e = 1.0 - a * T;
    const double ex1 = a / e, ex2 = b / e;
    auto rel = [](double got, double want) {
        return std::fabs(got - want) / (1.0 + std::fabs(want)) < 1e-9;
    };
    check(rel(y_cppad[0], ex1) && rel(y_cppad[1], ex2), "CppAD matches exact");
    check(rel(y_pure[0], ex1) && rel(y_pure[1], ex2), "UserODERHS matches exact");
    // Both backends use the same algorithm + Jacobian, so they agree tightly.
    check(std::fabs(y_cppad[0] - y_pure[0]) < 1e-12 && std::fabs(y_cppad[1] - y_pure[1]) < 1e-12,
          "CppAD == UserODERHS");

    if (failures) {
        std::cerr << failures << " RHS-backend check(s) failed\n";
        return 1;
    }
    std::cout << "PASS: CppAD == UserODERHS == exact\n";
    return 0;
}
