/*
 * Minimal YABI example: autonomous scalar ODE
 *
 *   y' = -0.5 * y,   y(0) = 1
 *
 * Exact solution: y(t) = exp(-0.5 t).
 */

#include <yabi/bdf_integrator.hpp>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

int main() {
    CppAD::vector<CppAD::AD<double> > x(1);
    x[0] = 1.0;
    CppAD::Independent(x);

    CppAD::vector<CppAD::AD<double> > f(1);
    f[0] = -0.5 * x[0];

    CppAD::ADFun<double> rhs;
    rhs.Dependent(x, f);
    rhs.optimize();

    yabi::BDFIntegrator<double> integrator(rhs);

    double y_start[1] = {1.0};
    double absolute_error[1] = {1e-12};
    double y[1];

    if (!integrator.resetIntegration(y_start, 0.1, 1e-10, absolute_error,
                                     yabi::IntegrationOptions<double>()) ||
        !integrator.performIntegration(2.0, 100000) ||
        !integrator.getSolutionVector(y)) {
        std::cerr << "integration failed\n";
        return 1;
    }

    const double exact = std::exp(-0.5 * 2.0);
    std::cout << std::setprecision(15);
    std::cout << "y(2) = " << y[0] << "\n";
    std::cout << "exact = " << exact << "\n";
    std::cout << "error = " << std::fabs(y[0] - exact) << "\n";
    return (std::fabs(y[0] - exact) < 1e-8) ? 0 : 1;
}
