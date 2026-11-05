// T11e — Damped pendulum (mechanical stiff, large damping)
#include <yabi/bdf_integrator.hpp>
#include <cppad/cppad.hpp>
#include <cmath>
#include <iostream>
#include <limits>

int main() {
    const double mu = 10.0;  // large damping = stiff
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(2);
    x[0] = M_PI/2.0; x[1] = 0.0;  // initial angle π/2, zero velocity
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(2);
    f[0] = x[1];
    f[1] = -mu*x[1] - 9.81*sin(x[0]);
    rhs.Dependent(x, f);
    rhs.optimize();

    yabi::BDFIntegrator<double> integrator(rhs);
    double y0[2] = {M_PI/2.0, 0.0};
    double abs_err[2] = {1e-10, 1e-10};
    double y[2];

    if (!integrator.resetIntegration(y0, 5.0, 1e-10, abs_err, false, false, NULL, NULL, NULL, NULL, 0)) return 1;
    if (!integrator.setAlgorithmSettings(5, std::numeric_limits<double>::epsilon(), true, 1)) return 1;
    if (!integrator.performIntegration(5.0, 1000000)) {
        std::cout << "FAIL" << std::endl; return 1;
    }
    if (!integrator.getSolutionVector(y)) return 1;
    std::cout << std::setprecision(12) << "y(5) = (" << y[0] << "," << y[1] << ")" << std::endl;
    std::cout << "PASS" << std::endl;
    return 0;
}
