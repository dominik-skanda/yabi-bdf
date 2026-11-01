// T15 — E5 problem (stiff, autonomous) from Hairer-Wanner §IV.10
#include <yabi/bdf_integrator.hpp>
#include <cppad/cppad.hpp>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>

int main() {
    std::cout << "=== Test 30: E5 (stiff autonomous) ===" << std::endl;
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double> > x(3);
    x[0] = 1.76e-3; x[1] = 0.0; x[2] = 0.0;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double> > f(3);
    f[0] = -7.89e-10*x[0] - 1.1e7*x[0]*x[2];
    f[1] =  7.89e-10*x[0] - 1.13e9*x[1]*x[2];
    f[2] =  7.89e-10*x[0] - 1.1e7*x[0]*x[2] - 1.13e9*x[1]*x[2];
    rhs.Dependent(x, f);
    rhs.optimize();

    yabi::BDFIntegrator<double> integrator(rhs);
    double y0[3] = {1.76e-3, 0.0, 0.0};
    double abs_err[3] = {1e-18, 1e-24, 1e-18};
    double y[3];

    if (!integrator.resetIntegration(y0, 1e6, 1e-12, abs_err, false, false, NULL,NULL,NULL,NULL,0)) {
        std::cerr << "FAIL: reset" << std::endl; return 1;
    }
    if (!integrator.setAlgorithmSettings(5, std::numeric_limits<double>::epsilon(), true, 1)) {
        std::cerr << "FAIL: setAlgorithmSettings" << std::endl; return 1;
    }
    if (!integrator.performIntegration(1e6, 10000000)) {
        std::cerr << "FAIL: integrate" << std::endl; return 1;
    }
    if (!integrator.getSolutionVector(y)) {
        std::cerr << "FAIL: get" << std::endl; return 1;
    }

    std::cout << std::setprecision(12);
    std::cout << "y1=" << y[0] << " y2=" << y[1] << " y3=" << y[2] << std::endl;
    if (std::isfinite(y[0]) && std::isfinite(y[1]) && std::isfinite(y[2])) {
        std::cout << "PASS" << std::endl; return 0;
    }
    std::cout << "FAIL" << std::endl; return 1;
}
