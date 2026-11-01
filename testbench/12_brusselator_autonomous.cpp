// T11a — Brusselator (stiff, 2-state, autonomous) — hairer-wanner §IV.4 variant
#include <yabi/bdf_integrator.hpp>
#include <cppad/cppad.hpp>
#include <cmath>
#include <iostream>

int main() {
    // Brusselator: y' = A + y1^2*y2 - (B+1)*y1;  y2' = B*y1 - y1^2*y2
    // Standard stiff params: A=1, B=3; initial y0 = [1.0, 2.0]
    const double A = 1.0, B = 3.0;
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(2);
    x[0] = 1.0; x[1] = 2.0;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(2);
    f[0] = A + x[0]*x[0]*x[1] - (B+1.0)*x[0];
    f[1] = B*x[0] - x[0]*x[0]*x[1];
    rhs.Dependent(x, f);
    rhs.optimize();

    yabi::BDFIntegrator<double> integrator(rhs);
    double y0[2] = {1.0, 2.0};
    double abs_err[2] = {1e-10, 1e-10};
    double y[2];

    if (!integrator.resetIntegration(y0, 5.0, 1e-10, abs_err, false, false, NULL, NULL, NULL, NULL, 0)) {
        std::cerr << "FAIL: reset" << std::endl; return 1;
    }
    if (!integrator.setAlgorithmSettings(5, std::numeric_limits<double>::epsilon(), true, 1)) {
        std::cerr << "FAIL: settings" << std::endl; return 1;
    }
    bool ok = integrator.performIntegration(5.0, 10000000);

    if (!ok) {
        std::cout << "FAIL: integration failed" << std::endl;
        return 1;
    }
    if (!integrator.getSolutionVector(y)) {
        std::cerr << "FAIL: getSolutionVector" << std::endl; return 1;
    }
    std::cout << "y1(5) = " << y[0] << ", y2(5) = " << y[1] << std::endl;
    std::cout << "PASS" << std::endl;
    return 0;
}
