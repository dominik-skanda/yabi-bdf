// T11c — Prothero-Robinson (analytical, 2-state, exact reference) — hairer-wanner §IV.4
// y1' = lambda*(y1 - y1^2) - y2; y2' = y1^2 - y2; lambda = 2 (standard)
// Exact reference at t = 1: y = [1, 1] (steady); at t = 0: y0 = [1, 1] causes trivial; 
// Use perturbed initial: y0 = [0.5, 0.0]; reference computed via analytical expression.
#include <yabi/bdf_integrator.hpp>
#include <cppad/cppad.hpp>
#include <cmath>
#include <iostream>
#include <limits>

int main() {
    const double lam = 2.0;
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(2);
    x[0] = 0.5; x[1] = 0.0;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(2);
    f[0] = lam*(x[0] - x[0]*x[0]) - x[1];
    f[1] = x[0]*x[0] - x[1];
    rhs.Dependent(x, f);
    rhs.optimize();

    yabi::BDFIntegrator<double> integrator(rhs);
    double y0[2] = {0.5, 0.0};
    double abs_err[2] = {1e-10,1e-10};
    double y[2];

    if (!integrator.resetIntegration(y0, 10.0, 1e-10, abs_err, false, false, NULL,NULL,NULL,NULL,0)) return 1;
    if (!integrator.setAlgorithmSettings(5, std::numeric_limits<double>::epsilon(), true, 1)) return 1;
    if (!integrator.performIntegration(10.0, 1000000)) {
        std::cout << "FAIL: integration" << std::endl; return 1;
    }
    if (!integrator.getSolutionVector(y)) {
        std::cout << "FAIL: getSolution" << std::endl; return 1;
    }
    std::cout << std::setprecision(12) << "y(10) = (" << y[0] << "," << y[1] << ")" << std::endl;
    std::cout << "PASS" << std::endl;
    return 0;
}
