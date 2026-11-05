/*
 * Test 1: Linear autonomous ODE - y' = -lambda y
 * Exact solution known, easy to verify accuracy
 */
#include <yabi/bdf_integrator.hpp>
#include <cmath>
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "=== Test 1: Linear autonomous y' = -lambda*y ===" << std::endl;
    const double lambda = 1.0;
    
    CppAD::vector<CppAD::AD<double> > x(1);
    x[0] = 1.0;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double> > f(1);
    f[0] = -lambda * x[0];
    CppAD::ADFun<double> rhs;
    rhs.Dependent(x, f);
    rhs.optimize();
    
    yabi::BDFIntegrator<double> integrator(rhs);
    double y0[1] = {1.0};
    double abs_err[1] = {1e-14};
    double y[1];
    
    if (!integrator.resetIntegration(y0, 0.1, 1e-12, abs_err, false, false, NULL,NULL,NULL,NULL,0)) {
        std::cerr << "FAIL: resetIntegration" << std::endl; return 1;
    }
    if (!integrator.performIntegration(5.0, 100000)) {
        std::cerr << "FAIL: performIntegration" << std::endl; return 1;
    }
    if (!integrator.getSolutionVector(y)) {
        std::cerr << "FAIL: getSolutionVector" << std::endl; return 1;
    }
    
    double exact = exp(-lambda * 5.0);
    double err = fabs(y[0] - exact);
    std::cout << "y(5) = " << std::setprecision(12) << y[0] << ", exact = " << exact << ", error = " << err << std::endl;
    if (err < 1e-10) { std::cout << "PASS" << std::endl; return 0; }
    else { std::cout << "FAIL" << std::endl; return 1; }
}
