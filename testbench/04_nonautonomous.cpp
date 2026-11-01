/*
 * Test 4: Non-autonomous - matches example
 */
#include <yabi/bdf_integrator.hpp>
#include <cmath>
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "=== Test 4: Non-autonomous y' = -0.5*y + t ===" << std::endl;
    
    CppAD::vector<CppAD::AD<double> > x(2);
    x[0] = 1.0; // y
    x[1] = 0.0; // t
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double> > f(1);
    f[0] = -0.5*x[0] + x[1];
    CppAD::ADFun<double> rhs;
    rhs.Dependent(x, f);
    rhs.optimize();
    
    yabi::BDFIntegrator<double> integrator(rhs);
    double y0[1] = {1.0};
    double abs_err[1] = {1e-12};
    double y[1];
    
    if (!integrator.resetIntegration(y0, 0.0, 0.1, 1e-10, abs_err, false, false, NULL,NULL,NULL,NULL,0)) {
        std::cerr << "FAIL: reset" << std::endl; return 1;
    }
    if (!integrator.performIntegration(2.0, 100000)) {
        std::cerr << "FAIL: integrate" << std::endl; return 1;
    }
    if (!integrator.getSolutionVector(y)) {
        std::cerr << "FAIL: get" << std::endl; return 1;
    }
    
    double exact = 5.0*std::exp(-1.0);
    double err = fabs(y[0]-exact);
    std::cout << "y(2)=" << y[0] << " exact=" << exact << " err=" << err << std::endl;
    if (err < 1e-8) { std::cout << "PASS" << std::endl; return 0; }
    else { std::cout << "FAIL" << std::endl; return 1; }
}
