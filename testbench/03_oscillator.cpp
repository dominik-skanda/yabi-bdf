/*
 * Test 3: Harmonic oscillator y'' + y = 0 -> y1'=y2, y2'=-y1
 */
#include <yabi/bdf_integrator.hpp>
#include <cmath>
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "=== Test 3: Harmonic oscillator ===" << std::endl;
    
    CppAD::vector<CppAD::AD<double> > x(2);
    x[0] = 1.0; x[1] = 0.0;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double> > f(2);
    f[0] = x[1];
    f[1] = -x[0];
    CppAD::ADFun<double> rhs;
    rhs.Dependent(x, f);
    rhs.optimize();
    
    yabi::BDFIntegrator<double> integrator(rhs);
    double y0[2] = {1.0, 0.0};
    double abs_err[2] = {1e-12, 1e-12};
    double y[2];
    
    if (!integrator.resetIntegration(y0, 0.01, 1e-12, abs_err, false, false, NULL,NULL,NULL,NULL,0)) {
        std::cerr << "FAIL: reset" << std::endl; return 1;
    }
    if (!integrator.performIntegration(2*M_PI, 100000)) {
        std::cerr << "FAIL: integrate" << std::endl; return 1;
    }
    if (!integrator.getSolutionVector(y)) {
        std::cerr << "FAIL: get" << std::endl; return 1;
    }
    
    double err1 = fabs(y[0]-1.0), err2 = fabs(y[1]-0.0);
    std::cout << "y1(2pi)=" << y[0] << " (exact 1.0), err=" << err1 << std::endl;
    std::cout << "y2(2pi)=" << y[1] << " (exact 0.0), err=" << err2 << std::endl;
    if (err1 < 1e-8 && err2 < 1e-8) { std::cout << "PASS" << std::endl; return 0; }
    else { std::cout << "FAIL" << std::endl; return 1; }
}
