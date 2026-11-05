/*
 * Test 2: Linear system - simple 2x2
 * dy1/dt = -y1 + 2*y2, dy2/dt = -2*y1 - y2
 */
#include <yabi/bdf_integrator.hpp>
#include <cmath>
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "=== Test 2: Linear 2x2 system ===" << std::endl;
    
    CppAD::vector<CppAD::AD<double> > x(2);
    x[0] = 1.0; x[1] = 0.0;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double> > f(2);
    f[0] = -x[0] + 2.0*x[1];
    f[1] = -2.0*x[0] - x[1];
    CppAD::ADFun<double> rhs;
    rhs.Dependent(x, f);
    rhs.optimize();
    
    yabi::BDFIntegrator<double> integrator(rhs);
    double y0[2] = {1.0, 0.0};
    double abs_err[2] = {1e-14, 1e-14};
    double y[2];
    
    if (!integrator.resetIntegration(y0, 0.1, 1e-12, abs_err, false, false, NULL,NULL,NULL,NULL,0)) {
        std::cerr << "FAIL: reset" << std::endl; return 1;
    }
    if (!integrator.performIntegration(1.0, 100000)) {
        std::cerr << "FAIL: integrate" << std::endl; return 1;
    }
    if (!integrator.getSolutionVector(y)) {
        std::cerr << "FAIL: get" << std::endl; return 1;
    }
    
    // At t=1: y1 = exp(-1)*cos(2), y2 = -exp(-1)*sin(2)
    double exact1 = exp(-1.0)*cos(2.0);
    double exact2 = -exp(-1.0)*sin(2.0);
    double err1 = fabs(y[0]-exact1), err2 = fabs(y[1]-exact2);
    std::cout << "y1(1)=" << y[0] << " exact=" << exact1 << " err=" << err1 << std::endl;
    std::cout << "y2(1)=" << y[1] << " exact=" << exact2 << " err=" << err2 << std::endl;
    if (err1 < 1e-9 && err2 < 1e-9) { std::cout << "PASS" << std::endl; return 0; }
    else { std::cout << "FAIL" << std::endl; return 1; }
}
