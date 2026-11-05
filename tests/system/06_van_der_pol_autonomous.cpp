/*
 * Test 6: Van der Pol oscillator - autonomous
 * y1' = y2, y2' = mu*(1-y1^2)*y2 - y1, y(0)=(2,0), mu=1000
 */
#include <yabi/bdf_integrator.hpp>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>

int main() {
    std::cout << "=== Test 6: Van der Pol (mu=1000, autonomous) ===" << std::endl;
    
    CppAD::vector<CppAD::AD<double> > x(2);
    x[0] = 2.0; x[1] = 0.0;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double> > f(2);
    double mu = 1000.0;
    f[0] = x[1];
    f[1] = mu*(1.0 - x[0]*x[0])*x[1] - x[0];
    CppAD::ADFun<double> rhs;
    rhs.Dependent(x, f);
    rhs.optimize();
    
    yabi::BDFIntegrator<double> integrator(rhs);
    double y0[2] = {2.0, 0.0};
    double abs_err[2] = {1e-10, 1e-8};
    double y[2];
    
    if (!integrator.resetIntegration(y0, 1.0, 1e-10, abs_err, false, false, NULL,NULL,NULL,NULL,0)) {
        std::cerr << "FAIL: reset" << std::endl; return 1;
    }
    // Arguments: max order, machine precision, land exactly on t_end,
    // error-weight index.
    if (!integrator.setAlgorithmSettings(5, std::numeric_limits<double>::epsilon(), true, 1)) {
        std::cerr << "FAIL: setAlgorithmSettings" << std::endl; return 1;
    }
    if (!integrator.performIntegration(3000.0, 10000000)) {
        std::cerr << "FAIL: integrate" << std::endl; return 1;
    }
    if (!integrator.getSolutionVector(y)) {
        std::cerr << "FAIL: get" << std::endl; return 1;
    }
    
    std::cout << std::setprecision(10) << "y1=" << y[0] << " y2=" << y[1] << std::endl;

    // Reference at t=3000 (Radau, rtol=1e-12)
    const double ref[2] = {-1.51060694, 0.00117838};
    const double tol[2] = {1e-4, 1e-6};
    if (!(std::isfinite(y[0]) && std::isfinite(y[1]))) {
        std::cout << "NONFINITE" << std::endl;
        return 1;
    }
    for (int i = 0; i < 2; ++i) {
        if (std::fabs(y[i] - ref[i]) > tol[i]) {
            std::cout << "ACCURACY_FAIL: component " << i
                      << " got " << y[i] << " expected " << ref[i] << std::endl;
            return 1;
        }
    }
    std::cout << "ACCURACY: OK" << std::endl;
    std::cout << "PASS" << std::endl;
    return 0;
}
