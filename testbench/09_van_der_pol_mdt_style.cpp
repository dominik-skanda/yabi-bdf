/*
 * Van der Pol - MDT style (parameterized form)
 */
#include <yabi/bdf_integrator.hpp>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>

int main() {
    std::cout << "=== Test 9: Van der Pol mu=1000 (MDT style) ===" << std::endl;
    
    const unsigned int N = 2;
    const unsigned int P = 1;
    
    // [y0,y1, t, mu]
    CppAD::vector<CppAD::AD<double> > x(N + 1 + P);
    x[0] = 2.0; x[1] = 0.0; // y
    x[2] = 0.0;             // t
    x[3] = 1000.0;          // mu
    CppAD::Independent(x);
    
    CppAD::vector<CppAD::AD<double> > f(N);
    f[0] = x[1];
    f[1] = x[3] * (1.0 - x[0]*x[0]) * x[1] - x[0];
    
    CppAD::ADFun<double> rhs;
    rhs.Dependent(x, f);
    rhs.optimize();
    
    double y0[2] = {2.0, 0.0};
    double p[1]  = {1000.0};
    double abs_err[2] = {1e-10, 1e-8};
    double y[2];
    
    yabi::BDFIntegrator<double> integrator(rhs);
    if (!integrator.resetIntegration(y0, 0.0, p, 1.0, 1e-10, abs_err, 
                                     false, false, NULL, NULL, NULL, NULL, 0)) {
        std::cerr << "FAIL: resetIntegration" << std::endl;
        return 1;
    }
    // Arguments: max order, machine precision, land exactly on t_end,
    // error-weight index.
    if (!integrator.setAlgorithmSettings(5, std::numeric_limits<double>::epsilon(), true, 1)) {
        std::cerr << "FAIL: setAlgorithmSettings" << std::endl;
        return 1;
    }
    
    if (!integrator.performIntegration(3000.0, 10000000)) {
        std::cerr << "FAIL: performIntegration" << std::endl;
        return 1;
    }
    if (!integrator.getSolutionVector(y)) {
        std::cerr << "FAIL: getSolutionVector" << std::endl;
        return 1;
    }
    
    std::cout << std::setprecision(10) << "y1=" << y[0] << " y2=" << y[1] << std::endl;
    if (!(std::isfinite(y[0]) && std::isfinite(y[1]))) {
        std::cout << "NONFINITE" << std::endl;
        return 1;
    }
    std::cout << "FINITE: OK" << std::endl;

    // Reference at t=3000 (Radau, rtol=1e-12)
    const double ref[2] = {-1.51060694, 0.00117838};
    const double tol[2] = {1e-4, 1e-6};
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
