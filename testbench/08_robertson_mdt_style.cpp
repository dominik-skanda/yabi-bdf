/*
 * Robertson - MDT/sensitivity_demo style (parameterized form)
 * This matches the calling convention this integrator was designed for
 */
#include <yabi/bdf_integrator.hpp>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>

int main() {
    std::cout << "=== Test 8: Robertson (MDT style, parameterized) ===" << std::endl;
    
    const unsigned int N = 3;
    const unsigned int P = 3;
    
    // [y0,y1,y2, t, k1,k2,k3]
    CppAD::vector<CppAD::AD<double> > x(N + 1 + P);
    x[0] = 1.0; x[1] = 0.0; x[2] = 0.0;
    x[3] = 0.0;
    x[4] = 0.04; x[5] = 1.0e4; x[6] = 3.0e7;
    CppAD::Independent(x);
    
    CppAD::vector<CppAD::AD<double> > f(N);
    f[0] = -x[4]*x[0] + x[5]*x[1]*x[2];
    f[1] =  x[4]*x[0] - x[5]*x[1]*x[2] - x[6]*x[1]*x[1];
    f[2] =                        x[6]*x[1]*x[1];
    
    CppAD::ADFun<double> rhs;
    rhs.Dependent(x, f);
    rhs.optimize();
    
    double y0[3] = {1.0, 0.0, 0.0};
    double p[3]  = {0.04, 1.0e4, 3.0e7};
    double abs_err[3] = {1.0e-12, 1.0e-18, 1.0e-12};
    double y[3];
    
    yabi::BDFIntegrator<double> integrator(rhs);
    if (!integrator.resetIntegration(y0, 0.0, p, 100.0, 1e-10, abs_err, 
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
    
    if (!integrator.performIntegration(1.0e4, 10000000)) {
        std::cerr << "FAIL: performIntegration" << std::endl;
        return 1;
    }
    if (!integrator.getSolutionVector(y)) {
        std::cerr << "FAIL: getSolutionVector" << std::endl;
        return 1;
    }
    
    std::cout << std::setprecision(12);
    std::cout << "y1=" << y[0] << " y2=" << y[1] << " y3=" << y[2] << " sum=" << (y[0]+y[1]+y[2]) << std::endl;
    
    // Check mass conservation
    if (std::fabs(y[0]+y[1]+y[2] - 1.0) >= 1e-6) {
        std::cout << "MASS_NOT_CONSERVED" << std::endl;
        return 1;
    }
    std::cout << "MASS_CONSERVED: OK" << std::endl;

    if (!(std::isfinite(y[0]) && std::isfinite(y[1]) && std::isfinite(y[2]))) {
        std::cout << "NONFINITE" << std::endl;
        return 1;
    }
    std::cout << "FINITE: OK" << std::endl;

    // Reference at t=1e4 (Radau, rtol=1e-11)
    const double ref[3] = {0.107300429, 4.800167e-07, 0.892699091};
    const double tol[3] = {1e-4, 1e-9, 1e-4};
    for (int i = 0; i < 3; ++i) {
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
