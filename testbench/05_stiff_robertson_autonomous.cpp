/*
 * Test 5: Robertson's problem - autonomous form
 * dy1/dt = -0.04*y1 + 1e4*y2*y3
 * dy2/dt = 0.04*y1 - 1e4*y2*y3 - 3e7*y2^2
 * dy3/dt = 3e7*y2^2
 * y(0) = (1,0,0)
 * Reference at t=1e4 (Radau, rtol=1e-11):
 *   y = (0.107300429, 4.800167e-07, 0.892699091)
 */
#include <yabi/bdf_integrator.hpp>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>

int main() {
    std::cout << "=== Test 5: Robertson (autonomous) ===" << std::endl;
    
    CppAD::vector<CppAD::AD<double> > x(3);
    x[0] = 1.0; x[1] = 0.0; x[2] = 0.0;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double> > f(3);
    double k1 = 0.04, k2 = 1.0e4, k3 = 3.0e7;
    f[0] = -k1*x[0] + k2*x[1]*x[2];
    f[1] =  k1*x[0] - k2*x[1]*x[2] - k3*x[1]*x[1];
    f[2] =                    k3*x[1]*x[1];
    CppAD::ADFun<double> rhs;
    rhs.Dependent(x, f);
    rhs.optimize();
    
    yabi::BDFIntegrator<double> integrator(rhs);
    double y0[3] = {1.0, 0.0, 0.0};
    double abs_err[3] = {1e-12, 1e-18, 1e-12};
    double y[3];
    
    if (!integrator.resetIntegration(y0, 100.0, 1e-10, abs_err, false, false, NULL,NULL,NULL,NULL,0)) {
        std::cerr << "FAIL: reset" << std::endl; return 1;
    }
    // Arguments: max order, machine precision, land exactly on t_end,
    // error-weight index.
    if (!integrator.setAlgorithmSettings(5, std::numeric_limits<double>::epsilon(), true, 1)) {
        std::cerr << "FAIL: setAlgorithmSettings" << std::endl; return 1;
    }
    if (!integrator.performIntegration(1e4, 10000000)) {
        std::cerr << "FAIL: integrate" << std::endl; return 1;
    }
    if (!integrator.getSolutionVector(y)) {
        std::cerr << "FAIL: get" << std::endl; return 1;
    }
    
    std::cout << std::setprecision(12);
    std::cout << "y1=" << y[0] << " y2=" << y[1] << " y3=" << y[2] << " sum=" << (y[0]+y[1]+y[2]) << std::endl;

    const double ref[3] = {0.107300429, 4.800167e-07, 0.892699091};
    const double tol[3] = {1e-4, 1e-9, 1e-4};

    // Check conservation
    if (std::fabs(y[0]+y[1]+y[2] - 1.0) < 1e-6) {
        std::cout << "MASS_CONSERVED" << std::endl;
    } else {
        std::cout << "MASS_NOT_CONSERVED" << std::endl;
        return 1;
    }

    // Check against the reference solution
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
