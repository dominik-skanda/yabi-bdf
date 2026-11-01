// T11b — HIRES (verified coefficients from hwode hir_equation.f) — Hairer & Wanner
#include <yabi/bdf_integrator.hpp>
#include <cppad/cppad.hpp>
#include <cmath>
#include <iostream>
#include <limits>

int main() {
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(8);
    // Standard HIRES initial (Hairer-Wanner reference)
    x[0] = 1.0; x[1] = 0.0; x[2] = 0.0; x[3] = 0.0;
    x[4] = 0.0; x[5] = 0.0; x[6] = 0.0; x[7] = 0.0057;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(8);
    // Verified from hir_equation.f (cpmech/hwode, Hairer-Wanner §IV.4)
    f[0] = -1.71*x[0] + 0.43*x[1] + 8.32*x[2] + 0.0007;
    f[1] =  1.71*x[0] - 8.75*x[1];
    f[2] = -10.03*x[2] + 0.43*x[3] + 0.035*x[4];
    f[3] =  8.32*x[1] + 1.71*x[2] - 1.12*x[3];
    f[4] = -1.745*x[4] + 0.43*x[5] + 0.43*x[6];
    f[5] = -280.0*x[5]*x[7] + 0.69*x[3] + 1.71*x[4] - 0.43*x[5] + 0.69*x[6];
    f[6] =  280.0*x[5]*x[7] - 1.81*x[6];
    f[7] = -f[6];  // DY(8) = -DY(7)
    rhs.Dependent(x, f);
    rhs.optimize();

    yabi::BDFIntegrator<double> integrator(rhs);
    double y0[8] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057};
    double abs_err[8] = {1e-10,1e-10,1e-10,1e-10,1e-10,1e-10,1e-10,1e-10};
    double y[8];

    if (!integrator.resetIntegration(y0, 321.812, 1e-10, abs_err, false, false, NULL, NULL, NULL, NULL, 0)) {
        std::cerr << "FAIL: reset" << std::endl; return 1;
    }
    if (!integrator.setAlgorithmSettings(5, std::numeric_limits<double>::epsilon(), true, 1)) {
        std::cerr << "FAIL: settings" << std::endl; return 1;
    }
    bool ok = integrator.performIntegration(321.812, 10000000);
    if (!ok) {
        std::cout << "FAIL: integration (ok=false)" << std::endl; return 1;
    }
    if (!integrator.getSolutionVector(y)) {
        std::cerr << "FAIL: getSolution" << std::endl; return 1;
    }
    std::cout << std::setprecision(12);
    std::cout << "HIRES t=321.812 => y = (" << y[0] << "," << y[1] << "," << y[2] << "," << y[3] << ","
              << y[4] << "," << y[5] << "," << y[6] << "," << y[7] << ")" << std::endl;
    std::cout << "PASS" << std::endl;
    return 0;
}
