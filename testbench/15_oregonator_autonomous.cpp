// T11d — Oregonator (Belousov-Zhabotinsky, 3-state oscillatory stiff)
#include <yabi/bdf_integrator.hpp>
#include <cppad/cppad.hpp>
#include <cmath>
#include <iostream>
#include <limits>

int main() {
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(3);
    x[0] = 1.0; x[1] = 1.0; x[2] = 1.0;  // Hairer-Wanner Oregonator initial
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(3);
    f[0] = 77.27*(x[1] + x[0]*(1.0 - 8.375e-6*x[0] - x[1]));
    f[1] = (x[2] - (1.0 + x[0])*x[1])/77.27;
    f[2] = 0.161*(x[0] - x[2]);
    rhs.Dependent(x, f);
    rhs.optimize();

    yabi::BDFIntegrator<double> integrator(rhs);
    double y0[3] = {1.0, 1.0, 1.0};
    double abs_err[3] = {1e-10,1e-10,1e-10};
    double y[3];

    if (!integrator.resetIntegration(y0, 1.0, 1e-10, abs_err, false, false, NULL,NULL,NULL,NULL,0)) return 1;
    if (!integrator.setAlgorithmSettings(5, std::numeric_limits<double>::epsilon(), true, 1)) return 1;
    bool ok = integrator.performIntegration(1.0, 5000000);
    if (!ok) { std::cout << "FAIL" << std::endl; return 1; }
    if (!integrator.getSolutionVector(y)) { std::cout << "FAIL get" << std::endl; return 1; }
    std::cout << "Oregonator t=1 => y = (" << y[0] << "," << y[1] << "," << y[2] << ")" << std::endl;
    std::cout << "PASS" << std::endl;
    return 0;
}
