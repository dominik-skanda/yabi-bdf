// T11g — Large parameterized chemical reaction network (6 species, sensitivity-ready)
#include <yabi/bdf_integrator.hpp>
#include <cppad/cppad.hpp>
#include <cmath>
#include <iostream>
#include <limits>

int main() {
    const int N = 6;
    const int P = 3;
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(6);
    x[0] = 1.0; x[1] = 0.0; x[2] = 0.0; x[3] = 0.0; x[4] = 0.0; x[5] = 0.0;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(N);
    f[0] = -0.04*x[0] + 1.0e4*x[1]*x[2];
    f[1] =  0.04*x[0] - 1.0e4*x[1]*x[2] - 3.0e7*x[1]*x[1];
    f[2] =  3.0e7*x[1]*x[1];
    f[3] =  3.0e7*x[1]*x[1] - 0.1*x[3];
    f[4] =  0.1*x[3] - 5.0*x[4];
    f[5] =  5.0*x[4] - 0.01*x[5];
    rhs.Dependent(x, f);
    rhs.optimize();
    yabi::BDFIntegrator<double> integrator(rhs);
    double y0[6] = {1.0,0.0,0.0,0.0,0.0,0.0};
    double abs_err[6] = {1e-10,1e-18,1e-10,1e-10,1e-10,1e-10};
    double y[6];
    if (!integrator.resetIntegration(y0, 100.0, 1e-10, abs_err, false, false, NULL,NULL,NULL,NULL,0)) return 1;
    if (!integrator.setAlgorithmSettings(5, std::numeric_limits<double>::epsilon(), true, 1)) return 1;
    bool ok = integrator.performIntegration(100.0, 1000000);
    if (!ok) { std::cout << "FAIL" << std::endl; return 1; }
    if (!integrator.getSolutionVector(y)) return 1;
    std::cout << "Network t=100 => y0=" << y[0] << " y5=" << y[5] << std::endl;
    std::cout << "PASS" << std::endl;
    return 0;
}
