// T11f — Polanyi-Wigner / semi-discretized PDE (20-node, sparse large-system)
#include <yabi/bdf_integrator.hpp>
#include <cppad/cppad.hpp>
#include <cmath>
#include <iostream>
#include <limits>

int main() {
    const int n = 20;
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(n);
    for(int i=0;i<n;i++) x[i] = 1.0 + 0.01*i; // initial profile
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(n);
    for(int i=0;i<n;i++) {
        CppAD::AD<double> left = (i>0) ? x[i-1] : CppAD::AD<double>(0.0);
        CppAD::AD<double> right = (i<n-1) ? x[i+1] : CppAD::AD<double>(0.0);
        f[i] = 0.01*(left - 2.0*x[i] + right) - 0.1*x[i]; // diffusion + decay
    }
    rhs.Dependent(x, f);
    rhs.optimize();

    yabi::BDFIntegrator<double> integrator(rhs);
    double y0[20]; for(int i=0;i<20;i++) y0[i]=1.0+0.01*i;
    double abs_err[20]; for(int i=0;i<20;i++) abs_err[i]=1e-6;
    double y[20];

    if (!integrator.resetIntegration(y0, 10.0, 1e-6, abs_err, false, false, NULL,NULL,NULL,NULL,0)) return 1;
    if (!integrator.setAlgorithmSettings(5, std::numeric_limits<double>::epsilon(), true, 1)) return 1;
    bool ok = integrator.performIntegration(10.0, 1000000);
    if (!ok) { std::cout << "FAIL (ok=false)" << std::endl; return 1; }
    if (!integrator.getSolutionVector(y)) { std::cout << "FAIL get" << std::endl; return 1; }
    std::cout << "Polanyi-Wigner n=20 t=10 => y0=" << y[0] << " y19=" << y[19] << std::endl;
    std::cout << "PASS" << std::endl;
    return 0;
}
