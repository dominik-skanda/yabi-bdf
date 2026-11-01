// Sensitivity with UMFPACK: use same pattern as sensitivity_demo (dense-verified API)
#include <yabi/bdf_integrator.hpp>
#include <yabi/detail/umfpack_linear_solver.ipp>
#include <cppad/cppad.hpp>
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    const int N = 5;
    const double alpha = 0.01;
    const double dx = 1.0/(N+1);
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(N);
    for (int i = 0; i < N; i++) x[i] = sin(M_PI * (i+1)*dx);
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(N);
    for (int i = 0; i < N; i++) {
        CppAD::AD<double> uim1 = (i == 0) ? CppAD::AD<double>(0.0) : x[i-1];
        CppAD::AD<double> uip1 = (i == N-1) ? CppAD::AD<double>(0.0) : x[i+1];
        f[i] = alpha/(dx*dx) * (uim1 - 2.0*x[i] + uip1);
    }
    rhs.Dependent(x, f);
    rhs.optimize();

    yabi::BDFIntegrator<double> integrator(rhs);
    integrator.setLinearSolver(new yabi::UmfpackLinearSolver<double>());

    std::vector<double> y0(N);
    for (int i = 0; i < N; i++) y0[i] = sin(M_PI * (i+1)*dx);
    std::vector<double> abs_err(N, 1e-8);
    double yseed[5] = {0}; yseed[0] = 1.0;
    double pseed[5] = {0};
    if (!integrator.resetIntegration(&y0[0], 0.0, NULL, 0.01, 1e-8, &abs_err[0], true, false, NULL, NULL, NULL, NULL, 0)) return 1;
    if (!integrator.setAlgorithmSettings(5, 1e-16, true, 1)) return 1;
    if (!integrator.performIntegration(0.01, 100000)) { std::cout << "FAIL" << std::endl; return 1; }
    if (!integrator.forward(1, yseed, pseed, false)) { std::cout << "FAIL sens" << std::endl; return 1; }
    std::cout << "Heat1D UMFPACK sens PASS" << std::endl;
    return 0;
}
