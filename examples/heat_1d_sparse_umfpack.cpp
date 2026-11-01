// Example: 1D heat equation (semi-discretized) with UMFPACK sparse solver
#include <yabi/bdf_integrator.hpp>
#include <yabi/detail/umfpack_linear_solver.ipp>
#include <cppad/cppad.hpp>
#include <iostream>
#include <vector>
#include <cmath>

int main() {
    const int N = 50;
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
    yabi::UmfpackLinearSolver<double> *ls = new yabi::UmfpackLinearSolver<double>();
    integrator.setLinearSolver(ls);

    std::vector<double> y0(N);
    for (int i = 0; i < N; i++) y0[i] = sin(M_PI * (i+1)*dx);
    double *abs_err = new double[N];
    for (int i = 0; i < N; i++) abs_err[i] = 1e-8;
    if (!integrator.resetIntegration(&y0[0], 0.1, 1e-6, abs_err, false, false, NULL,NULL,NULL,NULL,0)) return 1;
    if (!integrator.setAlgorithmSettings(5, 1e-16, true, 1)) return 1;
    bool ok = integrator.performIntegration(0.1, 100000);
    if (!ok) { std::cout << "FAIL" << std::endl; delete[] abs_err; return 1; }
    std::vector<double> y(N);
    integrator.getSolutionVector(&y[0]);
    std::cout << "Heat1D UMFPACK: y[0]=" << y[0] << " y[" << N/2 << "]=" << y[N/2] << " PASS" << std::endl;
    delete[] abs_err;
    return 0;
}
