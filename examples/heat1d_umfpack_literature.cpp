// Heat equation vs literature (analytic) with UMFPACK sparse solver
#include <yabi/bdf_integrator.hpp>
#include <yabi/detail/umfpack_linear_solver.ipp>
#include <cppad/cppad.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>

int main() {
    const int N = 50;
    const double alpha = 0.01;
    const double dx = 1.0/(N+1);
    const double t_end = 0.01;
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
    std::vector<double> abs_err(N, 1e-10);
    if (!integrator.resetIntegration(&y0[0], t_end, 1e-9, &abs_err[0], false, false, NULL,NULL,NULL,NULL,0)) return 1;
    if (!integrator.setAlgorithmSettings(5, 1e-16, true, 1)) return 1;
    if (!integrator.performIntegration(t_end, 100000)) { std::cout << "FAIL" << std::endl; return 1; }
    std::vector<double> y(N);
    integrator.getSolutionVector(&y[0]);
    double xmid = 0.5;
    double u_analytic = exp(-alpha*M_PI*M_PI*t_end)*sin(M_PI*xmid);
    printf("Heat1D UMFPACK: y_mid=%g analytic=%g rel_err=%g\n", y[N/2], u_analytic, fabs(y[N/2]-u_analytic)/fabs(u_analytic+1e-18));
    return 0;
}
