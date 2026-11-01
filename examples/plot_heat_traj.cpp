#include <yabi/bdf_integrator.hpp>
#include <cppad/cppad.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include <limits>

int main() {
    const int N = 100;
    const double alpha = 0.01;
    const double dx = 1.0/(N+1);
    const double t_end = 50.0;
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
    std::vector<double> y0(N);
    for (int i = 0; i < N; i++) y0[i] = sin(M_PI * (i+1)*dx);
    std::vector<double> abs_err(N, 1e-12);
    if (!integrator.resetIntegration(&y0[0], t_end, 1e-12, &abs_err[0], false, false, NULL,NULL,NULL,NULL,0)) return 1;
    if (!integrator.setAlgorithmSettings(5, std::numeric_limits<double>::epsilon(), true, 1)) return 1;

    std::cout << "t,u_mid" << std::endl;
    std::cout << "0," << y0[N/2] << std::endl;
    for (int k = 1; k <= 50; ++k) {
        double t = (double)k;
        if (!integrator.performIntegration(t, 1000000)) return 1;
        std::vector<double> y(N);
        integrator.getSolutionVector(&y[0]);
        std::cout << t << "," << y[N/2] << std::endl;
    }
    return 0;
}
