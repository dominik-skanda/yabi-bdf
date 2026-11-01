// T12j — Prothero-Robinson: forward-mode AD vs central finite-difference cross-check
#include <yabi/bdf_integrator.hpp>
#include <cppad/cppad.hpp>
#include <cmath>
#include <iostream>
#include <limits>

int main() {
    std::cout << "=== T12j Prothero-Robinson: AD vs central-diff ===" << std::endl;
    constexpr unsigned N = 2;
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(N);
    x[0] = 0.5; x[1] = 0.0;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(N);
    f[0] = 2.0*(x[0] - x[0]*x[0]) - x[1];
    f[1] = x[0]*x[0] - x[1];
    rhs.Dependent(x, f); rhs.optimize();

    double y0[N] = {0.5, 0.0};
    double abs_err[N] = {1e-10, 1e-10};
    const double T = 10.0;
    double seed[N] = {1.0, 0.0};
    bool ok = true;

    // Forward-mode AD sensitivity
    double dir_AD[N];
    {
        yabi::BDFIntegrator<double> it(rhs);
        if (!it.resetIntegration(y0, 0.0, NULL, 0.1, 1e-10, abs_err, true, false, NULL,NULL,NULL,NULL,0) ||
            !it.performIntegration(T, 1000000) || !it.forward(1, seed, NULL, false)) { std::cerr << "FAIL AD\n"; return 1; }
        it.getForwardSolutionVector(dir_AD);
    }

    // Central finite-difference of the solution (2nd-order accurate in eps)
    const double eps = 1e-5;
    double yp[N], ym[N], yp_sol[N], ym_sol[N];
    for (unsigned i=0;i<N;i++) { yp[i] = y0[i] + eps*seed[i]; ym[i] = y0[i] - eps*seed[i]; }
    {
        yabi::BDFIntegrator<double> itp(rhs);
        if (!itp.resetIntegration(yp, 0.0, NULL, 0.1, 1e-10, abs_err, false, false, NULL,NULL,NULL,NULL,0) ||
            !itp.performIntegration(T, 1000000) || !itp.getSolutionVector(yp_sol)) { std::cerr << "FAIL FD+\n"; return 1; }
        yabi::BDFIntegrator<double> itm(rhs);
        if (!itm.resetIntegration(ym, 0.0, NULL, 0.1, 1e-10, abs_err, false, false, NULL,NULL,NULL,NULL,0) ||
            !itm.performIntegration(T, 1000000) || !itm.getSolutionVector(ym_sol)) { std::cerr << "FAIL FD-\n"; return 1; }
    }
    double dir_FD[N];
    for (unsigned i=0;i<N;i++) dir_FD[i] = (yp_sol[i] - ym_sol[i]) / (2.0*eps);

    std::cout << std::setprecision(12);
    for (unsigned i=0;i<N;i++) {
        double rel = std::fabs(dir_AD[i] - dir_FD[i]) / (1.0 + std::fabs(dir_FD[i]));
        std::cout << "dir[" << i << "] AD=" << dir_AD[i] << " FD=" << dir_FD[i] << " rel=" << rel << std::endl;
        if (rel > 1e-5) { std::cerr << "MISMATCH component " << i << std::endl; ok = false; }
    }

    std::cout << (ok ? "PASS" : "FAIL") << std::endl;
    return ok ? 0 : 1;
}
