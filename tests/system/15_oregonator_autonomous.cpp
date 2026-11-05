// T11d — Oregonator (Belousov-Zhabotinsky, 3-state oscillatory stiff)
//
// FIX (2026-11-01): the step controller previously reduced only the step size on rejection,
// never the order, so the high-order BDF became stuck across the sharp relaxation-oscillation
// transition (y0 collapses 1.1e5 -> ~1 between t=1 and t=5) and aborted with "too much steps
// rejected". The solver now reduces the order on repeated rejection (newton.ipp), so this
// benchmark runs at the default MAXORDER=6. Verified against scipy Radau (rtol=1e-10) at
// t=1, t=5 and t=10.
#include <yabi/bdf_integrator.hpp>
#include <cppad/cppad.hpp>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>

static bool run(double T, const double* ref, double rel_tol) {
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
    double abs_err[3] = {1e-10, 1e-10, 1e-10};
    double y[3];

    if (!integrator.resetIntegration(y0, 1.0, 1e-10, abs_err, false, false, NULL,NULL,NULL,NULL,0)) return false;
    if (!integrator.setAlgorithmSettings(6, std::numeric_limits<double>::epsilon(), true, 1)) return false; // default MAXORDER
    if (!integrator.performIntegration(T, 10000000)) { std::cout << "t=" << T << " => FAIL (integration)" << std::endl; return false; }
    if (!integrator.getSolutionVector(y)) return false;

    bool ok = true;
    for (int i = 0; i < 3; i++) {
        double rel = std::fabs(y[i] - ref[i]) / (1.0 + std::fabs(ref[i]));
        if (rel > rel_tol) ok = false;
    }
    std::cout << std::setprecision(10);
    std::cout << "t=" << T << " => y=(" << y[0] << "," << y[1] << "," << y[2] << ") " << (ok ? "OK" : "MISMATCH") << std::endl;
    return ok;
}

int main() {
    // scipy Radau rtol=1e-10 references (independent cross-check)
    const double ref1[3]  = {1.1255947590e+05, 5.9527208434e-02, 6.7128954331e+03};
    const double ref5[3]  = {1.0017533392e+00, 5.7134218980e+02, 2.3962699156e+04};
    const double ref10[3] = {1.0006705243e+00, 1.4923585398e+03, 1.0713986894e+04};
    bool ok = true;
    ok &= run(1.0,  ref1,  1e-4);
    ok &= run(5.0,  ref5,  1e-4);
    ok &= run(10.0, ref10, 1e-4);
    std::cout << (ok ? "PASS" : "FAIL") << std::endl;
    return ok ? 0 : 1;
}
