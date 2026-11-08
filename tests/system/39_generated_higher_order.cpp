// T40 — higher-order forward sensitivity using the generated derivative code.
//
// Cross-checks the decoupled tools/generate_derivatives.py output: the same
// ODE (y0'=y0^2 y1, y1'=y0 y1^2) is solved through the pure UserODERHS backend
// wired to the GENERATED f/jacobian/orderKDirectional code (generated_cubic.hpp),
// and its forwardOrder(k) is compared to the CppAD discrete forward(k).
#include <yabi/bdf_integrator.hpp>
#include <yabi/user_ode_rhs.hpp>

#include <cppad/cppad.hpp>

#include <cmath>
#include <cstdio>

#include "generated_cubic.hpp"

static int fails = 0;
static void check(const char *name, double got, double exact, double tol = 1e-6) {
    double rel = std::fabs(got - exact) / (1.0 + std::fabs(exact));
    std::printf("  %-22s pure=%.10g cppad=%.10g rel=%.2e%s\n", name, got, exact, rel,
                (rel > tol ? "  <-- FAIL" : ""));
    if (rel > tol) fails++;
}

int main() {
    std::printf("=== T40 generated derivatives (pure vs CppAD) ===\n");
    const double a = 2.0, b = 3.0, T = 0.02; // T well before the finite blow-up (~1/12)
    double seed[2] = {1.0, 0.0}, d[2];

    // Pure mode wired to the GENERATED code.
    yabi::UserODERHS<double> prhs(2, generated::f, generated::jacobian);
    prhs.setOrderKDirectional(generated::orderKDirectional);
    double y0[2] = {a, b}, ab[2] = {1e-12, 1e-12};
    yabi::BDFIntegrator<double> pit(prhs);
    if (!pit.resetIntegration(y0, 0.0, NULL, 0.01, 1e-12, ab, false, false, NULL, NULL, NULL, NULL,
                              0) ||
        !pit.performIntegration(T, 100000)) {
        std::printf("  pure integration FAILED\n");
        return 1;
    }

    // CppAD reference (discrete forward).
    CppAD::ADFun<double> crhs;
    CppAD::vector<CppAD::AD<double>> x(2);
    x[0] = a; x[1] = b;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(2);
    f[0] = x[0] * x[0] * x[1];
    f[1] = x[0] * x[1] * x[1];
    crhs.Dependent(x, f);
    crhs.optimize();
    yabi::BDFIntegrator<double> cit(crhs);
    cit.resetIntegration(y0, 0.0, NULL, 0.01, 1e-12, ab, true, false, NULL, NULL, NULL, NULL, 0);
    cit.performIntegration(T, 100000);

    for (int k = 1; k <= 3; k++) {
        if (!pit.forwardOrder(k, seed, d)) {
            std::printf("  forwardOrder(%d) FAILED\n", k);
            fails++;
            continue;
        }
        if (k == 1)
            cit.forward(1, seed, NULL, true);
        else
            cit.forward(k, seed, NULL, false);
        double fd[2];
        cit.getForwardSolutionVector(fd);
        char name[64];
        std::snprintf(name, sizeof(name), "forwardOrder(%d)[0]", k);
        check(name, d[0], fd[0]);
        std::snprintf(name, sizeof(name), "forwardOrder(%d)[1]", k);
        check(name, d[1], fd[1]);
    }

    std::printf("FAILS=%d\n", fails);
    return fails ? 1 : 0;
}
