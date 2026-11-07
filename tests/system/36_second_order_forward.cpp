// T27 — Second-order forward sensitivity (d^2 y / d tau^2, stable).
//
// secondOrderForward(seed) returns the second-order directional derivative
// d^2 y / d tau^2 in the seed direction, computed stably for long/stiff
// problems — the stable complement to forward(2) (IND, A-stability-bound).
// Verified against exact solutions and cross-checked vs forward(2).
#include <yabi/bdf_integrator.hpp>
#include <cppad/cppad.hpp>
#include <cmath>
#include <cstdio>

static int fails = 0;
static void check(const char *name, double got, double exact, double tol = 1e-6) {
    double rel = std::fabs(got - exact) / (1.0 + std::fabs(exact));
    std::printf("  %-22s got=%.10g exact=%.10g rel=%.2e%s\n", name, got, exact, rel,
                (rel > tol ? "  <-- FAIL" : ""));
    if (rel > tol) fails++;
}

// y' = y^2: d^2 y/dy0^2 = 2T/(1 - y0 T)^3.
static void scalar() {
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(1);
    x[0] = 1.0;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(1);
    f[0] = x[0] * x[0];
    rhs.Dependent(x, f);
    rhs.optimize();

    double y0[1] = {1.0}, ab[1] = {1e-12}, s[1] = {1.0}, d2[1];
    yabi::BDFIntegrator<double> it(rhs);
    if (!it.resetIntegration(y0, 0.0, NULL, 0.1, 1e-12, ab, false, false, NULL, NULL, NULL, NULL,
                             0) ||
        !it.performIntegration(0.5, 100000) || !it.secondOrderForward(s, d2)) {
        std::printf("  scalar: secondOrderForward FAILED\n");
        fails++;
        return;
    }
    check("y'=y^2 d^2y/dy0^2", d2[0], 2.0 * 0.5 / std::pow(1.0 - 0.5, 3.0)); // 8
}

// Coupled nonlinear y1'=y1^2, y2'=y1*y2, seed s=(1,0): d^2y1/da^2 = 2t/(1-at)^3,
// d^2y2/da^2 = 2 b t^2/(1-at)^3; cross-check forward(2).
static void coupled() {
    double a = 2.0, b = 3.0, T = 0.25;
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(2);
    x[0] = a; x[1] = b;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(2);
    f[0] = x[0] * x[0];
    f[1] = x[0] * x[1];
    rhs.Dependent(x, f);
    rhs.optimize();

    double y0[2] = {a, b}, ab[2] = {1e-13, 1e-13}, s[2] = {1.0, 0.0}, d2[2];
    yabi::BDFIntegrator<double> it(rhs);
    if (!it.resetIntegration(y0, 0.0, NULL, 0.01, 1e-13, ab, false, false, NULL, NULL, NULL, NULL,
                             0) ||
        !it.performIntegration(T, 100000) || !it.secondOrderForward(s, d2)) {
        std::printf("  coupled: secondOrderForward FAILED\n");
        fails++;
        return;
    }
    check("coupled d^2y1/da^2", d2[0], 2.0 * T / std::pow(1 - a * T, 3.0));        // 4
    check("coupled d^2y2/da^2", d2[1], 2.0 * b * T * T / std::pow(1 - a * T, 3.0)); // 3

    // Cross-check vs forward(2) (IND, stable at this short interval).
    yabi::BDFIntegrator<double> it2(rhs);
    it2.resetIntegration(y0, 0.0, NULL, 0.01, 1e-13, ab, true, false, NULL, NULL, NULL, NULL, 0);
    it2.performIntegration(T, 100000);
    it2.forward(1, s, NULL, true); it2.forward(1, s, NULL, false);
    it2.forward(2, s, NULL, true); it2.forward(2, s, NULL, false);
    double f2[2];
    it2.getForwardSolutionVector(f2);
    check("forward(2) d^2y1/da^2", f2[0], d2[0]);
    check("forward(2) d^2y2/da^2", f2[1], d2[1]);
}

int main() {
    std::printf("=== T27 Second-order forward sensitivity ===\n");
    scalar();
    coupled();
    std::printf("FAILS=%d\n", fails);
    return fails ? 1 : 0;
}
