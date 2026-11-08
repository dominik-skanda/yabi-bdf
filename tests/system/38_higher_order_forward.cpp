// T40 — Higher-order forward (variational) sensitivity, d^k y / d tau^k.
//
// forwardOrder(k, seed) returns the k-th-order directional derivative in the
// seed direction, computed stably for long/stiff problems — the stable
// complement to the discrete forward(k) (IND), which is A-stability-bound at
// order >= 3.  Verified against exact solutions, cross-checked against the
// discrete forward(3), and against secondOrderForward for k = 2.
#include <yabi/bdf_integrator.hpp>
#include <cppad/cppad.hpp>
#include <cmath>
#include <cstdio>

static int fails = 0;
static void check(const char *name, double got, double exact, double tol = 1e-6) {
    double rel = std::fabs(got - exact) / (1.0 + std::fabs(exact));
    std::printf("  %-24s got=%.10g exact=%.10g rel=%.2e%s\n", name, got, exact, rel,
                (rel > tol ? "  <-- FAIL" : ""));
    if (rel > tol) fails++;
}

// y' = y^2: d^k y/dy0^k = k! t^{k-1} / (1 - y0 t)^{k+1}.
static void scalar() {
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(1);
    x[0] = 1.0;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(1);
    f[0] = x[0] * x[0];
    rhs.Dependent(x, f);
    rhs.optimize();

    double y0[1] = {1.0}, ab[1] = {1e-12}, s[1] = {1.0}, d[1];
    yabi::BDFIntegrator<double> it(rhs);
    if (!it.resetIntegration(y0, 0.0, NULL, 0.1, 1e-12, ab, false, false, NULL, NULL, NULL, NULL,
                             0) ||
        !it.performIntegration(0.5, 100000)) {
        std::printf("  scalar: integration FAILED\n");
        fails++;
        return;
    }
    const double T = 0.5;
    for (int k = 1; k <= 4; k++) {
        if (!it.forwardOrder(k, s, d)) {
            std::printf("  scalar: forwardOrder(%d) FAILED\n", k);
            fails++;
            continue;
        }
        double exact = 1.0;
        for (int i = 2; i <= k; i++) exact *= (double)i; // k!
        exact *= std::pow(T, k - 1) / std::pow(1.0 - T, k + 1);
        char name[64];
        std::snprintf(name, sizeof(name), "y'=y^2 d^%d y/dy0^%d", k, k);
        check(name, d[0], exact);
    }
}

// Coupled y1'=y1^2, y2'=y1*y2, seed s=(1,0): d y1/da^k = k! t^{k-1}/(1-at)^{k+1},
// d y2/da^k = k! b t^k/(1-at)^{k+1}.
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

    double y0[2] = {a, b}, ab[2] = {1e-13, 1e-13}, s[2] = {1.0, 0.0}, d[2];
    yabi::BDFIntegrator<double> it(rhs);
    if (!it.resetIntegration(y0, 0.0, NULL, 0.01, 1e-13, ab, false, false, NULL, NULL, NULL, NULL,
                             0) ||
        !it.performIntegration(T, 100000)) {
        std::printf("  coupled: integration FAILED\n");
        fails++;
        return;
    }
    const double e = 1.0 - a * T;
    for (int k = 1; k <= 4; k++) {
        if (!it.forwardOrder(k, s, d)) {
            std::printf("  coupled: forwardOrder(%d) FAILED\n", k);
            fails++;
            continue;
        }
        double fact = 1.0;
        for (int i = 2; i <= k; i++) fact *= (double)i; // k!
        char name[64];
        std::snprintf(name, sizeof(name), "coupled y1 d^%d/da^%d", k, k);
        check(name, d[0], fact * std::pow(T, k - 1) / std::pow(e, k + 1));
        std::snprintf(name, sizeof(name), "coupled y2 d^%d/da^%d", k, k);
        check(name, d[1], fact * b * std::pow(T, k) / std::pow(e, k + 1));
    }

    // Cross-check forwardOrder(3) vs the discrete forward(3) (IND, stable here).
    yabi::BDFIntegrator<double> it2(rhs);
    it2.resetIntegration(y0, 0.0, NULL, 0.01, 1e-13, ab, true, false, NULL, NULL, NULL, NULL, 0);
    it2.performIntegration(T, 100000);
    it2.forward(1, s, NULL, true);
    it2.forward(2, s, NULL, false);
    it2.forward(3, s, NULL, false);
    double f3[2];
    it2.getForwardSolutionVector(f3);
    it.forwardOrder(3, s, d);
    check("forwardOrder(3)==forward(3) y1", d[0], f3[0]);
    check("forwardOrder(3)==forward(3) y2", d[1], f3[1]);

    // forwardOrder(2) coincides with secondOrderForward.
    double s2[2];
    it.secondOrderForward(s, s2);
    it.forwardOrder(2, s, d);
    check("forwardOrder(2)==secondOrderForward y1", d[0], s2[0]);
    check("forwardOrder(2)==secondOrderForward y2", d[1], s2[1]);
}

int main() {
    std::printf("=== T40 Higher-order forward sensitivity ===\n");
    scalar();
    coupled();
    std::printf("FAILS=%d\n", fails);
    return fails ? 1 : 0;
}
