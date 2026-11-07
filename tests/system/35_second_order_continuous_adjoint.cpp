// T26 — Second-order continuous adjoint (Hessian-vector product, stable).
//
// secondOrderAdjoint(w, s) returns H g * s, the Hessian of g = w^T y(t_end)
// contracted with the direction s, computed stably for long/stiff problems —
// the stable complement to the discrete-adjoint reverse(2) (IND), which is
// A-stability-bound.  Verified against exact solutions and, where IND is
// stable, cross-checked against reverse(2).
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

// y' = y^2, g = y(T): d^2 g/dy0^2 = 2T/(1 - y0 T)^3.
static void scalar() {
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(1);
    x[0] = 1.0;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(1);
    f[0] = x[0] * x[0];
    rhs.Dependent(x, f);
    rhs.optimize();

    double y0[1] = {1.0}, ab[1] = {1e-12}, w[1] = {1.0}, s[1] = {1.0}, H[1];
    yabi::BDFIntegrator<double> it(rhs);
    if (!it.resetIntegration(y0, 0.0, NULL, 0.1, 1e-12, ab, false, false, NULL, NULL, NULL, NULL,
                             0) ||
        !it.performIntegration(0.5, 100000) || !it.secondOrderAdjoint(w, s, H)) {
        std::printf("  scalar: secondOrderAdjoint FAILED\n");
        fails++;
        return;
    }
    check("y'=y^2 d^2g/dy0^2", H[0], 2.0 * 0.5 / std::pow(1.0 - 0.5, 3.0)); // 8
}

// Coupled nonlinear y1'=y1^2, y2'=y1*y2, g=y2 (w=(0,1)), seed s=(1,0).
// d2 y2/da^2 = 2 b t^2/(1-at)^3, d2 y2/(da db) = t/(1-at)^2; cross-check reverse(2).
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

    double y0[2] = {a, b}, ab[2] = {1e-13, 1e-13}, w[2] = {0.0, 1.0}, s[2] = {1.0, 0.0}, H[2];
    yabi::BDFIntegrator<double> it(rhs);
    if (!it.resetIntegration(y0, 0.0, NULL, 0.01, 1e-13, ab, false, false, NULL, NULL, NULL, NULL,
                             0) ||
        !it.performIntegration(T, 100000) || !it.secondOrderAdjoint(w, s, H)) {
        std::printf("  coupled: secondOrderAdjoint FAILED\n");
        fails++;
        return;
    }
    check("coupled d2y2/da2", H[0], 2.0 * b * T * T / std::pow(1 - a * T, 3.0)); // 3
    check("coupled d2y2/(da db)", H[1], T / std::pow(1 - a * T, 2.0));           // 1

    // Cross-check vs reverse(2) (IND, stable at this short interval).
    yabi::BDFIntegrator<double> it2(rhs);
    it2.resetIntegration(y0, 0.0, NULL, 0.01, 1e-13, ab, true, false, NULL, NULL, NULL, NULL, 0);
    it2.performIntegration(T, 100000);
    it2.forward(1, s, NULL, true); it2.forward(1, s, NULL, false);
    it2.forward(2, s, NULL, true); it2.forward(2, s, NULL, false);
    double *adj[2];
    adj[0] = new double[2]; adj[1] = new double[2];
    if (it2.reverse(2, w)) it2.getReverseSolutionVector(adj, NULL);
    check("reverse(2) adj[0][1]", adj[0][1], H[0]);
    check("reverse(2) adj[1][1]", adj[1][1], H[1]);
    delete[] adj[0]; delete[] adj[1];
}

// Linear oscillator y1'=y2, y2'=-y1: g = y1(T) is linear in y0, so H = 0.
static void linear() {
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(2);
    x[0] = 1.0; x[1] = 0.0;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(2);
    f[0] = x[1];
    f[1] = -x[0];
    rhs.Dependent(x, f);
    rhs.optimize();

    double y0[2] = {1.0, 0.0}, ab[2] = {1e-12, 1e-12}, w[2] = {1.0, 0.0}, s[2] = {1.0, 0.0}, H[2];
    yabi::BDFIntegrator<double> it(rhs);
    if (!it.resetIntegration(y0, 0.0, NULL, 0.1, 1e-12, ab, false, false, NULL, NULL, NULL, NULL,
                             0) ||
        !it.performIntegration(1.0, 100000) || !it.secondOrderAdjoint(w, s, H)) {
        std::printf("  linear: secondOrderAdjoint FAILED\n");
        fails++;
        return;
    }
    check("linear H*s[0]", H[0], 0.0);
    check("linear H*s[1]", H[1], 0.0);
}

int main() {
    std::printf("=== T26 Second-order continuous adjoint ===\n");
    scalar();
    coupled();
    linear();
    std::printf("FAILS=%d\n", fails);
    return fails ? 1 : 0;
}
