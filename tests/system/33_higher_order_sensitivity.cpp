// T12i — Arbitrary-order (3+) forward and reverse sensitivities on literature
// test problems, checked against exact analytical derivatives.
//
// Three canonical nonlinear problems exercise every RHS class:
//   1. coupled nonlinear     y1' = y1^2,  y2' = y1*y2         (RHS_type 0)
//        exact: y1 = a/(1 - a t),  y2 = b/(1 - a t)
//   2. non-autonomous        y'  = e^t y^2                    (RHS_type 1)
//        exact: y = y0/(1 - y0 (e^t - 1))
//   3. parameter dependent   y'  = p y^2                      (RHS_type 2)
//        exact: y = y0/(1 - y0 p t)
//
// forward(order) returns the order-th directional derivative d^order y / d seed^order;
// reverse(order) returns the per-state/parameter adjoint coefficients of orders
// 1..order, aligned to the same pure-derivative convention (CppAD's Reverse weights
// the k-th Taylor coefficient by 1/k!, which the engine rescales by k!).
#include <yabi/bdf_integrator.hpp>
#include <cppad/cppad.hpp>
#include <cmath>
#include <cstdio>

static double factorial(unsigned n) {
    double f = 1.0;
    for (unsigned k = 2; k <= n; k++) f *= (double)k;
    return f;
}

static int fails = 0;

static void check(const char *name, double got, double exact) {
    double rel = std::fabs(got - exact) / (1.0 + std::fabs(exact));
    std::printf("  %-30s got=%.10g exact=%.10g rel=%.2e%s\n", name, got, exact, rel,
                (rel > 1e-6 ? "  <-- FAIL" : ""));
    if (rel > 1e-6) fails++;
}

// 1. Coupled nonlinear two-species system.  Seed the first species, observe the
//    second: forward d^q y2/da^q = b q! t^q / (1 - a t)^(q+1).  The adjoint of
//    state 0 is the same derivative; the adjoint of state 1 is the mixed
//    derivative d^q y2/(da^(q-1) db) = (q-1)! t^(q-1) / (1 - a t)^q.
static void coupled_nonlinear() {
    std::printf("coupled nonlinear: y1'=y1^2, y2'=y1*y2\n");
    const double a = 2.0, b = 3.0, t = 0.25;
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(2);
    x[0] = a; x[1] = b;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(2);
    f[0] = x[0] * x[0];
    f[1] = x[0] * x[1];
    rhs.Dependent(x, f);
    rhs.optimize();

    const double y0[2] = {a, b}, abs_err[2] = {1e-13, 1e-13};
    const double seed[2] = {1.0, 0.0}, w[2] = {0.0, 1.0};
    for (unsigned q = 1; q <= 4; q++) {
        yabi::BDFIntegrator<double> it(rhs);
        it.setMinTaylorCapacity(8);
        if (!it.resetIntegration(const_cast<double *>(y0), 0.0, NULL, 0.01, 1e-13,
                                 const_cast<double *>(abs_err), true, false, NULL, NULL, NULL,
                                 NULL, 0))
            std::abort();
        if (!it.performIntegration(t, 100000)) std::abort();

        it.forward(q, const_cast<double *>(seed), NULL, true);
        double d[2];
        it.getForwardSolutionVector(d);
        double ex = b * factorial(q) * std::pow(t, (double)q) /
                    std::pow(1.0 - a * t, (double)(q + 1));
        char name[64];
        std::snprintf(name, sizeof name, "forward d^%u y2/da", q);
        check(name, d[1], ex);

        double *adj[2];
        adj[0] = new double[q];
        adj[1] = new double[q];
        if (!it.reverse(q, const_cast<double *>(w))) { std::printf("  reverse FAILED\n"); fails++; }
        else {
            it.getReverseSolutionVector(adj, NULL);
            std::snprintf(name, sizeof name, "reverse d^%u g/da (g=y2)", q);
            check(name, adj[0][q - 1], ex);
            double ex_mixed = factorial(q - 1) * std::pow(t, (double)(q - 1)) /
                              std::pow(1.0 - a * t, (double)q);
            std::snprintf(name, sizeof name, "reverse d^%u g/(da^(q-1) db)", q);
            check(name, adj[1][q - 1], ex_mixed);
        }
        delete[] adj[0];
        delete[] adj[1];
    }
}

// 2. Non-autonomous system (RHS_type 1).
//    d^n y/dy0^n = n! (e^t - 1)^(n-1) / (1 - y0 (e^t - 1))^(n+1).
static void non_autonomous() {
    std::printf("non-autonomous: y'=e^t y^2\n");
    const double y0v = 1.5, t = 0.4;
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(2);
    x[0] = y0v; x[1] = 0.0;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(1);
    f[0] = CppAD::exp(x[1]) * x[0] * x[0];
    rhs.Dependent(x, f);
    rhs.optimize();

    const double B = std::exp(t) - 1.0;
    const double y0[1] = {y0v}, abs_err[1] = {1e-13};
    const double seed[1] = {1.0}, w[1] = {1.0};
    for (unsigned q = 1; q <= 4; q++) {
        yabi::BDFIntegrator<double> it(rhs);
        it.setMinTaylorCapacity(8);
        if (!it.resetIntegration(const_cast<double *>(y0), 0.0, NULL, 0.01, 1e-13,
                                 const_cast<double *>(abs_err), true, false, NULL, NULL, NULL,
                                 NULL, 0))
            std::abort();
        if (!it.performIntegration(t, 100000)) std::abort();

        it.forward(q, const_cast<double *>(seed), NULL, true);
        double d[1];
        it.getForwardSolutionVector(d);
        double ex = factorial(q) * std::pow(B, (double)(q - 1)) /
                    std::pow(1.0 - y0v * B, (double)(q + 1));
        char name[64];
        std::snprintf(name, sizeof name, "forward d^%u/dy0", q);
        check(name, d[0], ex);

        double *adj[1];
        adj[0] = new double[q];
        if (!it.reverse(q, const_cast<double *>(w))) { std::printf("  reverse FAILED\n"); fails++; }
        else {
            it.getReverseSolutionVector(adj, NULL);
            std::snprintf(name, sizeof name, "reverse d^%u/dy0", q);
            check(name, adj[0][q - 1], ex);
        }
        delete[] adj[0];
    }
}

// 3. Parameter-dependent system (RHS_type 2, Domain = 3: y, t, p).
//    d^n y/dp^n = n! y0^(n+1) t^n / (1 - y0 p t)^(n+1).
static void parameter_dependent() {
    std::printf("parameter: y'=p y^2\n");
    const double y0v = 2.0, p = 1.5, t = 0.2;
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(3);
    x[0] = y0v; x[1] = 0.0; x[2] = p;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(1);
    f[0] = x[2] * x[0] * x[0];
    rhs.Dependent(x, f);
    rhs.optimize();

    const double y0[1] = {y0v}, pp[1] = {p}, abs_err[1] = {1e-13};
    const double seed[1] = {0.0}, pseed[1] = {1.0}, w[1] = {1.0};
    for (unsigned q = 1; q <= 4; q++) {
        yabi::BDFIntegrator<double> it(rhs);
        it.setMinTaylorCapacity(8);
        if (!it.resetIntegration(const_cast<double *>(y0), 0.0, const_cast<double *>(pp), 0.01,
                                 1e-13, const_cast<double *>(abs_err), true, false, NULL, NULL,
                                 NULL, NULL, 0))
            std::abort();
        if (!it.performIntegration(t, 100000)) std::abort();

        it.forward(q, const_cast<double *>(seed), const_cast<double *>(pseed), true);
        double d[1];
        it.getForwardSolutionVector(d);
        double ex = factorial(q) * std::pow(y0v, (double)(q + 1)) * std::pow(t, (double)q) /
                    std::pow(1.0 - y0v * p * t, (double)(q + 1));
        char name[64];
        std::snprintf(name, sizeof name, "forward d^%u/dp", q);
        check(name, d[0], ex);

        double *adj[1], *adp[1];
        adj[0] = new double[q];
        adp[0] = new double[q];
        if (!it.reverse(q, const_cast<double *>(w))) { std::printf("  reverse FAILED\n"); fails++; }
        else {
            it.getReverseSolutionVector(adj, adp);
            std::snprintf(name, sizeof name, "reverse dp^%u", q);
            check(name, adp[0][q - 1], ex);
        }
        delete[] adj[0];
        delete[] adp[0];
    }
}

int main() {
    std::printf("=== T12i Arbitrary-order sensitivity (literature problems) ===\n");
    coupled_nonlinear();
    non_autonomous();
    parameter_dependent();
    std::printf("FAILS=%d\n", fails);
    return fails ? 1 : 0;
}
