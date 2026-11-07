// T25 — First-order continuous adjoint (stable reverse sensitivity).
//
// The continuous adjoint integrates lambda' = -J(y)^T lambda backward from
// t_end, giving d g / d y0 for g = w^T y(t_end) stably on long/stiff problems —
// the stable complement to the discrete-adjoint reverse() (IND), which is
// A-stability-bound (see T24).  Verified against exact solutions and, where
// IND is stable, cross-checked against reverse(1).
#include <yabi/bdf_integrator.hpp>
#include <cppad/cppad.hpp>
#include <cmath>
#include <cstdio>

static int fails = 0;
static void check(const char *name, double got, double exact, double tol = 1e-7) {
    double rel = std::fabs(got - exact) / (1.0 + std::fabs(exact));
    std::printf("  %-28s got=%.10g exact=%.10g rel=%.2e%s\n", name, got, exact, rel,
                (rel > tol ? "  <-- FAIL" : ""));
    if (rel > tol) fails++;
}

// y' = y^2, g = y(T), dg/dy0 = 1/(1 - y0 T)^2.
static void autonomous() {
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(1);
    x[0] = 1.0;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(1);
    f[0] = x[0] * x[0];
    rhs.Dependent(x, f);
    rhs.optimize();

    double y0[1] = {1.0}, ab[1] = {1e-12}, w[1] = {1.0}, adj[1];
    yabi::BDFIntegrator<double> it(rhs);
    if (!it.resetIntegration(y0, 0.0, NULL, 0.1, 1e-12, ab, false, false, NULL, NULL, NULL, NULL,
                             0) ||
        !it.performIntegration(0.5, 100000) || !it.continuousAdjoint(w, adj, NULL)) {
        std::printf("  autonomous: continuousAdjoint FAILED\n");
        fails++;
        return;
    }
    check("autonomous dg/dy0", adj[0], 4.0);
}

// y' = e^t y^2 (RHS_type 1), y = y0/(1 - y0 (e^t - 1)), dg/dy0 = 1/(1 - y0 B)^2.
static void time_dependent() {
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(2);
    x[0] = 1.5;
    x[1] = 0.0;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(1);
    f[0] = CppAD::exp(x[1]) * x[0] * x[0];
    rhs.Dependent(x, f);
    rhs.optimize();

    const double y0v = 1.5, T = 0.4, B = std::exp(T) - 1.0;
    double y0[1] = {y0v}, ab[1] = {1e-12}, w[1] = {1.0}, adj[1];
    yabi::BDFIntegrator<double> it(rhs);
    if (!it.resetIntegration(y0, 0.0, NULL, 0.01, 1e-12, ab, false, false, NULL, NULL, NULL, NULL,
                             0) ||
        !it.performIntegration(T, 100000) || !it.continuousAdjoint(w, adj, NULL)) {
        std::printf("  time-dependent: continuousAdjoint FAILED\n");
        fails++;
        return;
    }
    check("time-dependent dg/dy0", adj[0], 1.0 / std::pow(1.0 - y0v * B, 2.0));
}

// y' = p y^2 (RHS_type 2, Domain = 3: y, t, p), dg/dy0 = 1/(1 - y0 p T)^2.
static void parameter_dependent() {
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(3);
    x[0] = 2.0;
    x[1] = 0.0;
    x[2] = 1.5;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(1);
    f[0] = x[2] * x[0] * x[0];
    rhs.Dependent(x, f);
    rhs.optimize();

    const double y0v = 2.0, p = 1.5, T = 0.2;
    double y0[1] = {y0v}, pp[1] = {p}, ab[1] = {1e-12}, w[1] = {1.0}, adj[1], adj_p[1];
    yabi::BDFIntegrator<double> it(rhs);
    if (!it.resetIntegration(y0, 0.0, pp, 0.01, 1e-12, ab, false, false, NULL, NULL, NULL, NULL,
                             0) ||
        !it.performIntegration(T, 100000) || !it.continuousAdjoint(w, adj, adj_p)) {
        std::printf("  parameter-dependent: continuousAdjoint FAILED\n");
        fails++;
        return;
    }
    const double u = 1.0 - y0v * p * T;
    check("parameter-dependent dg/dy0", adj[0], 1.0 / (u * u));
    check("parameter-dependent dg/dp", adj_p[0], y0v * y0v * T / (u * u));
}

// Robertson at t=1 (IND is stable there): cross-check continuous adjoint vs
// reverse(1).
static void robertson_cross_check() {
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(3);
    x[0] = 1.0; x[1] = 0.0; x[2] = 0.0;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(3);
    f[0] = -0.04 * x[0] + 1e4 * x[1] * x[2];
    f[1] = 0.04 * x[0] - 1e4 * x[1] * x[2] - 3e7 * x[1] * x[1];
    f[2] = 3e7 * x[1] * x[1];
    rhs.Dependent(x, f);
    rhs.optimize();

    const double T = 1.0;
    double y0[3] = {1.0, 0.0, 0.0}, ab[3] = {1e-8, 1e-8, 1e-8}, w[3] = {1.0, 0.0, 0.0};

    // IND reference.
    double ref[3] = {0, 0, 0};
    {
        yabi::BDFIntegrator<double> it(rhs);
        it.resetIntegration(y0, 0.0, NULL, 1.0, 1e-8, ab, true, false, NULL, NULL, NULL, NULL, 0);
        it.performIntegration(T, 10000000);
        double seed[3] = {0, 0, 0};
        it.forward(1, seed, NULL, false);
        double *a[3];
        for (int i = 0; i < 3; i++) a[i] = new double[1];
        if (it.reverse(1, w)) it.getReverseSolutionVector(a, NULL);
        for (int i = 0; i < 3; i++) { ref[i] = a[i][0]; delete[] a[i]; }
    }
    // Continuous adjoint.
    double adj[3];
    yabi::BDFIntegrator<double> it(rhs);
    it.resetIntegration(y0, 0.0, NULL, 1.0, 1e-8, ab, false, false, NULL, NULL, NULL, NULL, 0);
    it.performIntegration(T, 10000000);
    if (!it.continuousAdjoint(w, adj, NULL)) {
        std::printf("  robertson cross-check: continuousAdjoint FAILED\n");
        fails++;
        return;
    }
    // The fast stiff component (y2) has an IND reference that is only ~1e-3
    // accurate: IND oscillates 0.2586..0.2627 across rtol 1e-6..1e-8, while the
    // continuous adjoint is stable at 0.25988 (a demonstration of the T25
    // motivation).  The slow components agree to ~1e-6.  Cross-check with a
    // tolerance reflecting the IND reference's own uncertainty on the stiff
    // transient; the tight correctness anchors are the exact-solution checks
    // above.
    for (int i = 0; i < 3; i++) {
        char nm[64];
        std::snprintf(nm, sizeof nm, "robertson dg/dy0[%d]", i);
        check(nm, adj[i], ref[i], 5e-3);
    }
}

// HIRES to the full steady state t=321.812: the continuous adjoint is stable
// (~0.002662) where the IND replay overflows (~1e+114).
static void hires_stability() {
    constexpr unsigned N = 8;
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(N);
    x[0] = 1.0; x[1] = 0.0; x[2] = 0.0; x[3] = 0.0;
    x[4] = 0.0; x[5] = 0.0; x[6] = 0.0; x[7] = 0.0057;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(N);
    f[0] = -1.71 * x[0] + 0.43 * x[1] + 8.32 * x[2] + 0.0007;
    f[1] = 1.71 * x[0] - 8.75 * x[1];
    f[2] = -10.03 * x[2] + 0.43 * x[3] + 0.035 * x[4];
    f[3] = 8.32 * x[1] + 1.71 * x[2] - 1.12 * x[3];
    f[4] = -1.745 * x[4] + 0.43 * x[5] + 0.43 * x[6];
    f[5] = -280.0 * x[5] * x[7] + 0.69 * x[3] + 1.71 * x[4] - 0.43 * x[5] + 0.69 * x[6];
    f[6] = 280.0 * x[5] * x[7] - 1.81 * x[6];
    f[7] = -f[6];
    rhs.Dependent(x, f);
    rhs.optimize();

    double y0[N] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057}, ab[N], w[N];
    for (unsigned i = 0; i < N; i++) { ab[i] = 1e-10; w[i] = 0.0; }
    w[0] = 1.0;
    double adj[N];
    yabi::BDFIntegrator<double> it(rhs);
    it.resetIntegration(y0, 0.0, NULL, 0.1, 1e-10, ab, false, false, NULL, NULL, NULL, NULL, 0);
    it.performIntegration(321.812, 10000000);
    if (!it.continuousAdjoint(w, adj, NULL)) {
        std::printf("  hires: continuousAdjoint FAILED\n");
        fails++;
        return;
    }
    check("hires dg/dy0[0] (stable)", adj[0], 0.002662, 0.02);
}

int main() {
    std::printf("=== T25 Continuous adjoint (first order) ===\n");
    autonomous();
    time_dependent();
    parameter_dependent();
    robertson_cross_check();
    hires_stability();
    std::printf("FAILS=%d\n", fails);
    return fails ? 1 : 0;
}
