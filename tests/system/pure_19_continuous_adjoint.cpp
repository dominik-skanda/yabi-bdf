// Pure (CppAD-free) continuous adjoint — the UserODERHS analytic-Jacobian path.
// y' = y^2, g = y(T), dg/dy0 = 1/(1 - y0 T)^2; verified against the exact value
// (and matching the CppAD system test 34).
#include <yabi/bdf_integrator.hpp>
#include <yabi/user_ode_rhs.hpp>

#include <cmath>
#include <cstdio>

static int fails = 0;
static void check(const char *name, double got, double exact, double tol = 1e-7) {
    double rel = std::fabs(got - exact) / (1.0 + std::fabs(exact));
    std::printf("  %-20s got=%.10g exact=%.10g rel=%.2e%s\n", name, got, exact, rel,
                (rel > tol ? "  <-- FAIL" : ""));
    if (rel > tol) fails++;
}

int main() {
    std::printf("=== T25 Continuous adjoint (pure / UserODERHS) ===\n");

    // Scalar y' = y^2.
    {
        auto f = [](const double *y, double *F) { F[0] = y[0] * y[0]; };
        auto J = [](const double *y, double *Jm) { Jm[0] = 2.0 * y[0]; };
        yabi::UserODERHS<double> rhs(1, f, J);
        yabi::BDFIntegrator<double> it(rhs);

        double y0[1] = {1.0}, ab[1] = {1e-12}, w[1] = {1.0}, adj[1];
        if (!it.resetIntegration(y0, 0.0, NULL, 0.1, 1e-12, ab, false, false, NULL, NULL, NULL,
                                 NULL, 0) ||
            !it.performIntegration(0.5, 100000) || !it.continuousAdjoint(w, adj, NULL)) {
            std::printf("  y'=y^2: continuousAdjoint FAILED\n");
            return 1;
        }
        check("y'=y^2 dg/dy0", adj[0], 4.0);
    }

    // Linear decay y' = -y, dg/dy0 = e^{-T} (exercises a constant Jacobian).
    {
        auto f = [](const double *y, double *F) { F[0] = -y[0]; };
        auto J = [](const double *, double *Jm) { Jm[0] = -1.0; };
        yabi::UserODERHS<double> rhs(1, f, J);
        yabi::BDFIntegrator<double> it(rhs);

        double y0[1] = {1.0}, ab[1] = {1e-12}, w[1] = {1.0}, adj[1];
        if (!it.resetIntegration(y0, 0.0, NULL, 0.1, 1e-12, ab, false, false, NULL, NULL, NULL,
                                 NULL, 0) ||
            !it.performIntegration(1.0, 100000) || !it.continuousAdjoint(w, adj, NULL)) {
            std::printf("  y'=-y: continuousAdjoint FAILED\n");
            return 1;
        }
        check("y'=-y dg/dy0", adj[0], std::exp(-1.0));
    }

    std::printf("FAILS=%d\n", fails);
    return fails ? 1 : 0;
}
