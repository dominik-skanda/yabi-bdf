// Pure (CppAD-free) non-autonomous continuous adjoint — the UserODERHS
// time + parameter path (T23a).  Same problems as system test 34, hand-written:
//   time    : y' = e^t y^2
//   param   : y' = p y^2   (dg/dy0 and dg/dp)
#include <yabi/bdf_integrator.hpp>
#include <yabi/user_ode_rhs.hpp>

#include <cmath>
#include <cstdio>

static int fails = 0;
static void check(const char *name, double got, double exact, double tol = 1e-7) {
    double rel = std::fabs(got - exact) / (1.0 + std::fabs(exact));
    std::printf("  %-18s got=%.10g exact=%.10g rel=%.2e%s\n", name, got, exact, rel,
                (rel > tol ? "  <-- FAIL" : ""));
    if (rel > tol) fails++;
}

int main() {
    std::printf("=== T23a Continuous adjoint (pure non-autonomous) ===\n");

    // Time-dependent y' = e^t y^2, arg = (y, t).
    {
        auto f = [](const double *arg, double *F) { F[0] = std::exp(arg[1]) * arg[0] * arg[0]; };
        auto J = [](const double *arg, double *Jm) { Jm[0] = 2.0 * std::exp(arg[1]) * arg[0]; };
        auto ft = [](const double *arg, double *Ft) {
            Ft[0] = std::exp(arg[1]) * arg[0] * arg[0];
        };
        yabi::UserODERHS<double> rhs(1, f, J);
        rhs.setTimeDerivative(ft);

        const double y0v = 1.5, T = 0.4, B = std::exp(T) - 1.0;
        double y0[1] = {y0v}, ab[1] = {1e-12}, w[1] = {1.0}, adj[1];
        yabi::BDFIntegrator<double> it(rhs);
        if (!it.resetIntegration(y0, 0.0, NULL, 0.01, 1e-12, ab, false, false, NULL, NULL, NULL,
                                 NULL, 0) ||
            !it.performIntegration(T, 100000) || !it.continuousAdjoint(w, adj, NULL)) {
            std::printf("  time-dependent: continuousAdjoint FAILED\n");
            return 1;
        }
        check("time dg/dy0", adj[0], 1.0 / std::pow(1.0 - y0v * B, 2.0));
    }

    // Parameter-dependent y' = p y^2, arg = (y, t, p).
    {
        auto f = [](const double *arg, double *F) { F[0] = arg[2] * arg[0] * arg[0]; };
        auto J = [](const double *arg, double *Jm) { Jm[0] = 2.0 * arg[2] * arg[0]; };
        auto ft = [](const double *, double *Ft) { Ft[0] = 0.0; };
        auto fp = [](const double *arg, double *Fp) { Fp[0] = arg[0] * arg[0]; }; // df/dp = y^2
        yabi::UserODERHS<double> rhs(1, f, J);
        rhs.setTimeDerivative(ft);
        rhs.setParameterDerivative(1, fp);

        const double y0v = 2.0, p = 1.5, T = 0.2, u = 1.0 - y0v * p * T;
        double y0[1] = {y0v}, pp[1] = {p}, ab[1] = {1e-12}, w[1] = {1.0}, adj[1], adj_p[1];
        yabi::BDFIntegrator<double> it(rhs);
        if (!it.resetIntegration(y0, 0.0, pp, 0.01, 1e-12, ab, false, false, NULL, NULL, NULL,
                                 NULL, 0) ||
            !it.performIntegration(T, 100000) || !it.continuousAdjoint(w, adj, adj_p)) {
            std::printf("  parameter-dependent: continuousAdjoint FAILED\n");
            return 1;
        }
        check("param dg/dy0", adj[0], 1.0 / (u * u));
        check("param dg/dp", adj_p[0], y0v * y0v * T / (u * u));
    }

    std::printf("FAILS=%d\n", fails);
    return fails ? 1 : 0;
}
