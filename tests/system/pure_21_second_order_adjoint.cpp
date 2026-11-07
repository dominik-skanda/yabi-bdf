// Pure (CppAD-free) second-order continuous adjoint — the UserODERHS analytic
// Hessian-in-direction hook (T26).  y' = y^2 (Hess(lambda y^2)*v = 2*lambda*v)
// and a linear oscillator (zero Hessian).
#include <yabi/bdf_integrator.hpp>
#include <yabi/user_ode_rhs.hpp>

#include <cmath>
#include <cstdio>

static int fails = 0;
static void check(const char *name, double got, double exact, double tol = 1e-6) {
    double rel = std::fabs(got - exact) / (1.0 + std::fabs(exact));
    std::printf("  %-16s got=%.10g exact=%.10g rel=%.2e%s\n", name, got, exact, rel,
                (rel > tol ? "  <-- FAIL" : ""));
    if (rel > tol) fails++;
}

int main() {
    std::printf("=== T26 Second-order adjoint (pure / UserODERHS) ===\n");

    // y' = y^2: J = 2y, Hess(lambda^T f)*v = 2*lambda*v.
    {
        auto f = [](const double *y, double *F) { F[0] = y[0] * y[0]; };
        auto J = [](const double *y, double *Jm) { Jm[0] = 2.0 * y[0]; };
        auto hv = [](const double *, const double *lam, const double *v, double *r) {
            r[0] = 2.0 * lam[0] * v[0];
        };
        yabi::UserODERHS<double> rhs(1, f, J);
        rhs.setHessianVectorProduct(hv);

        double y0[1] = {1.0}, ab[1] = {1e-12}, w[1] = {1.0}, s[1] = {1.0}, H[1];
        yabi::BDFIntegrator<double> it(rhs);
        if (!it.resetIntegration(y0, 0.0, NULL, 0.1, 1e-12, ab, false, false, NULL, NULL, NULL,
                                 NULL, 0) ||
            !it.performIntegration(0.5, 100000) || !it.secondOrderAdjoint(w, s, H)) {
            std::printf("  y'=y^2: secondOrderAdjoint FAILED\n");
            return 1;
        }
        check("d^2g/dy0^2", H[0], 8.0);
    }

    // Linear decay y' = -y: Hessian of lambda^T f = -lambda*y is zero.
    {
        auto f = [](const double *y, double *F) { F[0] = -y[0]; };
        auto J = [](const double *, double *Jm) { Jm[0] = -1.0; };
        auto hv = [](const double *, const double *, const double *, double *r) { r[0] = 0.0; };
        yabi::UserODERHS<double> rhs(1, f, J);
        rhs.setHessianVectorProduct(hv);

        double y0[1] = {1.0}, ab[1] = {1e-12}, w[1] = {1.0}, s[1] = {1.0}, H[1];
        yabi::BDFIntegrator<double> it(rhs);
        if (!it.resetIntegration(y0, 0.0, NULL, 0.1, 1e-12, ab, false, false, NULL, NULL, NULL,
                                 NULL, 0) ||
            !it.performIntegration(1.0, 100000) || !it.secondOrderAdjoint(w, s, H)) {
            std::printf("  y'=-y: secondOrderAdjoint FAILED\n");
            return 1;
        }
        check("y'=-y d^2g/dy0^2", H[0], 0.0);
    }

    std::printf("FAILS=%d\n", fails);
    return fails ? 1 : 0;
}
