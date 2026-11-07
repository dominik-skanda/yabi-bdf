// Pure (CppAD-free) second-order forward sensitivity — the UserODERHS analytic
// second-order directional derivative hook (T27).  y' = y^2 (H v v = 2 v^2).
#include <yabi/bdf_integrator.hpp>
#include <yabi/user_ode_rhs.hpp>

#include <cmath>
#include <cstdio>

int main() {
    std::printf("=== T27 Second-order forward (pure / UserODERHS) ===\n");
    auto f = [](const double *y, double *F) { F[0] = y[0] * y[0]; };
    auto J = [](const double *y, double *Jm) { Jm[0] = 2.0 * y[0]; };
    auto sod = [](const double *, const double *v, double *r) { r[0] = 2.0 * v[0] * v[0]; };
    yabi::UserODERHS<double> rhs(1, f, J);
    rhs.setSecondOrderDirectional(sod);

    double y0[1] = {1.0}, ab[1] = {1e-12}, s[1] = {1.0}, d2[1];
    yabi::BDFIntegrator<double> it(rhs);
    if (!it.resetIntegration(y0, 0.0, NULL, 0.1, 1e-12, ab, false, false, NULL, NULL, NULL, NULL,
                             0) ||
        !it.performIntegration(0.5, 100000) || !it.secondOrderForward(s, d2)) {
        std::printf("  secondOrderForward FAILED\n");
        return 1;
    }
    const double exact = 8.0;
    double rel = std::fabs(d2[0] - exact) / (1.0 + exact);
    std::printf("  d^2y/dy0^2 got=%.10g exact=%.10g rel=%.2e%s\n", d2[0], exact, rel,
                (rel > 1e-6 ? "  <-- FAIL" : ""));
    return rel > 1e-6 ? 1 : 0;
}
