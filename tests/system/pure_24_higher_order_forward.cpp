// T40 — Higher-order forward sensitivity, pure (CppAD-free) path.
//
// UserODERHS supplies the k-th-order nonlinear source via the analytic
// setOrderKDirectional hook; forwardOrder(k) fails loudly without it.
// y' = y^2: source2 = H(S1,S1) = 2 S1^2, source3 = 3 H(S1,S2) = 6 S1 S2 (T = 0).
#include <yabi/bdf_integrator.hpp>
#include <yabi/user_ode_rhs.hpp>

#include <cmath>
#include <cstdio>

int main() {
    std::printf("=== T40 Higher-order forward (pure / UserODERHS) ===\n");
    auto f = [](const double *y, double *F) { F[0] = y[0] * y[0]; };
    auto J = [](const double *y, double *Jm) { Jm[0] = 2.0 * y[0]; };
    yabi::UserODERHS<double> rhs(1, f, J);
    rhs.setOrderKDirectional([](unsigned k, const double *, const double *S, double *r) {
        if (k == 2)
            r[0] = 2.0 * S[0] * S[0]; // H(S1,S1)
        else if (k == 3)
            r[0] = 6.0 * S[0] * S[1]; // 3 H(S1,S2)
        else
            r[0] = 0.0;
    });

    double y0[1] = {1.0}, ab[1] = {1e-12}, s[1] = {1.0}, d[1];
    yabi::BDFIntegrator<double> it(rhs);
    if (!it.resetIntegration(y0, 0.0, NULL, 0.1, 1e-12, ab, false, false, NULL, NULL, NULL, NULL,
                             0) ||
        !it.performIntegration(0.5, 100000)) {
        std::printf("  integration FAILED\n");
        return 1;
    }

    const double T = 0.5;
    int fails = 0;
    it.forwardOrder(2, s, d);
    double exact2 = 2.0 * T / std::pow(1.0 - T, 3.0); // 8
    double rel2 = std::fabs(d[0] - exact2) / (1.0 + exact2);
    std::printf("  d^2y/dy0^2 got=%.10g exact=%.10g rel=%.2e%s\n", d[0], exact2, rel2,
                (rel2 > 1e-6 ? "  <-- FAIL" : ""));
    if (rel2 > 1e-6) fails++;

    it.forwardOrder(3, s, d);
    double exact3 = 6.0 * T * T / std::pow(1.0 - T, 4.0); // 24
    double rel3 = std::fabs(d[0] - exact3) / (1.0 + exact3);
    std::printf("  d^3y/dy0^3 got=%.10g exact=%.10g rel=%.2e%s\n", d[0], exact3, rel3,
                (rel3 > 1e-6 ? "  <-- FAIL" : ""));
    if (rel3 > 1e-6) fails++;

    // Without the hook, forwardOrder(k) must fail loudly (not silently fall back).
    yabi::UserODERHS<double> rhs_nohook(1, f, J);
    yabi::BDFIntegrator<double> it2(rhs_nohook);
    it2.resetIntegration(y0, 0.0, NULL, 0.1, 1e-12, ab, false, false, NULL, NULL, NULL, NULL, 0);
    it2.performIntegration(0.5, 100000);
    bool nohook = it2.forwardOrder(3, s, d);
    std::printf("  forwardOrder(3) without hook returns %s (expect false)\n",
                nohook ? "true  <-- FAIL" : "false");
    if (nohook) fails++;

    std::printf("FAILS=%d\n", fails);
    return fails ? 1 : 0;
}
