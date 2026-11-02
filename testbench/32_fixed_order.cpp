/*
 * T21 — force-order API (setFixedOrder) + the deferred per-order convergence proof.
 *
 * For each order q = 2..6, setFixedOrder(q) pins the BDF order.  At a loose
 * tolerance the step size sits at the max-step cap, so halving the cap halves
 * the step and the global error of a q-th order method halves by 2^q:
 *
 *     slope = log2(err(h) / err(h/2))  ~=  q
 *
 * This is the T03-deferred per-order convergence measurement.  We also assert
 * getCurrentOrder() == q after the integration, and that setFixedOrder rejects
 * out-of-range orders (0 and 7).
 */

#include <yabi/bdf_integrator.hpp>

#include <cmath>
#include <cstdio>
#include <limits>

namespace {

struct Decay {
    CppAD::ADFun<double> rhs;

    Decay() {
        CppAD::vector<CppAD::AD<double> > x(1);
        x[0] = 1.0;
        CppAD::Independent(x);

        CppAD::vector<CppAD::AD<double> > f(1);
        f[0] = -x[0];

        rhs.Dependent(x, f);
        rhs.optimize();
    }
};

constexpr double T_END = 3.0;
constexpr double RTOL = 1e-2;
constexpr double EXACT = 0.04978706836786394;  // exp(-3)

// Integrate y' = -y at a fixed order q with a max-step cap hmax; return the
// relative global error at T_END and the final observed order.
bool integrate(unsigned int q, double hmax, double &rel_error, unsigned int &order) {
    Decay model;
    yabi::BDFIntegrator<double> integrator(model.rhs);

    if (!integrator.setFixedOrder(q))
        return false;

    double y0[1] = {1.0};
    double abs_err[1] = {1e-16};
    double y[1] = {0.0};
    if (!integrator.resetIntegration(y0, hmax, RTOL, abs_err, false, false, NULL, NULL, NULL, NULL,
                                     0) ||
        !integrator.performIntegration(T_END, 1000000) ||
        !integrator.getSolutionVector(y))
        return false;

    rel_error = std::fabs(y[0] - EXACT) / EXACT;
    order = integrator.getCurrentOrder();
    return true;
}

}  // namespace

int main() {
    int failures = 0;

    // ---- validation -------------------------------------------------------
    {
        Decay model;
        yabi::BDFIntegrator<double> integrator(model.rhs);
        if (integrator.setFixedOrder(0) || integrator.setFixedOrder(7)) {
            std::printf("FAIL: setFixedOrder(0 or 7) should return false\n");
            ++failures;
        }
        for (unsigned int q = 1; q <= 6; ++q)
            if (!integrator.setFixedOrder(q)) {
                std::printf("FAIL: setFixedOrder(%u) should return true\n", q);
                ++failures;
            }
    }

    // ---- per-order convergence slope --------------------------------------
    const double caps[2] = {0.1, 0.05};

    std::printf("per-order convergence (y'=-y, t<=%g, rtol=%g)\n", T_END, RTOL);
    std::printf("  %-6s %-8s %-8s %-8s %s\n", "order", "err(0.1)", "err(0.05)", "slope", "");

    for (unsigned int q = 2; q <= 6; ++q) {
        double err[2];
        unsigned int order[2];
        for (int i = 0; i < 2; ++i)
            if (!integrate(q, caps[i], err[i], order[i])) {
                std::printf("  FAIL: integration failed for order %u cap %g\n", q, caps[i]);
                ++failures;
                continue;
            }

        const double slope = std::log2(err[0] / err[1]);
        const bool ok = std::fabs(slope - static_cast<double>(q)) < 0.35 && order[0] == q &&
                        order[1] == q;
        std::printf("  %-6u %-8.2e %-8.2e %-8.3f %s\n", q, err[0], err[1], slope,
                    ok ? "OK" : "FAIL");
        if (!ok) {
            std::printf("     (final orders %u/%u)\n", order[0], order[1]);
            ++failures;
        }
    }

    if (failures != 0) {
        std::printf("\nFAIL: %d check(s) failed\n", failures);
        return 1;
    }

    std::printf("\nPASS\n");
    return 0;
}
