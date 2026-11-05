/*
 * T20 — dense output / continuous extension (getSolutionAt).
 *
 * The Nordsieck interpolant y(t) = sum_j z_corrector[n*j + i] * theta^j with
 * theta = (t - t_i[1]) / step_size_h is evaluated at interior points of the
 * last accepted step and checked against the exact solution of y' = -y:
 *
 *   1. getSolutionAt(t_ende) is bit-identical to getSolutionVector() (same
 *      theta, same GEMV).
 *   2. Interior points (theta in [-1, 0]) match exp(-t) to within a loose
 *      envelope of the integration tolerance.
 *   3. Out-of-domain (forward and backward) returns false.
 *   4. Calling before integration returns false.
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
constexpr double MAX_STEP = 0.5;
constexpr double RTOL = 1e-10;

}  // namespace

int main() {
    int failures = 0;

    Decay model;
    yabi::BDFIntegrator<double> integrator(model.rhs);

    // ---- 4. before integration: must return false ------------------------
    {
        double y_pre[1] = {-1.0};
        if (integrator.getSolutionAt(1.0, y_pre)) {
            std::printf("FAIL: getSolutionAt before integration should return false\n");
            ++failures;
        }
    }

    if (!integrator.setAlgorithmSettings(6, std::numeric_limits<double>::epsilon(), true, 1)) {
        std::printf("FAIL: setAlgorithmSettings\n");
        return 1;
    }

    double y0[1] = {1.0};
    double abs_err[1] = {1e-16};
    if (!integrator.resetIntegration(y0, MAX_STEP, RTOL, abs_err, false, false, NULL, NULL, NULL,
                                     NULL, 0) ||
        !integrator.performIntegration(T_END, 1000000)) {
        std::printf("FAIL: integration\n");
        return 1;
    }

    double y_end[1] = {0.0};
    if (!integrator.getSolutionVector(y_end)) {
        std::printf("FAIL: getSolutionVector\n");
        return 1;
    }

    const double h = integrator.getCurrentStepSize();
    const unsigned int order = integrator.getCurrentOrder();
    std::printf("last step h=%.6e order=%u t_end=%g\n", h, order, T_END);

    // ---- 1. endpoint consistency -----------------------------------------
    {
        double y_at[1] = {-1.0};
        if (!integrator.getSolutionAt(T_END, y_at)) {
            std::printf("FAIL: getSolutionAt(t_end) returned false\n");
            ++failures;
        } else if (y_at[0] != y_end[0]) {
            std::printf("FAIL: getSolutionAt(t_end)=%.17g != getSolutionVector=%.17g\n", y_at[0],
                        y_end[0]);
            ++failures;
        } else {
            std::printf("endpoint: getSolutionAt(t_end) == getSolutionVector (%.15g)\n", y_at[0]);
        }
    }

    // ---- 2. interior accuracy --------------------------------------------
    const double thetas[] = {-1.0, -0.75, -0.5, -0.25};
    for (double theta : thetas) {
        const double t = T_END + theta * h;
        double y_at[1] = {-1.0};
        if (!integrator.getSolutionAt(t, y_at)) {
            std::printf("FAIL: getSolutionAt(theta=%.2f) returned false\n", theta);
            ++failures;
            continue;
        }
        const double exact = std::exp(-t);
        const double err = std::fabs(y_at[0] - exact);
        // The interpolant is accurate to the method's order; 1e-6 is a loose
        // envelope far above the rtol=1e-10 global error.
        const bool ok = err < 1e-6;
        std::printf("theta=%+.2f t=%.12f y=%.15f exact=%.15f err=%.3e %s\n", theta, t, y_at[0],
                    exact, err, ok ? "OK" : "FAIL");
        if (!ok)
            ++failures;
    }

    // ---- 3. out-of-domain ------------------------------------------------
    {
        double y_out[1] = {0.0};
        if (integrator.getSolutionAt(T_END + 0.5 * h, y_out)) {
            std::printf("FAIL: forward extrapolation (theta>0) should return false\n");
            ++failures;
        }
        if (integrator.getSolutionAt(T_END - 2.0 * h, y_out)) {
            std::printf("FAIL: backward extrapolation (theta<-1) should return false\n");
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
