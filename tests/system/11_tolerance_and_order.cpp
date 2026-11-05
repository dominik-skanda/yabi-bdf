/*
 * T03 — solver validation that can actually be asserted through the public API.
 *
 * Two properties are checked:
 *
 *   A. Tolerance adherence.  For a smooth non-stiff problem the global error
 *      must track the requested relative tolerance: it stays within a modest
 *      constant factor of rtol, and tightening rtol by eight decades must
 *      improve the error by at least three decades.  This is the property a
 *      user relies on; it also catches a regression that silently ignores rtol.
 *
 *   B. First-order step convergence with MAXORDER = 1.  The public API cannot
 *      force orders 2-6: setAlgorithmSettings(MAXORDER, ...) only caps the
 *      order, the controller still chooses.  Order 1 is the one case that is
 *      genuinely pinned, so that is the one slope that is asserted here.  With
 *      a loose tolerance the step sits at max_StepLength, so halving the cap
 *      halves the global error (observed slope ~0.91-0.96).
 *
 * NOT asserted: "observed order == MAXORDER" for orders 2-6.  The controller's
 * order depends on the step/tolerance history and is not exposed by the API.
 * See tests/system/README.md and TODO/reports/T03-*.md.
 */

#include <yabi/bdf_integrator.hpp>

#include <cmath>
#include <cstdio>
#include <limits>

namespace {

struct Decay
{
    CppAD::ADFun<double> rhs;

    Decay()
    {
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
constexpr double MAX_STEP = 0.05;
constexpr double EXACT = 0.04978706836786394;  // exp(-3)

// Integrate y' = -y from 1 to exp(-T_END) with the given controls.
// Returns false if the integrator rejects the setup or the integration.
bool integrate(double rtol, double hmax, unsigned int maxorder, double &rel_error)
{
    Decay model;
    yabi::BDFIntegrator<double> integrator(model.rhs);

    if (!integrator.setAlgorithmSettings(maxorder,
                                         std::numeric_limits<double>::epsilon(),
                                         true, 1))
        return false;

    double y_start[1] = {1.0};
    double absolute_error[1] = {1e-16};
    double y[1] = {0.0};

    if (!integrator.resetIntegration(y_start, hmax, rtol, absolute_error,
                                     false, false, NULL, NULL, NULL, NULL, 0))
        return false;

    if (!integrator.performIntegration(T_END, 1000000))
        return false;

    if (!integrator.getSolutionVector(y))
        return false;

    rel_error = std::fabs(y[0] - EXACT) / EXACT;
    return true;
}

}  // namespace

int main()
{
    int failures = 0;

    // ---- A. tolerance adherence -------------------------------------------
    const double rts[] = {1e-4, 1e-6, 1e-8, 1e-10, 1e-12};
    const double allowed_factor[] = {500.0, 500.0, 500.0, 500.0, 500.0};
    double rel[5];

    std::printf("A. tolerance adherence (y' = -y, t<=3, max_step=%g)\n", MAX_STEP);
    std::printf("   %-8s %-12s %-10s %s\n", "rtol", "rel error", "err/rtol", "limit");

    for (int i = 0; i < 5; ++i)
    {
        if (!integrate(rts[i], MAX_STEP, 6, rel[i]))
        {
            std::printf("   FAIL: integration failed for rtol=%g\n", rts[i]);
            return 1;
        }

        const double ratio = rel[i] / rts[i];
        const bool ok = ratio <= allowed_factor[i];
        std::printf("   %-8.0e %-12.4e %-10.2f %-8.0f %s\n",
                    rts[i], rel[i], ratio, allowed_factor[i], ok ? "OK" : "FAIL");
        if (!ok)
        {
            std::printf("   FAIL: error exceeds the tolerance envelope\n");
            ++failures;
        }
    }

    // Tightening rtol by eight decades must buy at least three decades.
    const double improvement = rel[0] / rel[4];
    if (!(improvement > 1e3))
    {
        std::printf("   FAIL: rtol 1e-4 -> 1e-12 improved the error only %.1fx\n",
                    improvement);
        ++failures;
    }
    else
    {
        std::printf("   rtol 1e-4 -> 1e-12 improved the error by %.1e x\n", improvement);
    }

    // ---- B. first-order step convergence, MAXORDER = 1 ---------------------
    // Loose tolerance keeps the step at the cap, so the cap is the step size.
    // 0.2 is deliberately excluded: there the first-order local error is large
    // enough that the controller shortens the step below the cap, and the
    // cap/error relation breaks (measured slope 0.42).  From 0.1 downwards the
    // step is pinned and the relation is clean.
    std::printf("\nB. first-order step convergence (MAXORDER=1, rtol=1e-2)\n");
    const double caps[] = {0.1, 0.05, 0.025};
    double err[3];

    for (int i = 0; i < 3; ++i)
    {
        if (!integrate(1e-2, caps[i], 1, err[i]))
        {
            std::printf("   FAIL: integration failed for max_step=%g\n", caps[i]);
            return 1;
        }
        std::printf("   max_step=%-6g rel error=%.6e\n", caps[i], err[i]);
    }

    for (int i = 0; i < 2; ++i)
    {
        const double slope = std::log2(err[i] / err[i + 1]);
        const bool ok = slope > 0.85 && slope < 1.15;
        std::printf("   observed slope %g -> %g : %.3f %s\n",
                    caps[i], caps[i + 1], slope, ok ? "OK" : "FAIL");
        if (!ok)
        {
            std::printf("   FAIL: expected first-order convergence (slope ~1)\n");
            ++failures;
        }
    }

    if (failures != 0)
    {
        std::printf("\nFAIL: %d check(s) failed\n", failures);
        return 1;
    }

    std::printf("\nPASS\n");
    return 0;
}
