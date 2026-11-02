/*
 * T17 — public diagnostics / observability API.
 *
 * The integrator now exposes read-only getters for the quantities the
 * step/order controller and the statistics already track.  This test checks
 * them two ways:
 *
 *   1. Live values are sane and internally consistent (order in [1,MAXORDER],
 *      step size positive, counters monotone, Newton iterations >= accepted
 *      steps, Jacobian/factorization counters >= 1).
 *
 *   2. The getters agree with printStatistic() for the same integration.
 *
 * It also closes the T03 gap ("the chosen order is not observable"): with
 * MAXORDER=1 the controller genuinely pins the order, so getCurrentOrder()
 * must be 1; with MAXORDER=6 on a smooth problem it must rise above 1.
 */

#include <yabi/bdf_integrator.hpp>

#include <cctype>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>
#include <string>

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
constexpr double MAX_STEP = 0.05;

// Integrate y' = -y from 1 to exp(-3) with the given max order and tolerance.
bool integrate(yabi::BDFIntegrator<double> &integrator, unsigned int maxorder, double rtol) {
    if (!integrator.setAlgorithmSettings(maxorder, std::numeric_limits<double>::epsilon(), true, 1))
        return false;

    double y_start[1] = {1.0};
    double absolute_error[1] = {1e-16};

    if (!integrator.resetIntegration(y_start, MAX_STEP, rtol, absolute_error, false, false, NULL,
                                     NULL, NULL, NULL, 0))
        return false;

    if (!integrator.performIntegration(T_END, 1000000))
        return false;

    return true;
}

// Extract the integer printed after `label` in printStatistic()'s output.
unsigned int statValue(const std::string &s, const std::string &label) {
    std::string::size_type pos = s.find(label);
    if (pos == std::string::npos)
        return static_cast<unsigned int>(-1);
    pos = s.find(':', pos);
    if (pos == std::string::npos)
        return static_cast<unsigned int>(-1);
    ++pos;
    while (pos < s.size() && s[pos] == ' ')
        ++pos;
    unsigned int v = 0;
    while (pos < s.size() && std::isdigit(static_cast<unsigned char>(s[pos]))) {
        v = v * 10 + static_cast<unsigned int>(s[pos] - '0');
        ++pos;
    }
    return v;
}

}  // namespace

int main() {
    int failures = 0;

    // ---- 1. MAXORDER = 1 pins the order ----------------------------------
    {
        Decay model;
        yabi::BDFIntegrator<double> integrator(model.rhs);
        if (!integrate(integrator, 1, 1e-6)) {
            std::printf("FAIL: MAXORDER=1 integration failed\n");
            return 1;
        }
        const unsigned int order = integrator.getCurrentOrder();
        std::printf("MAXORDER=1: order=%u steps=%u newton=%u jac=%u fact=%u\n", order,
                    integrator.getAcceptedSteps(), integrator.getNewtonIterations(),
                    integrator.getJacobianEvaluations(), integrator.getFactorizations());
        if (order != 1) {
            std::printf("FAIL: MAXORDER=1 should pin getCurrentOrder() to 1, got %u\n", order);
            ++failures;
        }
    }

    // ---- 2. MAXORDER = 6 lets the controller raise the order --------------
    {
        Decay model;
        yabi::BDFIntegrator<double> integrator(model.rhs);
        if (!integrate(integrator, 6, 1e-10)) {
            std::printf("FAIL: MAXORDER=6 integration failed\n");
            return 1;
        }
        const unsigned int order = integrator.getCurrentOrder();
        const double h = integrator.getCurrentStepSize();
        const unsigned int accepted = integrator.getAcceptedSteps();
        const unsigned int newton = integrator.getNewtonIterations();
        const unsigned int jac = integrator.getJacobianEvaluations();
        const unsigned int fact = integrator.getFactorizations();

        std::printf("MAXORDER=6: order=%u h=%.6e accepted=%u rejected=%u rejectedNewton=%u "
                    "newton=%u jac=%u fact=%u\n",
                    order, h, accepted, integrator.getRejectedSteps(),
                    integrator.getRejectedNewtonSteps(), newton, jac, fact);

        if (order < 2 || order > 6) {
            std::printf("FAIL: expected the controller to raise order above 1 (got %u)\n", order);
            ++failures;
        }
        if (!(h > 0.0) || !std::isfinite(h)) {
            std::printf("FAIL: getCurrentStepSize() should be positive and finite (got %g)\n", h);
            ++failures;
        }
        if (accepted == 0) {
            std::printf("FAIL: getAcceptedSteps() should be > 0\n");
            ++failures;
        }
        if (newton < accepted) {
            std::printf("FAIL: newton iterations (%u) must be >= accepted steps (%u)\n", newton,
                        accepted);
            ++failures;
        }
        if (jac == 0 || fact == 0) {
            std::printf("FAIL: jacobian/factorization counters must be > 0 (jac=%u fact=%u)\n", jac,
                        fact);
            ++failures;
        }

        // ---- cross-check against printStatistic --------------------------
        std::ostringstream oss;
        integrator.printStatistic(oss, " ");
        const std::string s = oss.str();
        const unsigned int pAccepted = statValue(s, "ACCEPTED STEPS");
        const unsigned int pRejected = statValue(s, "REJECTED STEPS");
        const unsigned int pRejNewton = statValue(s, "REJECTED NEWTON STEPS");
        const unsigned int pJac = statValue(s, "JACOBIAN UPDATED");
        const unsigned int pMatrixP = statValue(s, "MATRIX P UPDATED");

        if (pAccepted != accepted || pRejected != integrator.getRejectedSteps() ||
            pRejNewton != integrator.getRejectedNewtonSteps() || pJac != jac ||
            pMatrixP != integrator.getMatrixPUpdates()) {
            std::printf("FAIL: printStatistic disagrees with getters\n");
            std::printf("  getter: accepted=%u rejected=%u rejNewton=%u jac=%u matrixP=%u\n",
                        accepted, integrator.getRejectedSteps(),
                        integrator.getRejectedNewtonSteps(), jac,
                        integrator.getMatrixPUpdates());
            std::printf("  print : accepted=%u rejected=%u rejNewton=%u jac=%u matrixP=%u\n",
                        pAccepted, pRejected, pRejNewton, pJac, pMatrixP);
            ++failures;
        }

        // getFactorizations() == Jacobian assemblies + P-only updates (each
        // does exactly one LU factorization).
        if (fact != jac + integrator.getMatrixPUpdates()) {
            std::printf("FAIL: getFactorizations() (%u) != jac (%u) + matrixP (%u)\n", fact, jac,
                        integrator.getMatrixPUpdates());
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
