/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// T32b — pure (CppAD-free) matrix-free Newton-Krylov component test.
//
// Builds and runs in both configurations.  It proves that the analytic
// sparsity pattern and J*v hooks on UserODERHS activate the same sparse /
// matrix-free Newton path that CppAD tapes get through CppADRHS:
//   * a dense direct solve and a GMRES solve reproduce the same trajectory,
//   * the Krylov path is actually taken (operatorApplications() > 0), and
//   * there is no silent dense fallback.

#include <yabi/bdf_integrator.hpp>
#include <yabi/detail/dense_linear_solver.ipp>
#include <yabi/matrix_free_solver.hpp>
#include <yabi/user_ode_rhs.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string &what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << std::endl;
        ++failures;
    }
}

// Brusselator: same problem as system test 12 / pure_12.
constexpr int n = 2;
constexpr double A = 1.0;
constexpr double B = 3.0;

void ode(const double *y, double *f) {
    f[0] = A + y[0] * y[0] * y[1] - (B + 1.0) * y[0];
    f[1] = B * y[0] - y[0] * y[0] * y[1];
}

// Column-major analytic Jacobian: J[i + n*j] = df_i/dy_j.
void jacobian(const double *y, double *J) {
    J[0 + n * 0] = 2.0 * y[0] * y[1] - (B + 1.0);
    J[1 + n * 0] = B - 2.0 * y[0] * y[1];
    J[0 + n * 1] = y[0] * y[0];
    J[1 + n * 1] = -y[0] * y[0];
}

// Genuinely matrix-free product: J is never formed.
void jacobianVectorProduct(const double *y, const double *v, double *Jv) {
    const double diagonal = 2.0 * y[0] * y[1] - (B + 1.0);
    const double upper = y[0] * y[0];
    const double lower = B - 2.0 * y[0] * y[1];

    Jv[0] = diagonal * v[0] + upper * v[1];
    Jv[1] = lower * v[0] - upper * v[1];
}

yabi::SparsityPattern brusselatorPattern() {
    yabi::SparsityPattern pattern;
    pattern.rows = n;
    pattern.cols = n;
    pattern.column_rows = {{0, 1}, {0, 1}};
    return pattern;
}

struct RunResult {
    bool ok = false;
    std::vector<double> y;
    unsigned int jacobian_evaluations = 0;
    unsigned int fallbacks = 0;
    unsigned long long operator_applications = 0;
    bool last_converged = false;
};

RunResult integrate(bool matrix_free, bool analytic_jv, bool with_pattern = true,
                    std::string *stderr_capture = nullptr) {
    yabi::UserODERHS<double> rhs(n, ode, jacobian);
    if (with_pattern)
        rhs.setSparsityPattern(brusselatorPattern());
    if (analytic_jv)
        rhs.setJacobianVectorProduct(jacobianVectorProduct);

    yabi::BDFIntegrator<double> integrator(rhs);

    yabi::KrylovSolverBase<double> *krylov = nullptr;
    if (matrix_free) {
        krylov = new yabi::GmresLinearSolver<double>();
        krylov->setPreconditioner(yabi::KrylovPreconditioner::Jacobi);
        krylov->setTolerance(1e-12, 0.0, 200, 30);
        integrator.setLinearSolver(krylov);
        check(integrator.getMatrixFreeAssembly(),
              "installing a Krylov solver enables matrix-free assembly");
    } else {
        integrator.setLinearSolver(new yabi::DenseLUSolver<double>());
        check(!integrator.getMatrixFreeAssembly(),
              "the dense reference solver keeps matrix-free off");
    }

    double y0[n] = {1.0, 2.0};
    double abs_err[n] = {1e-10, 1e-10};

    yabi::IntegrationOptions<double> options;
    RunResult result;
    if (!integrator.resetIntegration(y0, 5.0, 1e-10, abs_err, options))
        return result;
    if (!integrator.setAlgorithmSettings(5, std::numeric_limits<double>::epsilon(), true, 1))
        return result;

    std::ostringstream captured;
    std::streambuf *previous = nullptr;
    if (stderr_capture)
        previous = std::cerr.rdbuf(captured.rdbuf());
    const bool performed = integrator.performIntegration(5.0, 10000000);
    if (stderr_capture) {
        std::cerr.rdbuf(previous);
        *stderr_capture = captured.str();
    }
    if (!performed)
        return result;

    result.y.resize(n);
    if (!integrator.getSolutionVector(result.y.data()))
        return result;

    result.jacobian_evaluations = integrator.getJacobianEvaluations();
    if (krylov) {
        result.fallbacks = krylov->fallbackCount();
        result.operator_applications = krylov->operatorApplications();
        result.last_converged = krylov->lastConverged();
    }
    result.ok = true;
    return result;
}

double maxDifference(const std::vector<double> &a, const std::vector<double> &b) {
    if (a.size() != b.size())
        return std::numeric_limits<double>::infinity();
    double diff = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        diff = std::max(diff, std::fabs(a[i] - b[i]));
    return diff;
}

void checkMatrixFree(const char *name, bool analytic_jv, const RunResult &dense) {
    const RunResult result = integrate(true, analytic_jv);
    check(result.ok, std::string(name) + ": integration succeeds");
    if (!result.ok || !dense.ok)
        return;

    const double diff = maxDifference(result.y, dense.y);
    check(diff < 1e-9, std::string(name) + ": state matches the direct solver (diff " +
                           std::to_string(diff) + ")");
    check(result.operator_applications > 0,
          std::string(name) + ": the Krylov operator was applied");
    check(result.fallbacks == 0, std::string(name) + ": no dense fallback");
    check(result.last_converged, std::string(name) + ": last Krylov solve converged");
    check(result.jacobian_evaluations == dense.jacobian_evaluations,
          std::string(name) + ": identical Jacobian evaluation count");

    std::cout << "  " << name << ": max|direct-krylov|=" << diff
              << ", J*v applications=" << result.operator_applications
              << ", fallbacks=" << result.fallbacks << "\n";
}

void checkMissingPatternIsNotSilent(const RunResult &dense) {
    std::string log;
    const RunResult result = integrate(true, true, false, &log);
    check(result.ok, "matrix-free without a pattern still integrates");
    if (!result.ok || !dense.ok)
        return;

    const double diff = maxDifference(result.y, dense.y);
    check(diff < 1e-9, "without a pattern the dense path matches the reference");
    check(result.operator_applications == 0,
          "without a pattern the Krylov operator is not applied");
    check(result.fallbacks == 0, "without a pattern the solve uses the dense delegation");

    const std::string marker = "no Jacobian sparsity pattern";
    const std::size_t first = log.find(marker);
    check(first != std::string::npos,
          "a missing pattern emits a diagnostic instead of a silent downgrade");
    check(log.find(marker, first + marker.size()) == std::string::npos,
          "the missing-pattern diagnostic is emitted only once");

    std::cout << "  missing pattern: max|direct-dense|=" << diff << ", warned once\n";
}

} // namespace

int main() {
    std::cout << "T32b pure matrix-free Newton-Krylov (Brusselator)\n";

    const RunResult dense = integrate(false, true);
    check(dense.ok, "direct reference integration succeeds");

    checkMatrixFree("GMRES/analytic-J*v", true, dense);
    // Without the analytic hook the default J*v still uses the analytic
    // Jacobian, so the same path must remain correct.
    checkMatrixFree("GMRES/default-J*v", false, dense);
    checkMissingPatternIsNotSilent(dense);

    if (failures != 0) {
        std::cerr << failures << " pure matrix-free check(s) failed" << std::endl;
        return 1;
    }
    std::cout << "PASS: pure UserODERHS drives sparse/matrix-free Newton\n";
    return 0;
}
