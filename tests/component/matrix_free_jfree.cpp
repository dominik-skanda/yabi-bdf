/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// T32c — J-free matrix-free Newton-Krylov component test.
//
// The RHS deliberately exposes no sparsity pattern and no assembled CSC
// matrix.  A user preconditioner declares that it does not require an
// assembled matrix, so the Newton solver stays fully matrix-free: it estimates
// the scaling from one J*v product and hands the Krylov solver an empty CSC
// matrix plus the live P*x operator.
//
// Runs in both CppAD and pure configurations (UserODERHS supplies f and J*v).

#include <yabi/bdf_integrator.hpp>
#include <yabi/detail/dense_linear_solver.ipp>
#include <yabi/matrix_free_solver.hpp>
#include <yabi/preconditioner.hpp>
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

constexpr int n = 2;
constexpr double A = 1.0;
constexpr double B = 3.0;

void ode(const double *y, double *f) {
    f[0] = A + y[0] * y[0] * y[1] - (B + 1.0) * y[0];
    f[1] = B * y[0] - y[0] * y[0] * y[1];
}

void jacobian(const double *y, double *J) {
    J[0 + n * 0] = 2.0 * y[0] * y[1] - (B + 1.0);
    J[1 + n * 0] = B - 2.0 * y[0] * y[1];
    J[0 + n * 1] = y[0] * y[0];
    J[1 + n * 1] = -y[0] * y[0];
}

void jacobianVectorProduct(const double *y, const double *v, double *Jv) {
    const double diagonal = 2.0 * y[0] * y[1] - (B + 1.0);
    const double upper = y[0] * y[0];
    const double lower = B - 2.0 * y[0] * y[1];

    Jv[0] = diagonal * v[0] + upper * v[1];
    Jv[1] = lower * v[0] - upper * v[1];
}

// Operator-based diagonal (Jacobi-like) preconditioner with no assembled
// matrix.  It extracts the diagonal from the live P*x operator by applying it
// to the unit vectors, so the test also proves the operator is available to a
// J-free preconditioner.
class OperatorDiagonalPreconditioner : public yabi::Preconditioner<double> {
  public:
    bool requiresAssembledMatrix() const override {
        return false;
    }

    bool update(const yabi::CscMatrix<double> *P,
                const yabi::MatrixFreeOperator<double> *op,
                const std::vector<double> &diagonal, unsigned int size) override {
        ++updates;
        saw_null_matrix = P == nullptr;
        saw_operator = op != nullptr;
        saw_empty_diagonal = diagonal.empty();
        if (!saw_null_matrix || !saw_operator || size != (unsigned int)n)
            return false;

        diagonal_.assign(size, 0.0);
        std::vector<double> unit(size, 0.0);
        std::vector<double> column(size, 0.0);
        for (unsigned int i = 0; i < size; ++i) {
            std::fill(unit.begin(), unit.end(), 0.0);
            unit[i] = 1.0;
            if (!op->apply(unit.data(), column.data()))
                return false;
            diagonal_[i] = column[i];
            if (!std::isfinite(diagonal_[i]) || diagonal_[i] == 0.0)
                return false;
        }
        return true;
    }

    bool apply(const double *r, double *z) override {
        ++applications;
        for (unsigned int i = 0; i < (unsigned int)n; ++i)
            z[i] = r[i] / diagonal_[i];
        return true;
    }

    unsigned int updates = 0;
    unsigned int applications = 0;
    bool saw_null_matrix = false;
    bool saw_operator = false;
    bool saw_empty_diagonal = false;

  private:
    std::vector<double> diagonal_;
};

struct RunResult {
    bool ok = false;
    std::vector<double> y;
    unsigned long long operator_applications = 0;
    unsigned int fallbacks = 0;
    bool last_converged = false;
    unsigned int preconditioner_updates = 0;
    unsigned int preconditioner_applications = 0;
    bool saw_null_matrix = false;
    bool saw_operator = false;
    bool saw_empty_diagonal = false;
    bool has_pattern = false;
};

RunResult integrate(bool jfree, std::string *stderr_capture = nullptr) {
    yabi::UserODERHS<double> rhs(n, ode, jacobian);
    // No pattern and no assembled matrix anywhere: the J-free path must infer
    // everything from f and J*v.
    rhs.setJacobianVectorProduct(jacobianVectorProduct);

    OperatorDiagonalPreconditioner preconditioner;
    yabi::BDFIntegrator<double> integrator(rhs);

    yabi::KrylovSolverBase<double> *krylov = nullptr;
    if (jfree) {
        krylov = new yabi::GmresLinearSolver<double>();
        krylov->setTolerance(1e-12, 0.0, 200, 30);
        krylov->setUserPreconditioner(&preconditioner);
        integrator.setLinearSolver(krylov);
        check(integrator.getMatrixFreeAssembly(),
              "installing a Krylov solver enables matrix-free assembly");
        check(!krylov->requiresAssembledMatrix(),
              "the user preconditioner selects the J-free path");
    } else {
        integrator.setLinearSolver(new yabi::DenseLUSolver<double>());
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

    result.has_pattern = rhs.hasSparsityPattern();
    if (krylov) {
        result.operator_applications = krylov->operatorApplications();
        result.fallbacks = krylov->fallbackCount();
        result.last_converged = krylov->lastConverged();
    }
    result.preconditioner_updates = preconditioner.updates;
    result.preconditioner_applications = preconditioner.applications;
    result.saw_null_matrix = preconditioner.saw_null_matrix;
    result.saw_operator = preconditioner.saw_operator;
    result.saw_empty_diagonal = preconditioner.saw_empty_diagonal;
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

} // namespace

int main() {
    std::cout << "T32c J-free matrix-free Newton-Krylov (no pattern, Brusselator)\n";

    const RunResult dense = integrate(false);
    check(dense.ok, "direct reference integration succeeds");
    check(!dense.has_pattern, "the reference RHS itself has no sparsity pattern");

    std::string log;
    const RunResult jfree = integrate(true, &log);
    check(jfree.ok, "J-free integration succeeds");
    if (!jfree.ok)
        std::cerr << "--- captured log ---\n" << log << "--- end log ---\n";
    if (jfree.ok && dense.ok) {
        const double diff = maxDifference(jfree.y, dense.y);
        check(diff < 1e-9, "J-free state matches the direct solver (diff " + std::to_string(diff) +
                               ")");
        check(jfree.operator_applications > 0, "the Krylov operator was applied");
        check(jfree.fallbacks == 0, "no dense fallback was used");
        check(jfree.last_converged, "the last Krylov solve converged");
        check(jfree.preconditioner_updates > 0, "the user preconditioner was updated");
        check(jfree.preconditioner_applications > 0, "the user preconditioner was applied");
        check(jfree.saw_null_matrix, "the preconditioner received a null assembled matrix");
        check(jfree.saw_operator, "the preconditioner received the live operator");
        check(jfree.saw_empty_diagonal, "no assembled diagonal was passed");

        const std::string marker = "no Jacobian sparsity pattern";
        check(log.find(marker) == std::string::npos,
              "the J-free path does not warn about a missing pattern");

        std::cout << "  max|direct-jfree|=" << diff
                  << ", J*v applications=" << jfree.operator_applications
                  << ", preconditioner updates=" << jfree.preconditioner_updates
                  << ", fallbacks=" << jfree.fallbacks << "\n";
    }

    if (failures != 0) {
        std::cerr << failures << " J-free matrix-free check(s) failed" << std::endl;
        return 1;
    }
    std::cout << "PASS: J-free Newton-Krylov with no sparsity pattern or CSC assembly\n";
    return 0;
}
