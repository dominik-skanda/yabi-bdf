// T33 — selectable Jacobian backends (pure / UserODERHS).
//
// Proves the explicit JacobianBackend seam works without CppAD: Dense, Sparse
// (analytic sparsity pattern) and MatrixFree (analytic J*v) reproduce the same
// Brusselator trajectory, and Sparse/MatrixFree without a pattern fail loudly
// instead of silently downgrading to dense.

#include <yabi/bdf_integrator.hpp>
#include <yabi/detail/dense_linear_solver.ipp>
#include <yabi/matrix_free_solver.hpp>
#include <yabi/user_ode_rhs.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
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

// Brusselator (same problem as system test 12 / pure_12).
constexpr int n = 2;
constexpr double A = 1.0;
constexpr double B = 3.0;

void ode(const double *y, double *f) {
    f[0] = A + y[0] * y[0] * y[1] - (B + 1.0) * y[0];
    f[1] = B * y[0] - y[0] * y[0] * y[1];
}

// Column-major analytic Jacobian J[i + n*j] = df_i/dy_j.
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
    pattern.column_rows = {{0, 1}, {0, 1}}; // Brusselator's 2x2 is structurally full.
    return pattern;
}

struct RunResult {
    bool ok = false;
    std::vector<double> y;
};

RunResult integrate(yabi::JacobianBackend backend, bool with_pattern, bool krylov) {
    yabi::UserODERHS<double> rhs(n, ode, jacobian);
    if (with_pattern)
        rhs.setSparsityPattern(brusselatorPattern());
    rhs.setJacobianVectorProduct(jacobianVectorProduct);

    yabi::BDFIntegrator<double> integrator(rhs);
    if (krylov) {
        auto *solver = new yabi::GmresLinearSolver<double>();
        solver->setPreconditioner(yabi::KrylovPreconditioner::Jacobi);
        solver->setTolerance(1e-12, 0.0, 200, 30);
        integrator.setLinearSolver(solver);
    }
    integrator.setJacobianBackend(backend);

    double y0[n] = {1.0, 2.0};
    double abs_err[n] = {1e-10, 1e-10};

    RunResult result;
    if (!integrator.resetIntegration(y0, 5.0, 1e-10, abs_err, false, false, NULL, NULL, NULL,
                                     NULL, 0))
        return result;
    if (!integrator.setAlgorithmSettings(5, std::numeric_limits<double>::epsilon(), true, 1))
        return result;
    if (!integrator.performIntegration(5.0, 10000000))
        return result;

    result.y.resize(n);
    if (!integrator.getSolutionVector(result.y.data()))
        return result;
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
    std::cout << "T33 selectable Jacobian backends (pure/UserODERHS)\n";

    const RunResult reference = integrate(yabi::JacobianBackend::Dense, false, false);
    check(reference.ok, "dense reference integration succeeds");

    const RunResult sparse = integrate(yabi::JacobianBackend::Sparse, true, false);
    const RunResult mfree = integrate(yabi::JacobianBackend::MatrixFree, true, true);

    check(sparse.ok, "Sparse integration succeeds");
    check(mfree.ok, "MatrixFree integration succeeds");
    check(maxDifference(sparse.y, reference.y) < 1e-9, "Sparse == reference");
    check(maxDifference(mfree.y, reference.y) < 1e-9, "MatrixFree == reference");

    // Explicit Sparse/MatrixFree without the analytic pattern fail loudly.
    const RunResult no_pattern_sparse = integrate(yabi::JacobianBackend::Sparse, false, false);
    check(!no_pattern_sparse.ok, "Sparse without a pattern fails loudly");
    const RunResult no_pattern_mfree =
        integrate(yabi::JacobianBackend::MatrixFree, false, true);
    check(!no_pattern_mfree.ok, "MatrixFree without a pattern fails loudly");

    if (failures != 0) {
        std::cerr << failures << " pure Jacobian-backend check(s) failed" << std::endl;
        return 1;
    }
    std::cout << "PASS: pure Dense/Sparse/MatrixFree agree and fail loudly when unsatisfiable\n";
    return 0;
}
