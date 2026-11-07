/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// T32c — preconditioner and inexact-Newton forcing contracts.

#include "test_support.hpp"

#include <yabi/csc_matrix.hpp>
#include <yabi/linear_solver.hpp>
#include <yabi/matrix_free_operator.hpp>
#include <yabi/matrix_free_solver.hpp>
#include <yabi/preconditioner.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <vector>

namespace {

class DenseOperator : public yabi::MatrixFreeOperator<double> {
  public:
    explicit DenseOperator(std::vector<double> matrix) : matrix_(std::move(matrix)) {
    }

    bool apply(const double *x, double *y) const override {
        const std::size_t n = (std::size_t)std::sqrt((double)matrix_.size());
        for (std::size_t i = 0; i < n; ++i) {
            double sum = 0.0;
            for (std::size_t j = 0; j < n; ++j)
                sum += matrix_[j * n + i] * x[j];
            y[i] = sum;
        }
        return true;
    }

  private:
    std::vector<double> matrix_;
};

yabi::CscMatrix<double> makeCsc(const std::vector<double> &dense, std::size_t n) {
    yabi::SparsityPattern pattern;
    pattern.rows = n;
    pattern.cols = n;
    pattern.column_rows.assign(n, {});
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < n; ++i)
            pattern.column_rows[j].push_back(i);

    yabi::CscMatrix<double> csc;
    csc.assignPattern(pattern, true);
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < n; ++i)
            csc.val[(std::size_t)csc.col_ptr[j] + i] = dense[j * n + i];
    return csc;
}

struct Problem {
    std::vector<double> matrix;
    std::vector<double> rhs;
};

Problem makeProblem() {
    Problem p;
    p.matrix = {4.0, 1.0, 0.0, 0.0, -1.0, 3.0, 1.0, 0.0, 0.0, -1.0, 5.0, 1.0, 0.0, 0.0, -1.0, 2.0};
    const std::vector<double> expected = {1.0, -2.0, 0.5, 3.0};
    p.rhs.assign(4, 0.0);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            p.rhs[i] += p.matrix[j * 4 + i] * expected[j];
    return p;
}

void testDiagonalExtraction() {
    const Problem p = makeProblem();
    yabi::CscMatrix<double> csc = makeCsc(p.matrix, 4);

    std::vector<double> diagonal;
    csc.extractDiagonal(diagonal);
    yabi_test::check(diagonal.size() == 4, "extractDiagonal returns n entries");
    double max_diff = 0.0;
    for (std::size_t i = 0; i < 4; ++i)
        max_diff = std::max(max_diff, std::fabs(diagonal[i] - p.matrix[i * 4 + i]));
    yabi_test::check(max_diff < 1e-15, "extractDiagonal matches value(i, i)");

    // A structurally missing diagonal entry is reported as zero.
    yabi::SparsityPattern pattern;
    pattern.rows = 2;
    pattern.cols = 2;
    pattern.column_rows = {{1}, {0, 1}};
    yabi::CscMatrix<double> off_diagonal;
    off_diagonal.assignPattern(pattern, false);
    off_diagonal.extractDiagonal(diagonal);
    yabi_test::check(diagonal.size() == 2 && diagonal[0] == 0.0,
                     "a missing diagonal entry is zero");
}

void testFunctionPreconditionerContract() {
    const Problem p = makeProblem();
    yabi::CscMatrix<double> csc = makeCsc(p.matrix, 4);
    DenseOperator op(p.matrix);

    bool update_called = false;
    bool saw_matrix = false;
    bool saw_operator = false;
    bool saw_diagonal = false;
    unsigned int update_n = 0;
    unsigned int apply_n = 0;

    yabi::FunctionPreconditioner<double> preconditioner(
        true,
        [&](const yabi::CscMatrix<double> *P, const yabi::MatrixFreeOperator<double> *matrix_free,
            const std::vector<double> &diagonal, unsigned int n) {
            update_called = true;
            saw_matrix = P != nullptr;
            saw_operator = matrix_free != nullptr;
            saw_diagonal = diagonal.size() == n && diagonal[0] == p.matrix[0];
            update_n = n;
            return true;
        },
        [&](const double *r, double *z, unsigned int n) {
            ++apply_n;
            for (unsigned int i = 0; i < n; ++i)
                z[i] = r[i];
            return true;
        });

    yabi_test::check(preconditioner.requiresAssembledMatrix(),
                     "FunctionPreconditioner reports its assembly requirement");
    yabi_test::check(preconditioner.update(&csc, &op, std::vector<double>{4.0, 3.0, 5.0, 2.0}, 4),
                     "FunctionPreconditioner update succeeds");
    yabi_test::check(update_called && saw_matrix && saw_operator && saw_diagonal,
                     "update receives P, operator, diagonal and n");
    yabi_test::check(update_n == 4, "update receives the system size");

    std::vector<double> r = p.rhs;
    std::vector<double> z(4, 0.0);
    yabi_test::check(preconditioner.apply(r.data(), z.data()),
                     "FunctionPreconditioner apply succeeds");
    yabi_test::check(apply_n == 1 && z == r, "apply forwards the callback result");
}

void testUserPreconditionerTakesPrecedence() {
    const Problem p = makeProblem();
    yabi::CscMatrix<double> csc = makeCsc(p.matrix, 4);
    DenseOperator op(p.matrix);

    unsigned int update_calls = 0;
    unsigned int apply_calls = 0;
    bool saw_null_matrix = false;

    // The built-in choice is None; the user object must still be used.
    yabi::GmresLinearSolver<double> solver;
    solver.initialize(4);
    solver.setPreconditioner(yabi::KrylovPreconditioner::None);
    solver.setTolerance(1e-14, 0.0, 100, 10);
    yabi::FunctionPreconditioner<double> identity(
        false,
        [&](const yabi::CscMatrix<double> *, const yabi::MatrixFreeOperator<double> *,
            const std::vector<double> &, unsigned int) {
            ++update_calls;
            return true;
        },
        [&](const double *r, double *z, unsigned int n) {
            ++apply_calls;
            (void)saw_null_matrix;
            for (unsigned int i = 0; i < n; ++i)
                z[i] = r[i];
            return true;
        });
    solver.setUserPreconditioner(&identity);
    yabi_test::check(!solver.requiresAssembledMatrix(),
                     "a user J-free preconditioner disables the assembly requirement");
    solver.setMatrixFreeOperator(&op);

    int ipiv[4] = {0, 0, 0, 0};
    long d1[4] = {0, 0, 0, 0};
    long d2[4] = {0, 0, 0, 0};
    yabi_test::check(solver.factorizeCsc(csc, ipiv, d1, d2, 4),
                     "user preconditioner factorization succeeds");
    std::vector<double> x = p.rhs;
    yabi_test::check(solver.solveCsc(csc, ipiv, d1, d2, x.data(), 4, 1) == 0,
                     "user preconditioner solve succeeds");
    yabi_test::check(update_calls == 1 && apply_calls > 0,
                     "the user preconditioner overrides the built-in enum");

    // The same contract on the genuinely J-free path: empty P, live operator.
    yabi::GmresLinearSolver<double> jfree;
    jfree.initialize(4);
    jfree.setTolerance(1e-14, 0.0, 100, 10);
    yabi::FunctionPreconditioner<double> jfree_preconditioner(
        false,
        [&](const yabi::CscMatrix<double> *P, const yabi::MatrixFreeOperator<double> *matrix_free,
            const std::vector<double> &diagonal, unsigned int n) {
            ++update_calls;
            saw_null_matrix = P == nullptr;
            return matrix_free != nullptr && diagonal.empty() && n == 4;
        },
        [&](const double *r, double *z, unsigned int n) {
            ++apply_calls;
            for (unsigned int i = 0; i < n; ++i)
                z[i] = r[i];
            return true;
        });
    jfree.setUserPreconditioner(&jfree_preconditioner);
    jfree.setMatrixFreeOperator(&op);
    yabi::CscMatrix<double> empty;
    yabi_test::check(jfree.factorizeCsc(empty, ipiv, d1, d2, 4),
                     "a J-free preconditioner updates from an empty matrix");
    std::vector<double> y = p.rhs;
    yabi_test::check(jfree.solveCsc(empty, ipiv, d1, d2, y.data(), 4, 1) == 0,
                     "a J-free preconditioner solves against the operator");
    yabi_test::check(saw_null_matrix, "the J-free update saw a null assembled matrix");
}

void testPreconditionerFailurePropagates() {
    const Problem p = makeProblem();
    yabi::CscMatrix<double> csc = makeCsc(p.matrix, 4);
    DenseOperator op(p.matrix);

    // update() failure must fail the factorization.
    yabi::GmresLinearSolver<double> solver;
    solver.initialize(4);
    solver.setTolerance(1e-14, 0.0, 100, 10);
    solver.setMatrixFreeOperator(&op);
    yabi::FunctionPreconditioner<double> bad_update(
        true,
        [](const yabi::CscMatrix<double> *, const yabi::MatrixFreeOperator<double> *,
           const std::vector<double> &, unsigned int) { return false; },
        [](const double *r, double *z, unsigned int n) {
            for (unsigned int i = 0; i < n; ++i)
                z[i] = r[i];
            return true;
        });
    solver.setUserPreconditioner(&bad_update);
    int ipiv[4] = {0, 0, 0, 0};
    long d1[4] = {0, 0, 0, 0};
    long d2[4] = {0, 0, 0, 0};
    yabi_test::check(!solver.factorizeCsc(csc, ipiv, d1, d2, 4),
                     "a failed preconditioner update fails factorization");

    // apply() failure is handled by the existing assembled direct fallback.
    yabi::GmresLinearSolver<double> fallback;
    fallback.initialize(4);
    fallback.setTolerance(1e-14, 0.0, 100, 10);
    fallback.setMatrixFreeOperator(&op);
    yabi::FunctionPreconditioner<double> bad_apply(
        true,
        [](const yabi::CscMatrix<double> *, const yabi::MatrixFreeOperator<double> *,
           const std::vector<double> &, unsigned int) { return true; },
        [](const double *, double *, unsigned int) { return false; });
    fallback.setUserPreconditioner(&bad_apply);
    yabi_test::check(fallback.factorizeCsc(csc, ipiv, d1, d2, 4),
                     "factorization succeeds before the apply failure");
    std::vector<double> x = p.rhs;
    yabi_test::check(fallback.solveCsc(csc, ipiv, d1, d2, x.data(), 4, 1) == 0,
                     "apply failure falls back to the assembled direct solve");
    yabi_test::check(fallback.fallbackCount() == 1, "the fallback is observable");
}

void testForcingTerm() {
    const Problem p = makeProblem();
    yabi::CscMatrix<double> csc = makeCsc(p.matrix, 4);
    DenseOperator op(p.matrix);
    int ipiv[4] = {0, 0, 0, 0};
    long d1[4] = {0, 0, 0, 0};
    long d2[4] = {0, 0, 0, 0};

    // Default: no callback -> fixed relative tolerance.
    yabi::GmresLinearSolver<double> fixed;
    fixed.initialize(4);
    fixed.setTolerance(1e-13, 0.0, 100, 10);
    fixed.setMatrixFreeOperator(&op);
    fixed.factorizeCsc(csc, ipiv, d1, d2, 4);
    std::vector<double> x = p.rhs;
    fixed.solveCsc(csc, ipiv, d1, d2, x.data(), 4, 1);
    yabi_test::check(!fixed.hasForcingTerm(), "no forcing term is installed by default");
    yabi_test::check(fixed.lastRelativeTolerance() == 1e-13,
                     "the fixed tolerance is used without a forcing term");

    // A callback changes the per-solve tolerance and sees the context.
    yabi::GmresLinearSolver<double> driven;
    driven.initialize(4);
    driven.setTolerance(1e-13, 0.0, 100, 10);
    driven.setMatrixFreeOperator(&op);
    yabi::NewtonLinearizationContext<double> recorded;
    unsigned int calls = 0;
    driven.setForcingTerm([&](const yabi::NewtonLinearizationContext<double> &context) {
        ++calls;
        recorded = context;
        return 0.25;
    });
    yabi::NewtonLinearizationContext<double> context;
    context.iteration = 3;
    context.residual = 1e-3;
    context.previous_residual = 2e-2;
    context.nonlinear_tolerance = 1e-8;
    driven.setNewtonLinearizationContext(context);
    driven.factorizeCsc(csc, ipiv, d1, d2, 4);
    std::vector<double> y = p.rhs;
    driven.solveCsc(csc, ipiv, d1, d2, y.data(), 4, 1);
    yabi_test::check(driven.hasForcingTerm(), "a forcing term is reported installed");
    yabi_test::check(calls == 1, "the forcing callback runs once per solve");
    yabi_test::check(recorded.iteration == 3 && recorded.residual == 1e-3 &&
                         recorded.previous_residual == 2e-2 &&
                         recorded.nonlinear_tolerance == 1e-8,
                     "the forcing callback receives the Newton context");
    yabi_test::check(driven.lastRelativeTolerance() == 0.25,
                     "the forcing callback's tolerance is used");

    // Invalid callback results are ignored rather than poisoning the solve.
    yabi::GmresLinearSolver<double> invalid;
    invalid.initialize(4);
    invalid.setTolerance(1e-11, 0.0, 100, 10);
    invalid.setMatrixFreeOperator(&op);
    invalid.setForcingTerm(
        [](const yabi::NewtonLinearizationContext<double> &) {
            return std::numeric_limits<double>::quiet_NaN();
        });
    invalid.factorizeCsc(csc, ipiv, d1, d2, 4);
    std::vector<double> z = p.rhs;
    invalid.solveCsc(csc, ipiv, d1, d2, z.data(), 4, 1);
    yabi_test::check(invalid.lastRelativeTolerance() == 1e-11,
                     "a non-finite forcing value falls back to the fixed tolerance");
}

void testEisenstatWalker() {
    const yabi::ForcingTerm<double> forcing = yabi::eisenstatWalkerForcing<double>();

    yabi::NewtonLinearizationContext<double> context;
    context.iteration = 1;
    context.residual = 1.0;
    yabi_test::check(forcing(context) == 0.9, "the first iteration uses max_eta");

    context.iteration = 2;
    context.previous_residual = 1.0;
    context.residual = 0.1;
    double eta = forcing(context);
    yabi_test::check(std::fabs(eta - 0.9 * 0.1 * 0.1) < 1e-15,
                     "later iterations use gamma*(r_k/r_{k-1})^alpha");

    context.residual = 10.0;
    yabi_test::check(forcing(context) == 0.9, "a growing residual is clamped to max_eta");

    context.residual = 1e-20;
    yabi_test::check(forcing(context) > 0.0, "a tiny ratio is clamped to a positive min_eta");

    context.residual = std::numeric_limits<double>::infinity();
    yabi_test::check(forcing(context) == 0.9, "a non-finite ratio uses max_eta");

    const yabi::ForcingTerm<double> tight = yabi::eisenstatWalkerForcing<double>(0.5, 1.0, 0.2, 1e-6);
    context.iteration = 2;
    context.previous_residual = 1.0;
    context.residual = 1.0;
    yabi_test::check(tight(context) == 0.2, "custom max_eta is respected");
}

} // namespace

int main() {
    testDiagonalExtraction();
    testFunctionPreconditionerContract();
    testUserPreconditionerTakesPrecedence();
    testPreconditionerFailurePropagates();
    testForcingTerm();
    testEisenstatWalker();
    return yabi_test::finish("preconditioner / forcing unit tests");
}
