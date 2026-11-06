/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

#include "test_support.hpp"

#include <yabi/csc_matrix.hpp>
#include <yabi/matrix_free_operator.hpp>
#include <yabi/matrix_free_solver.hpp>
#include <yabi/user_ode_rhs.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

// Dense matrix acting on vectors without ever exposing a matrix to the solver:
// this is exactly the contract a matrix-free Newton operator fulfils.
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

std::size_t indexInColumn(const yabi::CscMatrix<double> &csc, std::size_t j, std::size_t row) {
    for (int k = csc.col_ptr[j]; k < csc.col_ptr[j + 1]; ++k)
        if ((std::size_t)csc.row_idx[(std::size_t)k] == row)
            return (std::size_t)k - (std::size_t)csc.col_ptr[j];
    return 0;
}

yabi::CscMatrix<double> makeCsc(const std::vector<double> &dense, std::size_t n) {
    yabi::SparsityPattern pattern;
    pattern.rows = n;
    pattern.cols = n;
    pattern.column_rows.assign(n, {});
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < n; ++i)
            if (dense[j * n + i] != 0.0)
                pattern.column_rows[j].push_back(i);

    yabi::CscMatrix<double> csc;
    csc.assignPattern(pattern, true);
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < n; ++i)
            if (dense[j * n + i] != 0.0)
                csc.val[(std::size_t)csc.col_ptr[j] + indexInColumn(csc, j, i)] = dense[j * n + i];
    return csc;
}

struct Problem {
    std::vector<double> matrix; // column-major
    std::vector<double> rhs;
    std::vector<double> expected;
};

Problem makeProblem() {
    Problem p;
    // Nonsymmetric, diagonally dominant 4x4 system.
    p.matrix = {4.0, 1.0, 0.0, 0.0, -1.0, 3.0, 1.0, 0.0, 0.0, -1.0, 5.0, 1.0, 0.0, 0.0, -1.0, 2.0};
    p.expected = {1.0, -2.0, 0.5, 3.0};
    p.rhs.assign(4, 0.0);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            p.rhs[i] += p.matrix[j * 4 + i] * p.expected[j];
    return p;
}

template <typename Solver>
void checkSolve(const char *name, yabi::KrylovPreconditioner preconditioner) {
    const Problem p = makeProblem();
    const std::size_t n = 4;

    DenseOperator op(p.matrix);
    yabi::CscMatrix<double> csc = makeCsc(p.matrix, n);

    Solver solver;
    solver.initialize((unsigned int)n);
    solver.setPreconditioner(preconditioner);
    solver.setTolerance(1e-14, 0.0, 100, 10);
    solver.setMatrixFreeOperator(&op);

    int ipiv[4] = {0, 0, 0, 0};
    long d1[4] = {0, 0, 0, 0};
    long d2[4] = {0, 0, 0, 0};
    yabi_test::check(solver.factorizeCsc(csc, ipiv, d1, d2, (unsigned int)n),
                     std::string(name) + ": preconditioner factorization");

    std::vector<double> x = p.rhs;
    yabi_test::check(solver.solveCsc(csc, ipiv, d1, d2, x.data(), (unsigned int)n, 1) == 0,
                     std::string(name) + ": solve succeeds");
    yabi_test::check(solver.lastConverged(), std::string(name) + ": Krylov converged");
    yabi_test::check(solver.fallbackCount() == 0, std::string(name) + ": no direct fallback");

    double max_error = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        max_error = std::max(max_error, std::fabs(x[i] - p.expected[i]));
    yabi_test::check(max_error < 1e-10, std::string(name) + ": solution matches direct solve");
}

void testMethods() {
    checkSolve<yabi::GmresLinearSolver<double>>("GMRES", yabi::KrylovPreconditioner::Jacobi);
    checkSolve<yabi::FgmresLinearSolver<double>>("FGMRES", yabi::KrylovPreconditioner::Jacobi);
    checkSolve<yabi::BicgstabLinearSolver<double>>("BiCGSTAB", yabi::KrylovPreconditioner::Jacobi);

    checkSolve<yabi::GmresLinearSolver<double>>("GMRES/none", yabi::KrylovPreconditioner::None);
    checkSolve<yabi::FgmresLinearSolver<double>>("FGMRES/direct",
                                                 yabi::KrylovPreconditioner::Direct);

    yabi::GmresLinearSolver<double> gmres;
    yabi::FgmresLinearSolver<double> fgmres;
    yabi::BicgstabLinearSolver<double> bicgstab;
    yabi_test::check(gmres.method() == yabi::KrylovMethod::Gmres, "GMRES method tag");
    yabi_test::check(fgmres.method() == yabi::KrylovMethod::Fgmres, "FGMRES method tag");
    yabi_test::check(bicgstab.method() == yabi::KrylovMethod::Bicgstab, "BiCGSTAB method tag");
    yabi_test::check(gmres.supportsMatrixFree() && !gmres.supportsCsc(),
                     "Krylov solver advertises matrix-free, not CSC direct");

    // Multiple right-hand sides are solved independently.
    const Problem p = makeProblem();
    DenseOperator op(p.matrix);
    yabi::CscMatrix<double> csc = makeCsc(p.matrix, 4);
    yabi::GmresLinearSolver<double> solver;
    solver.initialize(4);
    solver.setTolerance(1e-14, 0.0, 100, 10);
    solver.setMatrixFreeOperator(&op);
    int ipiv[4] = {0, 0, 0, 0};
    long d1[4] = {0, 0, 0, 0};
    long d2[4] = {0, 0, 0, 0};
    solver.factorizeCsc(csc, ipiv, d1, d2, 4);
    std::vector<double> b(8);
    for (std::size_t i = 0; i < 4; ++i) {
        b[i] = p.rhs[i];
        b[4 + i] = 2.0 * p.rhs[i];
    }
    yabi_test::check(solver.solveCsc(csc, ipiv, d1, d2, b.data(), 4, 2) == 0,
                     "two right-hand sides solve");
    double max_error = 0.0;
    for (std::size_t i = 0; i < 4; ++i) {
        max_error = std::max(max_error, std::fabs(b[i] - p.expected[i]));
        max_error = std::max(max_error, std::fabs(b[4 + i] - 2.0 * p.expected[i]));
    }
    yabi_test::check(max_error < 1e-10, "both right-hand sides match");
}

void testFallback() {
    const Problem p = makeProblem();
    DenseOperator op(p.matrix);
    yabi::CscMatrix<double> csc = makeCsc(p.matrix, 4);

    // Too few iterations: the Krylov solve cannot converge, so the solver must
    // fall back to the direct factorization and still return the right answer.
    yabi::GmresLinearSolver<double> solver;
    solver.initialize(4);
    solver.setPreconditioner(yabi::KrylovPreconditioner::None);
    solver.setTolerance(1e-15, 0.0, 1, 1);
    solver.setMatrixFreeOperator(&op);
    int ipiv[4] = {0, 0, 0, 0};
    long d1[4] = {0, 0, 0, 0};
    long d2[4] = {0, 0, 0, 0};
    solver.factorizeCsc(csc, ipiv, d1, d2, 4);

    std::vector<double> x = p.rhs;
    yabi_test::check(solver.solveCsc(csc, ipiv, d1, d2, x.data(), 4, 1) == 0,
                     "failed Krylov solve falls back successfully");
    yabi_test::check(solver.fallbackCount() == 1, "fallback counter records the direct fallback");
    double max_error = 0.0;
    for (std::size_t i = 0; i < 4; ++i)
        max_error = std::max(max_error, std::fabs(x[i] - p.expected[i]));
    yabi_test::check(max_error < 1e-12, "direct fallback solution is exact");

    // No operator installed: the solver is a safe direct solver.
    yabi::BicgstabLinearSolver<double> direct;
    direct.initialize(4);
    direct.factorizeCsc(csc, ipiv, d1, d2, 4);
    std::vector<double> y = p.rhs;
    yabi_test::check(direct.solveCsc(csc, ipiv, d1, d2, y.data(), 4, 1) == 0,
                     "solver without an operator solves directly");
    max_error = 0.0;
    for (std::size_t i = 0; i < 4; ++i)
        max_error = std::max(max_error, std::fabs(y[i] - p.expected[i]));
    yabi_test::check(max_error < 1e-12, "solver without an operator returns the exact solution");
}

// The default AbstractRHS::jacobianVectorProduct is the non-CppAD path: it must
// still yield the exact analytic J*v for a pure UserODERHS backend.
void testAnalyticJacobianVectorProduct() {
    const std::size_t n = 3;
    auto f = [](const double *x, double *out) {
        out[0] = -x[0] + x[1] * x[2];
        out[1] = x[0] - 2.0 * x[1];
        out[2] = x[0] * x[1] - x[2];
    };
    auto jac = [n](const double *x, double *J) {
        J[0 + n * 0] = -1.0;
        J[1 + n * 0] = x[2];
        J[2 + n * 0] = x[1];
        J[0 + n * 1] = 1.0;
        J[1 + n * 1] = -2.0;
        J[2 + n * 1] = x[0];
        J[0 + n * 2] = 0.0;
        J[1 + n * 2] = 0.0;
        J[2 + n * 2] = -1.0;
    };

    yabi::UserODERHS<double> rhs(n, f, jac);
    std::vector<double> x = {1.5, -0.5, 2.0};
    std::vector<double> v = {0.25, -1.0, 0.75};

    std::vector<double> jv;
    yabi_test::check(rhs.jacobianVectorProduct(x, v, jv),
                     "analytic jacobianVectorProduct succeeds");
    yabi_test::check(jv.size() == n, "analytic jacobianVectorProduct returns n entries");

    std::vector<double> J(n * n, 0.0);
    jac(x.data(), J.data());
    std::vector<double> expected(n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            expected[i] += J[i + n * j] * v[j];

    double max_diff = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        max_diff = std::max(max_diff, std::fabs(jv[i] - expected[i]));
    yabi_test::check(max_diff < 1e-14, "analytic J*v is exact");
}

// The dense factorize()/solve() interface is the unchanged explicit-J path used
// by sensitivity solves; a Krylov solver must delegate to it exactly.
void testDenseDelegation() {
    const Problem p = makeProblem();
    yabi::FgmresLinearSolver<double> solver;
    solver.initialize(4);

    std::vector<double> dense = p.matrix;
    int ipiv[4] = {0, 0, 0, 0};
    long d1[4] = {0, 0, 0, 0};
    long d2[4] = {0, 0, 0, 0};
    yabi_test::check(solver.factorize(dense.data(), ipiv, d1, d2, 4), "dense delegation factorize");
    std::vector<double> x = p.rhs;
    yabi_test::check(solver.solve(dense.data(), ipiv, d1, d2, x.data(), 4, 1) == 0,
                     "dense delegation solve");
    double max_error = 0.0;
    for (std::size_t i = 0; i < 4; ++i)
        max_error = std::max(max_error, std::fabs(x[i] - p.expected[i]));
    yabi_test::check(max_error < 1e-12, "dense delegation returns the direct solution");
}

} // namespace

int main() {
    testMethods();
    testAnalyticJacobianVectorProduct();
    testFallback();
    testDenseDelegation();
    return yabi_test::finish("Krylov solver unit tests");
}
