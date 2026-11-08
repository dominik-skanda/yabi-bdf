/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

#include "test_support.hpp"

#include <yabi/csc_matrix.hpp>
#include <yabi/detail/dense_linear_solver.ipp>
#include <yabi/sparsity_pattern.hpp>

#include <cmath>
#include <vector>

namespace {

// Dense column-major P = [[2, 1], [1, 3]].
const double kP[4] = {2.0, 1.0, 1.0, 3.0};

void testDensePath() {
    yabi::DenseLUSolver<double> solver;
    yabi_test::check(solver.initialize(2), "DenseLUSolver::initialize");

    double P[4] = {kP[0], kP[1], kP[2], kP[3]};
    int ipiv[2] = {0, 0};
    long d1[2] = {0, 0};
    long d2[2] = {0, 0};
    yabi_test::check(solver.factorize(P, ipiv, d1, d2, 2), "DenseLUSolver::factorize");

    double b[2] = {5.0, 6.0};
    yabi_test::check(solver.solve(P, ipiv, d1, d2, b, 2, 1) == 0, "DenseLUSolver::solve");
    yabi_test::check(yabi_test::near(b[0], 1.8) && yabi_test::near(b[1], 1.4),
                     "DenseLUSolver solves P*x=b");

    // Multiple right-hand sides: same matrix, b = [5, 7; 6, 10] column-major.
    double P2[4] = {kP[0], kP[1], kP[2], kP[3]};
    int ipiv2[2] = {0, 0};
    yabi_test::check(solver.factorize(P2, ipiv2, d1, d2, 2), "DenseLUSolver::factorize (2 rhs)");
    double rhs[4] = {5.0, 6.0, 7.0, 10.0};
    yabi_test::check(solver.solve(P2, ipiv2, d1, d2, rhs, 2, 2) == 0,
                     "DenseLUSolver::solve (2 rhs)");
    yabi_test::check(yabi_test::near(rhs[0], 1.8) && yabi_test::near(rhs[1], 1.4) &&
                         yabi_test::near(rhs[2], 2.2) && yabi_test::near(rhs[3], 2.6),
                     "DenseLUSolver solves both right-hand sides");
}

// The reference CSC path must reproduce the dense LAPACK path exactly.
void testCscReferencePath() {
    yabi::SparsityPattern pattern;
    pattern.rows = 2;
    pattern.cols = 2;
    pattern.column_rows = {{0, 1}, {0, 1}};

    yabi::CscMatrix<double> csc;
    yabi_test::check(csc.assignPattern(pattern, false), "reference: assign pattern");
    csc.val = {kP[0], kP[1], kP[2], kP[3]};

    yabi::DenseLUSolver<double> solver;
    yabi_test::check(solver.supportsCsc(), "DenseLUSolver advertises CSC support");
    yabi_test::check(!solver.prefersSparseFactorization(),
                     "DenseLUSolver is not a production sparse solver");

    int ipiv[2] = {0, 0};
    long d1[2] = {0, 0};
    long d2[2] = {0, 0};
    yabi_test::check(solver.factorizeCsc(csc, ipiv, d1, d2, 2), "reference: factorizeCsc");
    yabi_test::check(solver.cscFactorizations() == 1, "reference: factorization counted");

    double b[2] = {5.0, 6.0};
    yabi_test::check(solver.solveCsc(csc, ipiv, d1, d2, b, 2, 1) == 0, "reference: solveCsc");
    yabi_test::check(yabi_test::near(b[0], 1.8) && yabi_test::near(b[1], 1.4),
                     "reference CSC path solves P*x=b");

    // Same values through the dense path must match bit-for-bit.
    yabi::DenseLUSolver<double> denseSolver;
    double Pd[4] = {kP[0], kP[1], kP[2], kP[3]};
    int dipiv[2] = {0, 0};
    denseSolver.factorize(Pd, dipiv, d1, d2, 2);
    double bd[2] = {5.0, 6.0};
    denseSolver.solve(Pd, dipiv, d1, d2, bd, 2, 1);
    yabi_test::check(bd[0] == b[0] && bd[1] == b[1], "CSC reference path is bit-identical");
}

// The LinearSolver interface defaults: a solver that does not opt into CSC /
// matrix-free / forcing must report the documented defaults, and the optional
// setter no-ops must be harmless.
class MinimalSolver : public yabi::LinearSolver<double> {
  public:
    bool initialize(unsigned int) override { return true; }
    bool factorize(double *, int *, long *, long *, unsigned int) override { return false; }
    int solve(double *, int *, long *, long *, double *, unsigned int, unsigned int) override {
        return -1;
    }
};

void testInterfaceDefaults() {
    MinimalSolver s;
    yabi_test::check(!s.supportsCsc(), "default supportsCsc is false");
    yabi_test::check(!s.prefersSparseFactorization(), "default prefersSparseFactorization is false");
    yabi_test::check(!s.supportsMatrixFree(), "default supportsMatrixFree is false");
    yabi_test::check(s.requiresAssembledMatrix(), "default requiresAssembledMatrix is true");
    yabi_test::check(!s.hasForcingTerm(), "default hasForcingTerm is false");
    // The optional setter no-ops must not crash and must not change the flags.
    s.setMatrixFreeOperator(nullptr);
    s.setUserPreconditioner(nullptr);
    s.setForcingTerm(yabi::ForcingTerm<double>{});
    s.setNewtonLinearizationContext(yabi::NewtonLinearizationContext<double>{});
    s.setJacobianPattern(nullptr);
    s.setUserData(nullptr);
    yabi_test::check(!s.hasForcingTerm(), "no forcing after a null setForcingTerm");
}

} // namespace

int main() {
    testDensePath();
    testCscReferencePath();
    testInterfaceDefaults();
    return yabi_test::finish("dense/CSC linear solver unit tests");
}
