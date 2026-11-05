/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

#ifndef YABI_LINEAR_SOLVER_HPP
#define YABI_LINEAR_SOLVER_HPP

#include <vector>

#include <yabi/csc_matrix.hpp>

namespace yabi {

template <typename Scalar>
class LinearSolver {
public:
    virtual ~LinearSolver() {}

    // Inform solver of system size
    virtual bool initialize(unsigned int n) = 0;

    // Factor the linear system matrix P (n x n) in place as prepared by the
    // integrator. The integrator retains full control of:
    //  - Jacobian assembly J
    //  - formation of P = I - (h/l1)*J (and any future sparsity pattern)
    //  - Newton scaling (D1,D2) and pivot array handling as required
    // Implementations must not alter the mathematical form of P.
    virtual bool factorize(Scalar *P, int *ipiv, long *scalingD1, long *scalingD2,
                           unsigned int n) = 0;

    // Solve P * x = b in place for nrhs right hand sides using the
    // factorization produced by factorize(). The integrator applies/removes
    // Newton scaling before/after as it does today. The solve must be
    // mathematically identical to the dense LAPACK path.
    virtual int solve(Scalar *P, int *ipiv, long *scalingD1, long *scalingD2,
                      Scalar *b, unsigned int n, unsigned int nrhs) = 0;

    // ---- Native CSC interface (T31) ----
    //
    // A solver that can consume compressed sparse column storage overrides the
    // three methods below. The integrator uses them only when the RHS exposes a
    // Jacobian pattern and this solver reports supportsCsc(); otherwise the
    // dense path above is used unchanged.
    virtual bool supportsCsc() const { return false; }

    // True for solvers whose CSC path is the intended production path
    // (UMFPACK/KLU). The integrator auto-enables sparse assembly when such a
    // solver is installed; setSparseAssembly() can still override it.
    virtual bool prefersSparseFactorization() const { return false; }

    virtual bool factorizeCsc(const CscMatrix<Scalar> &P, int *ipiv, long *scalingD1,
                              long *scalingD2, unsigned int n) {
        (void)P;
        (void)ipiv;
        (void)scalingD1;
        (void)scalingD2;
        (void)n;
        return false;
    }

    virtual int solveCsc(const CscMatrix<Scalar> &P, int *ipiv, long *scalingD1, long *scalingD2,
                         Scalar *b, unsigned int n, unsigned int nrhs) {
        (void)P;
        (void)ipiv;
        (void)scalingD1;
        (void)scalingD2;
        (void)b;
        (void)n;
        (void)nrhs;
        return -1;
    }

    // Optional hooks for future sparse/direct iterative solvers
    virtual void setJacobianPattern(void *pattern) { (void)pattern; }
    virtual void setUserData(void *user_data) { (void)user_data; }
};

} // namespace yabi

#endif
