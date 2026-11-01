/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

#ifndef YABI_LINEAR_SOLVER_HPP
#define YABI_LINEAR_SOLVER_HPP

#include <vector>

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

    // Optional hooks for future sparse/direct iterative solvers
    virtual void setJacobianPattern(void *pattern) { (void)pattern; }
    virtual void setUserData(void *user_data) { (void)user_data; }
};

} // namespace yabi

#endif
