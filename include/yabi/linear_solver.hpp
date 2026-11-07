/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

#ifndef YABI_LINEAR_SOLVER_HPP
#define YABI_LINEAR_SOLVER_HPP

#include <cmath>
#include <functional>
#include <vector>

#include <yabi/csc_matrix.hpp>
#include <yabi/matrix_free_operator.hpp>
#include <yabi/preconditioner.hpp>

namespace yabi {

// Nonlinear state handed to an inexact-Newton forcing callback.  The Krylov
// solver uses the returned value as the relative linear tolerance for the
// current Newton iteration.  `residual` / `previous_residual` are norms of the
// unscaled Newton residual F(x_k).
template <typename Scalar> struct NewtonLinearizationContext {
    unsigned int iteration = 0;
    Scalar residual = Scalar(0);
    Scalar previous_residual = Scalar(0);
    Scalar nonlinear_tolerance = Scalar(0);
};

template <typename Scalar>
using ForcingTerm =
    std::function<Scalar(const NewtonLinearizationContext<Scalar> &context)>;

// Eisenstat-Walker-style forcing:
//
//   eta_k = gamma * (||F_k|| / ||F_{k-1}||)^alpha
//
// clamped to [min_eta, max_eta].  The first iteration and degenerate residual
// ratios use max_eta.  Install with LinearSolver::setForcingTerm().
template <typename Scalar>
ForcingTerm<Scalar> eisenstatWalkerForcing(Scalar gamma = Scalar(0.9),
                                           Scalar alpha = Scalar(2),
                                           Scalar max_eta = Scalar(0.9),
                                           Scalar min_eta = Scalar(1e-12)) {
    return [=](const NewtonLinearizationContext<Scalar> &context) {
        Scalar eta = max_eta;
        if (context.iteration > 1 && context.previous_residual > Scalar(0) &&
            context.residual > Scalar(0) && std::isfinite(context.previous_residual) &&
            std::isfinite(context.residual))
            eta = gamma * std::pow(context.residual / context.previous_residual, alpha);
        if (!std::isfinite(eta))
            eta = max_eta;
        if (eta < min_eta)
            eta = min_eta;
        if (eta > max_eta)
            eta = max_eta;
        return eta;
    };
}

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

    // ---- Matrix-free interface (T32) ----
    //
    // A Krylov solver advertises supportsMatrixFree().  The integrator then
    // installs a non-owning operator for the current Newton matrix P and calls
    // factorizeCsc()/solveCsc() exactly as it does for a CSC solver.  The
    // operator outlives the solve and is only used by the solver that owns it.
    virtual bool supportsMatrixFree() const { return false; }

    virtual void setMatrixFreeOperator(const MatrixFreeOperator<Scalar> *op) { (void)op; }

    // ---- Pluggable preconditioners / inexact-Newton forcing (T32c) ----
    //
    // A user preconditioner takes precedence over the solver's built-in
    // choice.  `requiresAssembledMatrix()` defaults to true so existing solvers
    // keep the historical assembled path; a solver may override it when its
    // current configuration can stay matrix-free.  The solver borrows the
    // preconditioner; the caller keeps ownership and lifetime.
    virtual void setUserPreconditioner(Preconditioner<Scalar> *preconditioner) {
        (void)preconditioner;
    }
    virtual bool requiresAssembledMatrix() const { return true; }

    // Optional inexact-Newton forcing.  Without a callback the solver keeps
    // the fixed relative tolerance from its own configuration.
    virtual void setForcingTerm(ForcingTerm<Scalar> forcing) { (void)forcing; }
    virtual bool hasForcingTerm() const { return false; }
    virtual void setNewtonLinearizationContext(const NewtonLinearizationContext<Scalar> &context) {
        (void)context;
    }

    // Optional hooks for other sparse/direct iterative solvers
    virtual void setJacobianPattern(void *pattern) { (void)pattern; }
    virtual void setUserData(void *user_data) { (void)user_data; }
};

} // namespace yabi

#endif
