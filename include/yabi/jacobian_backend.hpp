#pragma once

namespace yabi {

// How the Newton-matrix Jacobian J is formed for the *plain state* solve
// (the solve with no simultaneous-derivative seeds and no replay taping).
//
// The Jacobian *source* is orthogonal and is determined by the AbstractRHS you
// construct: CppADRHS provides it by AD, UserODERHS by analytic callbacks.  Both
// sources feed every representation below, so "user-analytic" is not a separate
// backend value — it is the source, and it flows through Dense/Sparse/MatrixFree
// identically (dense via Forward(1) columns, sparse via sparseJacobian(),
// matrix-free via jacobianVectorProduct()).
//
// Sensitivity solves (simultaneous seeds, or a taped forward/reverse replay)
// always use the explicit dense J/H path regardless of this setting — the
// sparse and matrix-free paths cover the plain state Newton solve only, and are
// documented as such (see T32/T33).
enum class JacobianBackend {
    // Capability-driven selection (the historical behaviour): dense by default,
    // sparse assembly when a CSC-preferred solver (UMFPACK/KLU) is installed,
    // matrix-free when a Krylov solver is installed.  The integrator falls back
    // silently exactly as it did before this seam existed.
    Auto,

    // Dense J/P via Forward(1) directional columns, dense LAPACK solve.
    // Always available.  This is the AD/analytic "dense" backend.
    Dense,

    // Sparse J/P in CSC via sparseJacobian(), factorized by a CSC-capable
    // solver.  Requires the RHS to expose a Jacobian sparsity pattern and a
    // solver with supportsCsc(); otherwise the solve fails loudly.
    Sparse,

    // Matrix-free J*v via jacobianVectorProduct(), solved by a Krylov solver.
    // Requires a solver with supportsMatrixFree(); otherwise the solve fails
    // loudly.  If the solver's preconditioner needs an assembled matrix, a
    // sparsity pattern is additionally required (as in the Auto path).
    MatrixFree
};

} // namespace yabi
