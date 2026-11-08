#pragma once

/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// EXPERIMENTAL (T36d) — sparse solver adapter delegating to AMGCL
// (https://github.com/ddemidov/amgcl), a header-only MIT algebraic-multigrid +
// Krylov library.  This is an EXAMPLE of the open solver architecture: it only
// implements the two solver-specific hooks of `ExternalSparseSolver` (the shared
// CSC->CSR handoff, factorize/solve plumbing, and nrhs loop are all inherited).
// Swapping AMGCL for Ginkgo/cuDSS/custom means writing another ~30-line adapter,
// never touching the integrator core.
//
// The BDF Newton matrix P = I - (h/l1) J is generally NON-symmetric, so the
// default is BiCGStab with an ILU(k) preconditioner (general).  AMG+CG (the
// classic AMGCL configuration) is the SPD-only alternative and is NOT the
// default.
//
// Completely optional: this header is never included by the default build; it
// needs AMGCL + Boost headers (see README) and is gated behind YABI_USE_AMGCL.

#ifndef YABI_AMGCL_LINEAR_SOLVER_HPP
#define YABI_AMGCL_LINEAR_SOLVER_HPP

#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/relaxation/iluk.hpp>
#include <amgcl/solver/bicgstab.hpp>

#include <cstddef>
#include <memory>
#include <tuple>
#include <vector>

#include <yabi/external_sparse_solver.hpp>

namespace yabi {

template <typename Scalar> class AmgclLinearSolver : public ExternalSparseSolver<Scalar> {
    using Backend = amgcl::backend::builtin<double>;
    // BiCGStab + ILU(k): general (non-symmetric) preconditioned iterative solve.
    using Precond = amgcl::relaxation::as_preconditioner<Backend, amgcl::relaxation::iluk>;
    using Solver = amgcl::make_solver<Precond, amgcl::solver::bicgstab<Backend>>;

  private:
    bool buildCsc(std::vector<int> &csr_ptr, std::vector<int> &csr_col,
                  std::vector<double> &csr_val) override {
        std::ptrdiff_t n = (std::ptrdiff_t)(csr_ptr.size() - 1);
        solver_.reset(new Solver(std::tie(n, csr_ptr, csr_col, csr_val)));
        return true;
    }

    bool solveOne(const std::vector<double> &rhs, std::vector<double> &x) override {
        std::size_t iters;
        double resid;
        std::tie(iters, resid) = solver_->operator()(rhs, x);
        (void)iters;
        (void)resid; // iterative; the BDF Newton drives its own convergence
        return true;
    }

    std::unique_ptr<Solver> solver_;
};

} // namespace yabi

#endif // YABI_AMGCL_LINEAR_SOLVER_HPP
