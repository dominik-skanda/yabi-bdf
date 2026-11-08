#pragma once

/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// The reusable, solver-agnostic half of the OPEN solver architecture (T36d).
//
// `ExternalSparseSolver<Scalar>` implements the `LinearSolver<Scalar>` CSC seam
// once — the CSC->CSR handoff (the common format every external sparse solver
// accepts), the factorize-once / solve-many plumbing, and the nrhs loop + buffer
// management — and leaves only two pure-virtual hooks for a concrete solver:
//
//   buildCsc(ptr, col, val)   build/analyze/factorize the solver from CSR arrays
//   solveOne(rhs, x)          solve ONE right-hand side into x
//
// So a new solver (AMGCL, Ginkgo, cuDSS, a commercial library, a custom one) is
// a ~30-line adapter that only speaks its own library; everything else is
// shared.  The integrator core never knows which solver is installed — it only
// sees the generic `LinearSolver` seam, and the solver is reached only through
// `setLinearSolver(...)`.  This header is never included by the default build.
//
// Only `double` is supported (matching the single BDFIntegrator instantiation).

#ifndef YABI_EXTERNAL_SPARSE_SOLVER_HPP
#define YABI_EXTERNAL_SPARSE_SOLVER_HPP

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <vector>

#include <yabi/linear_solver.hpp>

namespace yabi {

template <typename Scalar> class ExternalSparseSolver : public LinearSolver<Scalar> {
    static_assert(std::is_same<Scalar, double>::value,
                  "ExternalSparseSolver supports double only (single library instantiation)");

  public:
    bool initialize(unsigned int) override { return true; }

    // Dense path unsupported — this is a sparse (CSC) solver.
    bool factorize(Scalar *, int *, long *, long *, unsigned int) override { return false; }
    int solve(Scalar *, int *, long *, long *, Scalar *, unsigned int, unsigned int) override {
        return -1;
    }

    bool supportsCsc() const override { return true; }
    bool prefersSparseFactorization() const override { return true; }

    bool factorizeCsc(const CscMatrix<Scalar> &P, int *, long *, long *, unsigned int n) override {
        if (n != P.n || P.empty())
            return false;
        cscToCsr(P);
        return buildCsc(csr_ptr_, csr_col_, csr_val_);
    }

    int solveCsc(const CscMatrix<Scalar> &P, int *, long *, long *, Scalar *b, unsigned int n,
                 unsigned int nrhs) override {
        (void)P; // the solver was built in factorizeCsc from the same matrix
        std::vector<Scalar> rhs(n), x(n);
        for (unsigned int c = 0; c < nrhs; ++c) {
            std::copy(b + (std::size_t)c * n, b + (std::size_t)(c + 1) * n, rhs.begin());
            std::fill(x.begin(), x.end(), Scalar(0));
            if (!solveOne(rhs, x))
                return -1;
            std::copy(x.begin(), x.end(), b + (std::size_t)c * n);
        }
        return 0;
    }

  protected:
    // CSC (column-oriented CscMatrix) -> CSR (row pointers + column indices),
    // the common handoff format for external sparse solvers.
    void cscToCsr(const CscMatrix<Scalar> &P) {
        const std::size_t n = P.n;
        const std::size_t nnz = P.nnz();
        csr_ptr_.assign(n + 1, 0);
        csr_col_.resize(nnz);
        csr_val_.resize(nnz);
        for (std::size_t k = 0; k < nnz; ++k)
            csr_ptr_[(std::size_t)P.row_idx[k] + 1]++;
        for (std::size_t i = 0; i < n; ++i)
            csr_ptr_[i + 1] += csr_ptr_[i];
        std::vector<std::size_t> cursor(csr_ptr_.begin(), csr_ptr_.begin() + n);
        for (std::size_t j = 0; j < n; ++j) {
            for (int k = P.col_ptr[j]; k < P.col_ptr[j + 1]; ++k) {
                const std::size_t r = (std::size_t)P.row_idx[(std::size_t)k];
                const std::size_t pos = cursor[r]++;
                csr_col_[pos] = (int)j;
                csr_val_[pos] = P.val[(std::size_t)k];
            }
        }
    }

    // Build/analyze/factorize the concrete solver from the CSR arrays (which the
    // base just filled from the CSC matrix in factorizeCsc).
    virtual bool buildCsc(std::vector<int> &csr_ptr, std::vector<int> &csr_col,
                          std::vector<Scalar> &csr_val) = 0;

    // Solve ONE right-hand side.  x is the output (zero on entry).
    virtual bool solveOne(const std::vector<Scalar> &rhs, std::vector<Scalar> &x) = 0;

    std::vector<int> csr_ptr_;
    std::vector<int> csr_col_;
    std::vector<Scalar> csr_val_;
};

} // namespace yabi

#endif // YABI_EXTERNAL_SPARSE_SOLVER_HPP
