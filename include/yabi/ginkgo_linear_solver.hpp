#pragma once

/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// EXPERIMENTAL (T36d) — sparse solver adapter delegating to Ginkgo
// (https://github.com/ginkgo-project/ginkgo), a BSD numerical linear-algebra
// library with many-core (OpenMP/CUDA/HIP) backends.  This is a second EXAMPLE
// of the open solver architecture: it only implements the two solver-specific
// hooks of `ExternalSparseSolver` (the shared CSC->CSR handoff, factorize/solve
// plumbing, and nrhs loop are all inherited).  It uses the OpenMP (CPU) backend,
// so it works without a GPU; the CUDA backend is the same adapter with a
// `CudaExecutor` (and a supported GPU architecture).
//
// The BDF Newton matrix P = I - (h/l1) J is generally NON-symmetric, so the
// default is BiCGStab (general).  An ILU preconditioner can be added if a
// problem needs it.
//
// Completely optional: this header is never included by the default build; it
// needs the Ginkgo library + headers (see README) and is gated behind
// YABI_USE_GINKGO.

#ifndef YABI_GINKGO_LINEAR_SOLVER_HPP
#define YABI_GINKGO_LINEAR_SOLVER_HPP

#include <ginkgo/ginkgo.hpp>

#include <cstddef>
#include <memory>
#include <vector>

#include <yabi/external_sparse_solver.hpp>

namespace yabi {

template <typename Scalar> class GinkgoLinearSolver : public ExternalSparseSolver<Scalar> {
    using ValueType = double;
    using IndexType = int;
    using Csr = gko::matrix::Csr<ValueType, IndexType>;
    using Dense = gko::matrix::Dense<ValueType>;

  public:
    GinkgoLinearSolver() : exec_(gko::OmpExecutor::create()) {}
    ~GinkgoLinearSolver() override = default;

  private:
    bool buildCsc(std::vector<int> &csr_ptr, std::vector<int> &csr_col,
                  std::vector<double> &csr_val) override {
        n_ = csr_ptr.size() - 1;
        const std::size_t nnz = csr_val.size();

        gko::matrix_data<ValueType, IndexType> data(gko::dim<2>(n_, n_));
        data.nonzeros.reserve(nnz);
        for (std::size_t i = 0; i < n_; ++i)
            for (std::size_t k = (std::size_t)csr_ptr[i]; k < (std::size_t)csr_ptr[i + 1]; ++k)
                data.nonzeros.emplace_back((IndexType)i, csr_col[k], csr_val[k]);

        auto matrix = gko::share(Csr::create(exec_));
        matrix->read(data);

        solver_ = gko::solver::Bicgstab<ValueType>::build()
                      .with_criteria(
                          gko::stop::Iteration::build().with_max_iters(gko::size_type{2000}),
                          gko::stop::ResidualNorm<ValueType>::build().with_reduction_factor(1e-12))
                      .on(exec_)
                      ->generate(gko::give(matrix));
        return static_cast<bool>(solver_);
    }

    bool solveOne(const std::vector<double> &rhs, std::vector<double> &x) override {
        auto b = Dense::create(exec_, gko::dim<2>(n_, 1));
        auto xv = Dense::create(exec_, gko::dim<2>(n_, 1));
        for (std::size_t i = 0; i < n_; ++i) {
            b->get_values()[i] = rhs[i];
            xv->get_values()[i] = 0.0;
        }
        solver_->apply(b.get(), xv.get());
        const double *xh = xv->get_const_values();
        for (std::size_t i = 0; i < n_; ++i)
            x[i] = xh[i];
        return true;
    }

    std::shared_ptr<gko::OmpExecutor> exec_;
    std::unique_ptr<gko::LinOp> solver_;
    std::size_t n_ = 0;
};

} // namespace yabi

#endif // YABI_GINKGO_LINEAR_SOLVER_HPP
