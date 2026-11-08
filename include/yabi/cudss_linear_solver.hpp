#pragma once

/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// EXPERIMENTAL (T36d) — sparse *direct* solver adapter delegating to NVIDIA cuDSS
// (the Direct Sparse Solver library, distributed via `nvidia-cudss-cu12`).  This
// is a third EXAMPLE of the open solver architecture: it only implements the two
// solver-specific hooks of `ExternalSparseSolver` (the shared CSC->CSR handoff,
// factorize/solve plumbing, and nrhs loop are all inherited).
//
// cuDSS provides real phases — CUDSS_PHASE_ANALYSIS | FACTORIZATION in buildCsc,
// CUDSS_PHASE_SOLVE in solveOne — so the LU factors are computed once and reused
// across Newton iterations, matching the BDF factorize-once/solve-many pattern.
// (Unlike cusolverSp's sparse LU, which is host-only and one-shot.)
//
// Completely optional: this header is never included by the default build; it
// needs the cuDSS library + headers (see README) and is gated behind
// YABI_USE_CUDSS.  The library is never vendored — it comes from the official
// `nvidia-cudss-cu12` wheel.

#ifndef YABI_CUDSS_LINEAR_SOLVER_HPP
#define YABI_CUDSS_LINEAR_SOLVER_HPP

#include <cudss.h>

#include <cstddef>
#include <vector>

#include <yabi/external_sparse_solver.hpp>

namespace yabi {

template <typename Scalar> class CudssLinearSolver : public ExternalSparseSolver<Scalar> {
    using ValueType = double;
    using IndexType = int;

  public:
    CudssLinearSolver() {
        cudssCreate(&handle_);
        cudssConfigCreate(&config_);
        cudssDataCreate(handle_, &data_);
    }

    ~CudssLinearSolver() override {
        release();
        if (matrix_)
            cudssMatrixDestroy(matrix_);
        if (rhs_)
            cudssMatrixDestroy(rhs_);
        if (sol_)
            cudssMatrixDestroy(sol_);
        if (data_)
            cudssDataDestroy(handle_, data_);
        if (config_)
            cudssConfigDestroy(config_);
        if (handle_)
            cudssDestroy(handle_);
    }

  private:
    bool buildCsc(std::vector<int> &csr_ptr, std::vector<int> &csr_col,
                  std::vector<double> &csr_val) override {
        n_ = csr_ptr.size() - 1;
        nnz_ = csr_val.size();
        release();

        if (cudaMalloc(&d_rowPtr_, (n_ + 1) * sizeof(int)) != cudaSuccess ||
            cudaMalloc(&d_colInd_, nnz_ * sizeof(int)) != cudaSuccess ||
            cudaMalloc(&d_val_, nnz_ * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&d_rhs_, n_ * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&d_x_, n_ * sizeof(double)) != cudaSuccess)
            return false;
        cudaMemcpy(d_rowPtr_, csr_ptr.data(), (n_ + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_colInd_, csr_col.data(), nnz_ * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_val_, csr_val.data(), nnz_ * sizeof(double), cudaMemcpyHostToDevice);

        cudssMatrixCreateCsr(&matrix_, (int64_t)n_, (int64_t)n_, (int64_t)nnz_, d_rowPtr_, NULL,
                             d_colInd_, d_val_, CUDA_R_32I, CUDA_R_32I, CUDA_R_64F,
                             CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO);
        cudssMatrixCreateDn(&rhs_, (int64_t)n_, 1, (int64_t)n_, d_rhs_, CUDA_R_64F,
                            CUDSS_LAYOUT_COL_MAJOR);
        cudssMatrixCreateDn(&sol_, (int64_t)n_, 1, (int64_t)n_, d_x_, CUDA_R_64F,
                            CUDSS_LAYOUT_COL_MAJOR);

        cudssStatus_t st = cudssExecute(handle_, CUDSS_PHASE_ANALYSIS | CUDSS_PHASE_FACTORIZATION,
                                        config_, data_, matrix_, sol_, rhs_);
        cudaDeviceSynchronize();
        return st == CUDSS_STATUS_SUCCESS;
    }

    bool solveOne(const std::vector<double> &rhs, std::vector<double> &x) override {
        cudaMemcpy(d_rhs_, rhs.data(), n_ * sizeof(double), cudaMemcpyHostToDevice);
        cudssExecute(handle_, CUDSS_PHASE_SOLVE, config_, data_, matrix_, sol_, rhs_);
        cudaDeviceSynchronize();
        cudaMemcpy(x.data(), d_x_, n_ * sizeof(double), cudaMemcpyDeviceToHost);
        return true;
    }

    void release() {
        if (d_rowPtr_)
            cudaFree(d_rowPtr_);
        if (d_colInd_)
            cudaFree(d_colInd_);
        if (d_val_)
            cudaFree(d_val_);
        if (d_rhs_)
            cudaFree(d_rhs_);
        if (d_x_)
            cudaFree(d_x_);
        d_rowPtr_ = d_colInd_ = nullptr;
        d_val_ = d_rhs_ = d_x_ = nullptr;
    }

    cudssHandle_t handle_ = nullptr;
    cudssConfig_t config_ = nullptr;
    cudssData_t data_ = nullptr;
    cudssMatrix_t matrix_ = nullptr;
    cudssMatrix_t rhs_ = nullptr;
    cudssMatrix_t sol_ = nullptr;
    int *d_rowPtr_ = nullptr;
    int *d_colInd_ = nullptr;
    double *d_val_ = nullptr;
    double *d_rhs_ = nullptr;
    double *d_x_ = nullptr;
    std::size_t n_ = 0;
    std::size_t nnz_ = 0;
};

} // namespace yabi

#endif // YABI_CUDSS_LINEAR_SOLVER_HPP
