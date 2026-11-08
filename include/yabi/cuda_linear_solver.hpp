#pragma once

/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// EXPERIMENTAL (T36) — dense LU solver offloading the Newton linear solve to a
// GPU via cuSOLVER.  Only `double` is supported (matching the single
// BDFIntegrator instantiation).  Requires CUDA (built against 12.6) and an
// NVIDIA driver; compiled with the regular host compiler (no nvcc needed — the
// cuSOLVER host API is plain C), linked against `-lcusolver -lcudart`.
//
// Scope (honest): this offloads the *factorize + solve* of the Newton matrix P.
// The Jacobian J is still assembled on the host (CppAD / analytic); P is copied
// to the device, factored with cusolverDnDgetrf, and each solve is
// cusolverDnDgetrs on the device factors, with the result copied back.  The LU
// factors and pivots stay resident on the device between factorize() and
// solve(), matching the integrator's factorize-once / solve-per-iteration
// pattern.  It is the dense path only (supportsCsc() == false).
//
// Portability: the cuBLAS/cuSOLVER C API is stable across CUDA 12.6 -> 13; this
// solver is source-identical on both.  Only the build (arch flag / driver)
// changes.

#ifndef YABI_CUDA_LINEAR_SOLVER_HPP
#define YABI_CUDA_LINEAR_SOLVER_HPP

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cstddef>
#include <type_traits>

#include <yabi/linear_solver.hpp>

namespace yabi {

template <typename Scalar> class CudaLinearSolver : public LinearSolver<Scalar> {
    static_assert(std::is_same<Scalar, double>::value,
                  "CudaLinearSolver supports double only (single library instantiation)");

  public:
    CudaLinearSolver() { cusolverDnCreate(&handle_); }

    ~CudaLinearSolver() override {
        release();
        if (handle_)
            cusolverDnDestroy(handle_);
    }

    bool initialize(unsigned int n) override {
        if (n == 0)
            return false;
        if (n == n_)
            return true;
        release();
        n_ = n;
        if (cudaMalloc(&d_P_, (std::size_t)n * n * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&d_ipiv_, n * sizeof(int)) != cudaSuccess ||
            cudaMalloc(&d_info_, sizeof(int)) != cudaSuccess ||
            cudaMalloc(&d_b_, n * sizeof(double)) != cudaSuccess)
            return false;
        return true;
    }

    // Factor P = L U in place on the device; pivots stay resident.  The host P
    // is left unmodified (unlike the dense LAPACK path, which factorizes in
    // place); solve() uses the device factors.
    bool factorize(Scalar *P, int *ipiv, long * /*scalingD1*/, long * /*scalingD2*/,
                   unsigned int n) override {
        (void)P; // copied below
        if (!handle_ || n != n_)
            return false;
        if (lwork_ == 0) {
            cusolverDnDgetrf_bufferSize(handle_, (int)n, (int)n, d_P_, (int)n, &lwork_);
            if (cudaMalloc(&d_W_, (std::size_t)lwork_ * sizeof(double)) != cudaSuccess)
                return false;
        }
        if (cudaMemcpy(d_P_, P, (std::size_t)n * n * sizeof(double), cudaMemcpyHostToDevice) !=
            cudaSuccess)
            return false;
        cusolverDnDgetrf(handle_, (int)n, (int)n, d_P_, (int)n, d_W_, d_ipiv_, d_info_);
        cudaMemcpy(ipiv, d_ipiv_, n * sizeof(int), cudaMemcpyDeviceToHost); // interface contract
        int info = 0;
        cudaMemcpy(&info, d_info_, sizeof(int), cudaMemcpyDeviceToHost);
        return info == 0;
    }

    // Solve P x = b (n x nrhs, column-major b) using the resident LU factors.
    int solve(Scalar *P, int *ipiv, long * /*scalingD1*/, long * /*scalingD2*/, Scalar *b,
              unsigned int n, unsigned int nrhs) override {
        (void)P;   // factors already on device
        (void)ipiv; // pivots already on device
        if (!handle_ || n != n_)
            return -1;
        for (unsigned int c = 0; c < nrhs; ++c) {
            if (cudaMemcpy(d_b_, b + (std::size_t)c * n, n * sizeof(double),
                           cudaMemcpyHostToDevice) != cudaSuccess)
                return -1;
            cusolverDnDgetrs(handle_, CUBLAS_OP_N, (int)n, 1, d_P_, (int)n, d_ipiv_, d_b_, (int)n,
                             d_info_);
            cudaMemcpy(b + (std::size_t)c * n, d_b_, n * sizeof(double), cudaMemcpyDeviceToHost);
        }
        int info = 0;
        cudaMemcpy(&info, d_info_, sizeof(int), cudaMemcpyDeviceToHost);
        return info;
    }

    // Dense path only for the experimental pass.
    bool supportsCsc() const override { return false; }

  private:
    void release() {
        if (d_P_)
            cudaFree(d_P_);
        if (d_ipiv_)
            cudaFree(d_ipiv_);
        if (d_W_)
            cudaFree(d_W_);
        if (d_info_)
            cudaFree(d_info_);
        if (d_b_)
            cudaFree(d_b_);
        d_P_ = nullptr;
        d_ipiv_ = nullptr;
        d_W_ = nullptr;
        d_info_ = nullptr;
        d_b_ = nullptr;
        lwork_ = 0;
    }

    cusolverDnHandle_t handle_ = nullptr;
    double *d_P_ = nullptr; // LU factors
    int *d_ipiv_ = nullptr; // pivots
    double *d_W_ = nullptr; // workspace
    int *d_info_ = nullptr; // cuSOLVER status
    double *d_b_ = nullptr; // RHS scratch
    unsigned int n_ = 0;
    int lwork_ = 0;
};

} // namespace yabi

#endif // YABI_CUDA_LINEAR_SOLVER_HPP
