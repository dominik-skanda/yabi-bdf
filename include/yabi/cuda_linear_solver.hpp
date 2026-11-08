#pragma once

/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// EXPERIMENTAL (T36/T36b) — dense LU solver offloading the Newton linear solve
// to a GPU via cuSOLVER.  Only `double` is supported (matching the single
// BDFIntegrator instantiation).  Requires CUDA (built against 12.6) and an
// NVIDIA driver; compiled with the regular host compiler (no nvcc needed — the
// cuSOLVER host API is plain C), linked against `-lcusolver -lcudart`.
//
// Each instance owns its own CUDA stream + cuSOLVER handle (cusolverDnSetStream)
// and its own device + pinned-host buffers, so multiple instances are
// independent and thread-safe — see T36c (parallel multi-instance).  Copies are
// stream-ordered (cudaMemcpyAsync) through pinned staging buffers, so transfer
// can overlap with compute across streams instead of serializing on the legacy
// default stream.
//
// Scope (honest): this offloads the *factorize + solve* of the Newton matrix P.
// The Jacobian J is still assembled on the host (CppAD / analytic); P is copied
// to the device, factored with cusolverDnDgetrf, and each solve is
// cusolverDnDgetrs on the device factors.  Dense path only (supportsCsc() ==
// false); the sparse cuSPARSE path is a separate follow-up (T36b).
//
// Portability: the cuBLAS/cuSOLVER C API is stable across CUDA 12.6 -> 13; this
// solver is source-identical on both.  Only the build (arch flag / driver)
// changes.  The default build is completely CUDA-free (this header is never
// included unless the user opts in).

#ifndef YABI_CUDA_LINEAR_SOLVER_HPP
#define YABI_CUDA_LINEAR_SOLVER_HPP

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cstddef>
#include <cstring>
#include <type_traits>

#include <yabi/linear_solver.hpp>

namespace yabi {

template <typename Scalar> class CudaLinearSolver : public LinearSolver<Scalar> {
    static_assert(std::is_same<Scalar, double>::value,
                  "CudaLinearSolver supports double only (single library instantiation)");

  public:
    CudaLinearSolver() {
        cudaStreamCreate(&stream_);
        cusolverDnCreate(&handle_);
        cusolverDnSetStream(handle_, stream_);
    }

    ~CudaLinearSolver() override {
        release();
        if (handle_)
            cusolverDnDestroy(handle_);
        if (stream_)
            cudaStreamDestroy(stream_);
    }

    bool initialize(unsigned int n) override {
        if (n == 0)
            return false;
        if (n == n_)
            return true;
        release();
        n_ = n;
        const std::size_t nn = (std::size_t)n * n;
        if (cudaMalloc(&d_P_, nn * sizeof(double)) != cudaSuccess ||
            cudaMalloc(&d_ipiv_, n * sizeof(int)) != cudaSuccess ||
            cudaMalloc(&d_info_, sizeof(int)) != cudaSuccess ||
            cudaMalloc(&d_b_, n * sizeof(double)) != cudaSuccess ||
            cudaMallocHost(&h_P_, nn * sizeof(double)) != cudaSuccess ||
            cudaMallocHost(&h_b_, n * sizeof(double)) != cudaSuccess ||
            cudaMallocHost(&h_x_, n * sizeof(double)) != cudaSuccess)
            return false;
        return true;
    }

    // Factor P = L U on the device (stream-ordered).  The host P is left
    // unmodified; solve() uses the resident device factors + pivots.
    bool factorize(Scalar *P, int *ipiv, long * /*scalingD1*/, long * /*scalingD2*/,
                   unsigned int n) override {
        if (!handle_ || n != n_)
            return false;
        if (lwork_ == 0) {
            cusolverDnDgetrf_bufferSize(handle_, (int)n, (int)n, d_P_, (int)n, &lwork_);
            if (cudaMalloc(&d_W_, (std::size_t)lwork_ * sizeof(double)) != cudaSuccess)
                return false;
        }
        std::memcpy(h_P_, P, (std::size_t)n * n * sizeof(double)); // pageable -> pinned
        cudaMemcpyAsync(d_P_, h_P_, (std::size_t)n * n * sizeof(double), cudaMemcpyHostToDevice,
                        stream_);
        cusolverDnDgetrf(handle_, (int)n, (int)n, d_P_, (int)n, d_W_, d_ipiv_, d_info_);
        cudaMemcpyAsync(ipiv, d_ipiv_, n * sizeof(int), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
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
            std::memcpy(h_b_, b + (std::size_t)c * n, n * sizeof(double)); // pageable -> pinned
            cudaMemcpyAsync(d_b_, h_b_, n * sizeof(double), cudaMemcpyHostToDevice, stream_);
            cusolverDnDgetrs(handle_, CUBLAS_OP_N, (int)n, 1, d_P_, (int)n, d_ipiv_, d_b_, (int)n,
                             d_info_);
            cudaMemcpyAsync(h_x_, d_b_, n * sizeof(double), cudaMemcpyDeviceToHost, stream_);
            cudaStreamSynchronize(stream_);
            std::memcpy(b + (std::size_t)c * n, h_x_, n * sizeof(double)); // pinned -> pageable
        }
        int info = 0;
        cudaMemcpy(&info, d_info_, sizeof(int), cudaMemcpyDeviceToHost);
        return info;
    }

    // Dense path only for this pass (sparse cuSPARSE is a follow-up).
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
        if (h_P_)
            cudaFreeHost(h_P_);
        if (h_b_)
            cudaFreeHost(h_b_);
        if (h_x_)
            cudaFreeHost(h_x_);
        d_P_ = nullptr;
        d_ipiv_ = nullptr;
        d_W_ = nullptr;
        d_info_ = nullptr;
        d_b_ = nullptr;
        h_P_ = nullptr;
        h_b_ = nullptr;
        h_x_ = nullptr;
        lwork_ = 0;
    }

    cudaStream_t stream_ = nullptr;
    cusolverDnHandle_t handle_ = nullptr;
    double *d_P_ = nullptr; // LU factors (device)
    int *d_ipiv_ = nullptr; // pivots (device)
    double *d_W_ = nullptr; // workspace (device)
    int *d_info_ = nullptr; // cuSOLVER status (device)
    double *d_b_ = nullptr; // RHS scratch (device)
    double *h_P_ = nullptr; // pinned staging for P
    double *h_b_ = nullptr; // pinned staging for b
    double *h_x_ = nullptr; // pinned staging for the result
    unsigned int n_ = 0;
    int lwork_ = 0;
};

} // namespace yabi

#endif // YABI_CUDA_LINEAR_SOLVER_HPP
