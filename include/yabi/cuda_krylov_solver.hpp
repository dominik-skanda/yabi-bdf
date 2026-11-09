#pragma once

/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// T36e — GPU sparse *iterative* Krylov solver (own implementation).  Closes the
// gap cuDSS (sparse *direct*, T36d) leaves: a BiCGStab Krylov solve whose vector
// ops run on cuBLAS and whose P·v runs on cuSPARSE SpMV, so large sparse Newton
// systems (n ≫ 100) can be solved iteratively on the GPU.
//
// Open architecture: this is a thin `LinearSolver` adapter (our own Krylov loop,
// no vendored library — cuBLAS/cuSPARSE ship with CUDA) reached only through
// `setLinearSolver(...)`; the integrator core stays CUDA-free in the default
// build.  Compiled with the regular host compiler (no nvcc — the cuBLAS/cuSPARSE
// host API is plain C), linked against `-lcusparse -lcublas -lcudart`.
//
// Design note (honest): unlike the CPU matrix-free path (T32), which applies P·v
// through a `MatrixFreeOperator` (J·v), this solver applies P·v through the
// *assembled* sparse P (formed by the integrator's sparse assembly) as a device
// CSR + cuSPARSE SpMV — there is no GPU J·v callback yet.  So it advertises
// supportsMatrixFree() to take the Krylov path (which forms P for the
// preconditioner) and then uses that P for the SpMV.  Only `double` is supported.

#ifndef YABI_CUDA_KRYLOV_SOLVER_HPP
#define YABI_CUDA_KRYLOV_SOLVER_HPP

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

#include <yabi/detail/dense_linear_solver.ipp>
#include <yabi/linear_solver.hpp>

namespace yabi {

template <typename Scalar> class CudaKrylovSolver : public LinearSolver<Scalar> {
    static_assert(std::is_same<Scalar, double>::value,
                  "CudaKrylovSolver supports double only (single library instantiation)");

  public:
    CudaKrylovSolver() {
        cudaStreamCreate(&stream_);
        cublasCreate(&cublas_);
        cublasSetStream(cublas_, stream_);
        cusparseCreate(&cusparse_);
        cusparseSetStream(cusparse_, stream_);
    }

    ~CudaKrylovSolver() override {
        release();
        if (cublas_)
            cublasDestroy(cublas_);
        if (cusparse_)
            cusparseDestroy(cusparse_);
        if (stream_)
            cudaStreamDestroy(stream_);
    }

    bool initialize(unsigned int n) override {
        if (n == 0)
            return false;
        if (n == n_ && !d_r_)
            return true;
        if (n == n_)
            return true;
        release();
        n_ = n;
        const std::size_t nn = n;
        bool ok = true;
        ok = ok && cudaMalloc(&d_r_, nn * sizeof(double)) == cudaSuccess;
        ok = ok && cudaMalloc(&d_rhat_, nn * sizeof(double)) == cudaSuccess;
        ok = ok && cudaMalloc(&d_v_, nn * sizeof(double)) == cudaSuccess;
        ok = ok && cudaMalloc(&d_p_, nn * sizeof(double)) == cudaSuccess;
        ok = ok && cudaMalloc(&d_phat_, nn * sizeof(double)) == cudaSuccess;
        ok = ok && cudaMalloc(&d_s_, nn * sizeof(double)) == cudaSuccess;
        ok = ok && cudaMalloc(&d_shat_, nn * sizeof(double)) == cudaSuccess;
        ok = ok && cudaMalloc(&d_t_, nn * sizeof(double)) == cudaSuccess;
        ok = ok && cudaMalloc(&d_x_, nn * sizeof(double)) == cudaSuccess;
        ok = ok && cudaMalloc(&d_invdiag_, nn * sizeof(double)) == cudaSuccess;
        ok = ok && cudaMallocHost(&h_s1_, sizeof(double)) == cudaSuccess;
        ok = ok && cudaMallocHost(&h_s2_, sizeof(double)) == cudaSuccess;
        if (!ok) {
            release();
            return false;
        }
        if (fallback_)
            fallback_->initialize(n);
        return true;
    }

    // Dense fallback (the sensitivity solve still uses the explicit dense J).
    bool factorize(Scalar *P, int *ipiv, long *sd1, long *sd2, unsigned int n) override {
        ensureFallback();
        return fallback_->factorize(P, ipiv, sd1, sd2, n);
    }
    int solve(Scalar *P, int *ipiv, long *sd1, long *sd2, Scalar *b, unsigned int n,
              unsigned int nrhs) override {
        ensureFallback();
        return fallback_->solve(P, ipiv, sd1, sd2, b, n, nrhs);
    }

    // Reached through the Krylov (matrix-free) path, which forms P for the
    // preconditioner; we use that assembled P for the SpMV.
    bool supportsCsc() const override { return false; }
    bool supportsMatrixFree() const override { return true; }

    // The J·v operator is not used — P·v is the assembled-P SpMV above.
    void setMatrixFreeOperator(const MatrixFreeOperator<Scalar> *) override {}

    bool factorizeCsc(const CscMatrix<Scalar> &P, int *, long *, long *, unsigned int n) override {
        if (!cusparse_ || n != n_ || P.empty() || n != P.n)
            return false;
        // CSC -> CSR on the host (column indices sorted per row by construction
        // of the CSR from a row-sorted CSC).
        cscToCsr(P);
        const std::size_t nnz = csr_val_.size();
        if (cudaMalloc(&d_row_ptr_, (n + 1) * sizeof(int)) != cudaSuccess ||
            cudaMalloc(&d_col_, nnz * sizeof(int)) != cudaSuccess ||
            cudaMalloc(&d_val_, nnz * sizeof(double)) != cudaSuccess) {
            releaseMatrix();
            return false;
        }
        cudaMemcpy(d_row_ptr_, csr_ptr_.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_, csr_col_.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_val_, csr_val_.data(), nnz * sizeof(double), cudaMemcpyHostToDevice);

        // Jacobi preconditioner: inverse diagonal (host -> device).
        P.extractDiagonal(diag_);
        std::vector<double> invdiag(n, 1.0);
        for (unsigned int i = 0; i < n; ++i)
            if (diag_[i] != 0.0)
                invdiag[i] = 1.0 / diag_[i];
        cudaMemcpy(d_invdiag_, invdiag.data(), n * sizeof(double), cudaMemcpyHostToDevice);

        // Build the cuSPARSE CSR SpMV descriptor + workspace.
        cusparseCreateCsr(&mat_, (int64_t)n, (int64_t)n, (int64_t)nnz, d_row_ptr_, d_col_, d_val_,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                          CUDA_R_64F);
        const double one = 1.0, zero = 0.0;
        size_t buf = 0;
        cusparseDnVecDescr_t vx = nullptr, vy = nullptr;
        cusparseCreateDnVec(&vx, (int64_t)n, d_r_, CUDA_R_64F);
        cusparseCreateDnVec(&vy, (int64_t)n, d_v_, CUDA_R_64F);
        cusparseSpMV_bufferSize(cusparse_, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat_, vx, &zero,
                                vy, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buf);
        cusparseDestroyDnVec(vx);
        cusparseDestroyDnVec(vy);
        if (cudaMalloc(&d_buffer_, buf) != cudaSuccess) {
            releaseMatrix();
            return false;
        }
        return true;
    }

    int solveCsc(const CscMatrix<Scalar> &, int *, long *, long *, Scalar *b, unsigned int n,
                 unsigned int nrhs) override {
        if (!mat_ || n != n_ || nrhs == 0)
            return -1;
        for (unsigned int c = 0; c < nrhs; ++c) {
            std::vector<double> rhs(b + (std::size_t)c * n, b + (std::size_t)(c + 1) * n);
            std::vector<double> x(n, 0.0);
            if (!bicgstab(rhs, x))
                return -1;
            std::copy(x.begin(), x.end(), b + (std::size_t)c * n);
        }
        return 0;
    }

    void setTolerance(double relative, double absolute, unsigned int max_iterations,
                      unsigned int /*restart unused for BiCGStab*/) {
        rel_tol_ = relative;
        abs_tol_ = absolute;
        max_iter_ = max_iterations == 0 ? 1 : max_iterations;
    }
    unsigned int lastIterations() const { return last_iter_; }
    bool lastConverged() const { return last_ok_; }

    // Inexact-Newton forcing (T32c): when installed, the Krylov tolerance is
    // eta*||b|| (loose near Newton convergence) instead of the fixed relative
    // tolerance.  This is what keeps the linear solve achievable on stiff
    // systems where the Newton residual becomes very small.
    void setForcingTerm(ForcingTerm<double> forcing) override { forcing_ = std::move(forcing); }
    bool hasForcingTerm() const override { return static_cast<bool>(forcing_); }
    void setNewtonLinearizationContext(const NewtonLinearizationContext<double> &ctx) override {
        context_ = ctx;
    }

  private:
    // CSC (column) -> CSR (row) on the host, reusing the T36d handoff logic.
    void cscToCsr(const CscMatrix<Scalar> &P) {
        const std::size_t n = P.n, nnz = P.nnz();
        csr_ptr_.assign(n + 1, 0);
        csr_col_.resize(nnz);
        csr_val_.resize(nnz);
        for (std::size_t k = 0; k < nnz; ++k)
            csr_ptr_[(std::size_t)P.row_idx[k] + 1]++;
        for (std::size_t i = 0; i < n; ++i)
            csr_ptr_[i + 1] += csr_ptr_[i];
        std::vector<std::size_t> cursor(csr_ptr_.begin(), csr_ptr_.begin() + n);
        for (std::size_t j = 0; j < n; ++j)
            for (int k = P.col_ptr[j]; k < P.col_ptr[j + 1]; ++k) {
                const std::size_t r = (std::size_t)P.row_idx[(std::size_t)k];
                const std::size_t pos = cursor[r]++;
                csr_col_[pos] = (int)j;
                csr_val_[pos] = P.val[(std::size_t)k];
            }
    }

    // y = A * x on the device (cuSPARSE SpMV).
    bool spmv(const double *x, double *y) {
        cusparseDnVecDescr_t vx = nullptr, vy = nullptr;
        cusparseCreateDnVec(&vx, (int64_t)n_, const_cast<double *>(x), CUDA_R_64F);
        cusparseCreateDnVec(&vy, (int64_t)n_, y, CUDA_R_64F);
        const double one = 1.0, zero = 0.0;
        const cusparseStatus_t st =
            cusparseSpMV(cusparse_, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat_, vx, &zero, vy,
                         CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer_);
        cusparseDestroyDnVec(vx);
        cusparseDestroyDnVec(vy);
        return st == CUSPARSE_STATUS_SUCCESS;
    }

    // y = M^-1 x (Jacobi), via cublasDdgmm (diag(invdiag) * x).
    bool jacobi(const double *x, double *y) {
        cudaMemcpyAsync(y, x, n_ * sizeof(double), cudaMemcpyDeviceToDevice, stream_);
        return cublasDdgmm(cublas_, CUBLAS_SIDE_LEFT, (int)n_, 1, y, (int)n_, d_invdiag_, 1, y,
                           (int)n_) == CUBLAS_STATUS_SUCCESS;
    }

    double dot(const double *x, const double *y) {
        cublasDdot(cublas_, (int)n_, x, 1, y, 1, h_s1_);
        cudaStreamSynchronize(stream_);
        return *h_s1_;
    }
    double nrm2(const double *x) {
        cublasDnrm2(cublas_, (int)n_, x, 1, h_s2_);
        cudaStreamSynchronize(stream_);
        return *h_s2_;
    }

    // Fixed relative tolerance unless an inexact-Newton forcing callback is
    // installed (then eta in [0,1] is used, matching the CPU Krylov base).
    double selectRelativeTolerance() const {
        double relative = rel_tol_;
        if (forcing_) {
            const double eta = forcing_(context_);
            if (std::isfinite(eta) && eta >= 0.0 && eta <= 1.0)
                relative = eta;
        }
        return relative;
    }

    bool bicgstab(const std::vector<double> &b, std::vector<double> &x) {
        const double bnorm = std::sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0));
        if (!std::isfinite(bnorm))
            return false;
        const double tol = abs_tol_ + selectRelativeTolerance() * bnorm;
        last_ok_ = false;
        last_iter_ = 0;
        if (bnorm <= tol) {
            last_ok_ = true;
            return true;
        }

        cudaMemcpy(d_r_, b.data(), n_ * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rhat_, b.data(), n_ * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(d_x_, 0, n_ * sizeof(double));
        cudaMemset(d_v_, 0, n_ * sizeof(double));
        cudaMemset(d_p_, 0, n_ * sizeof(double));

        double rho = 1.0, alpha = 1.0, omega = 1.0;
        const double tiny = std::numeric_limits<double>::min();

        for (unsigned int k = 0; k < max_iter_; ++k) {
            const double rho_new = dot(d_rhat_, d_r_);
            if (!std::isfinite(rho_new) || std::fabs(rho_new) <= tiny)
                break;
            const double beta = (rho_new / rho) * (alpha / omega);
            // p = r + beta*(p - omega*v)
            recomputeP(beta, omega);

            if (!jacobi(d_p_, d_phat_))
                return false;
            if (!spmv(d_phat_, d_v_))
                return false;

            const double denom = dot(d_rhat_, d_v_);
            if (!std::isfinite(denom) || std::fabs(denom) <= tiny)
                break;
            alpha = rho_new / denom;

            // s = r - alpha*v
            cudaMemcpyAsync(d_s_, d_r_, n_ * sizeof(double), cudaMemcpyDeviceToDevice, stream_);
            const double m_alpha = -alpha;
            cublasDaxpy(cublas_, (int)n_, &m_alpha, d_v_, 1, d_s_, 1);

            if (nrm2(d_s_) <= tol) {
                // x += alpha*phat
                cublasDaxpy(cublas_, (int)n_, &alpha, d_phat_, 1, d_x_, 1);
                copyOut(x);
                last_iter_ = k + 1;
                last_ok_ = true;
                return true;
            }

            if (!jacobi(d_s_, d_shat_))
                return false;
            if (!spmv(d_shat_, d_t_))
                return false;

            const double tt = dot(d_t_, d_t_);
            if (!std::isfinite(tt) || tt <= tiny)
                break;
            omega = dot(d_t_, d_s_) / tt;

            // x += alpha*phat + omega*shat
            cublasDaxpy(cublas_, (int)n_, &alpha, d_phat_, 1, d_x_, 1);
            cublasDaxpy(cublas_, (int)n_, &omega, d_shat_, 1, d_x_, 1);
            // r = s - omega*t
            cudaMemcpyAsync(d_r_, d_s_, n_ * sizeof(double), cudaMemcpyDeviceToDevice, stream_);
            const double m_omega = -omega;
            cublasDaxpy(cublas_, (int)n_, &m_omega, d_t_, 1, d_r_, 1);
            rho = rho_new;

            if (nrm2(d_r_) <= tol) {
                copyOut(x);
                last_iter_ = k + 1;
                last_ok_ = true;
                return true;
            }
        }

        copyOut(x);
        last_iter_ = max_iter_;
        return false;
    }

    // p = r + beta*(p - omega*v)  (device, correct sign).
    void recomputeP(double beta, double omega) {
        const double one = 1.0;
        const double m_omega = -omega;
        cublasDaxpy(cublas_, (int)n_, &m_omega, d_v_, 1, d_p_, 1); // p <- p - omega*v
        cublasDscal(cublas_, (int)n_, &beta, d_p_, 1);             // p <- beta*(p - omega*v)
        cublasDaxpy(cublas_, (int)n_, &one, d_r_, 1, d_p_, 1);     // p <- r + beta*(p - omega*v)
    }

    void copyOut(std::vector<double> &x) {
        cudaMemcpy(x.data(), d_x_, n_ * sizeof(double), cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(stream_);
    }

    void ensureFallback() {
        if (!fallback_)
            fallback_.reset(new DenseLUSolver<Scalar>());
    }

    void releaseMatrix() {
        if (mat_)
            cusparseDestroySpMat(mat_), mat_ = nullptr;
        if (d_row_ptr_)
            cudaFree(d_row_ptr_), d_row_ptr_ = nullptr;
        if (d_col_)
            cudaFree(d_col_), d_col_ = nullptr;
        if (d_val_)
            cudaFree(d_val_), d_val_ = nullptr;
        if (d_buffer_)
            cudaFree(d_buffer_), d_buffer_ = nullptr;
    }

    void release() {
        releaseMatrix();
        if (d_r_)
            cudaFree(d_r_), d_r_ = nullptr;
        if (d_rhat_)
            cudaFree(d_rhat_), d_rhat_ = nullptr;
        if (d_v_)
            cudaFree(d_v_), d_v_ = nullptr;
        if (d_p_)
            cudaFree(d_p_), d_p_ = nullptr;
        if (d_phat_)
            cudaFree(d_phat_), d_phat_ = nullptr;
        if (d_s_)
            cudaFree(d_s_), d_s_ = nullptr;
        if (d_shat_)
            cudaFree(d_shat_), d_shat_ = nullptr;
        if (d_t_)
            cudaFree(d_t_), d_t_ = nullptr;
        if (d_x_)
            cudaFree(d_x_), d_x_ = nullptr;
        if (d_invdiag_)
            cudaFree(d_invdiag_), d_invdiag_ = nullptr;
        if (h_s1_)
            cudaFreeHost(h_s1_), h_s1_ = nullptr;
        if (h_s2_)
            cudaFreeHost(h_s2_), h_s2_ = nullptr;
        d_r_ = d_rhat_ = d_v_ = d_p_ = d_phat_ = d_s_ = d_shat_ = d_t_ = d_x_ = d_invdiag_ = nullptr;
        n_ = 0;
    }

    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_ = nullptr;
    cusparseHandle_t cusparse_ = nullptr;
    cusparseSpMatDescr_t mat_ = nullptr;

    double *d_r_ = nullptr, *d_rhat_ = nullptr, *d_v_ = nullptr, *d_p_ = nullptr,
           *d_phat_ = nullptr, *d_s_ = nullptr, *d_shat_ = nullptr, *d_t_ = nullptr,
           *d_x_ = nullptr, *d_invdiag_ = nullptr, *d_buffer_ = nullptr;
    int *d_row_ptr_ = nullptr, *d_col_ = nullptr;
    double *d_val_ = nullptr;
    double *h_s1_ = nullptr, *h_s2_ = nullptr;

    std::vector<int> csr_ptr_, csr_col_;
    std::vector<double> csr_val_, diag_;
    unsigned int n_ = 0;

    double rel_tol_ = 1e-12, abs_tol_ = 0.0;
    unsigned int max_iter_ = 200;
    unsigned int last_iter_ = 0;
    bool last_ok_ = false;

    ForcingTerm<double> forcing_;
    NewtonLinearizationContext<double> context_;

    std::unique_ptr<LinearSolver<Scalar>> fallback_;
};

} // namespace yabi

#endif // YABI_CUDA_KRYLOV_SOLVER_HPP
