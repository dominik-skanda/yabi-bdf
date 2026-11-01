/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  Copyright (C) 2010-2012 Dominik Skanda
 *
 *  SPDX-License-Identifier: Apache-2.0
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
#ifndef YABI_LINEAR_ALGEBRA_HPP
#define YABI_LINEAR_ALGEBRA_HPP

// Thin, type-overloaded bindings to the Fortran BLAS/LAPACK symbols used by
// the integrator.

namespace yabi {
namespace linalg {

extern "C" double dnrm2_(const int *N, double *X, const int *incX);
extern "C" float snrm2_(const int *N, float *X, const int *incX);

extern "C" void dger_(const int *M, const int *N, double *alpha, double *X, const int *incX,
                      double *Y, const int *incY, double *A, const int *lda);
extern "C" void sger_(const int *M, const int *N, float *alpha, float *X, const int *incX, float *Y,
                      const int *incY, float *A, const int *lda);

extern "C" void dgemv_(const char *Trans, const int *M, const int *N, double *alpha, double *A,
                       const int *lda, double *X, const int *incX, double *beta, double *Y,
                       const int *incY);
extern "C" void sgemv_(const char *Trans, const int *M, const int *N, float *alpha, float *A,
                       const int *lda, float *X, const int *incX, float *beta, float *Y,
                       const int *incY);

extern "C" void daxpy_(const int *N, double *alpha, double *X, const int *incX, double *Y,
                       const int *incY);
extern "C" void saxpy_(const int *N, float *alpha, float *X, const int *incX, float *Y,
                       const int *incY);

extern "C" void dscal_(const int *N, double *alpha, double *X, const int *incX);
extern "C" void sscal_(const int *N, float *alpha, float *X, const int *incX);

extern "C" void dgemm_(const char *TransA, const char *TransB, const int *M, const int *N,
                       const int *K, double *alpha, double *A, const int *lda, double *B,
                       const int *ldb, double *beta, double *C, const int *ldc);
extern "C" void sgemm_(const char *TransA, const char *TransB, const int *M, const int *N,
                       const int *K, float *alpha, float *A, const int *lda, float *B,
                       const int *ldb, float *beta, float *C, const int *ldc);

extern "C" void dgetrs_(const char *Trans, const int *N, const int *NRHS, double *A, const int *lda,
                        int *ipiv, double *B, const int *ldb, int *info);
extern "C" void sgetrs_(const char *Trans, const int *N, const int *NRHS, float *A, const int *lda,
                        int *ipiv, float *B, const int *ldb, int *info);

extern "C" void dgetrf_(const int *M, const int *N, double *A, const int *lda, int *ipiv,
                        int *info);
extern "C" void sgetrf_(const int *M, const int *N, float *A, const int *lda, int *ipiv, int *info);

static const char charTrans = 'T';
static const char charNoTrans = 'N';

inline double NRM2(int const &N, double *X, int const &incX) {
    return dnrm2_(&N, &X[0], &incX);
}
inline float NRM2(int const &N, float *X, int const &incX) {
    return snrm2_(&N, &X[0], &incX);
}

inline void GER(int const &M, int const &N, double &alpha, double *X, int const &incX, double *Y,
                int const incY, double *A, int const &lda) {
    dger_(&M, &N, &alpha, &X[0], &incX, &Y[0], &incY, &A[0], &lda);
}
inline void GER(int const &M, int const &N, float &alpha, float *X, int const &incX, float *Y,
                int const incY, float *A, int const &lda) {
    sger_(&M, &N, &alpha, &X[0], &incX, &Y[0], &incY, &A[0], &lda);
}

inline void GEMV_NT(int const &M, int const &N, double &alpha, double *A, int const &lda, double *X,
                    int const &incX, double &beta, double *Y, int const &incY) {
    dgemv_(&charNoTrans, &M, &N, &alpha, &A[0], &lda, &X[0], &incX, &beta, &Y[0], &incY);
}
inline void GEMV_NT(int const &M, int const &N, float &alpha, float *A, int const &lda, float *X,
                    int const &incX, float &beta, float *Y, int const &incY) {
    sgemv_(&charNoTrans, &M, &N, &alpha, &A[0], &lda, &X[0], &incX, &beta, &Y[0], &incY);
}

inline void AXPY(int const &N, double &alpha, double *X, int const &incX, double *Y,
                 int const &incY) {
    daxpy_(&N, &alpha, &X[0], &incX, &Y[0], &incY);
}
inline void AXPY(int const &N, float &alpha, float *X, int const &incX, float *Y, int const &incY) {
    saxpy_(&N, &alpha, &X[0], &incX, &Y[0], &incY);
}

inline void SCAL(int const &N, double &alpha, double *X, int const &incX) {
    dscal_(&N, &alpha, &X[0], &incX);
}
inline void SCAL(int const &N, float &alpha, float *X, int const &incX) {
    sscal_(&N, &alpha, &X[0], &incX);
}

inline void DGEMM_NT_NT(int const &M, int const &N, int const &K, double &alpha, double *A,
                        int const &lda, double *B, int const &ldb, double &beta, double *C,
                        int const &ldc) {
    dgemm_(&charNoTrans, &charNoTrans, &M, &N, &K, &alpha, &A[0], &lda, &B[0], &ldb, &beta, &C[0],
           &ldc);
}
inline void DGEMM_NT_NT(int const &M, int const &N, int const &K, float &alpha, float *A,
                        int const &lda, float *B, int const &ldb, float &beta, float *C,
                        int const &ldc) {
    sgemm_(&charNoTrans, &charNoTrans, &M, &N, &K, &alpha, &A[0], &lda, &B[0], &ldb, &beta, &C[0],
           &ldc);
}

inline void DGEMM_NT_T(int const &M, int const &N, int const &K, double &alpha, double *A,
                       int const &lda, double *B, int const &ldb, double &beta, double *C,
                       int const &ldc) {
    dgemm_(&charNoTrans, &charTrans, &M, &N, &K, &alpha, &A[0], &lda, &B[0], &ldb, &beta, &C[0],
           &ldc);
}
inline void DGEMM_NT_T(int const &M, int const &N, int const &K, float &alpha, float *A,
                       int const &lda, float *B, int const &ldb, float &beta, float *C,
                       int const &ldc) {
    sgemm_(&charNoTrans, &charTrans, &M, &N, &K, &alpha, &A[0], &lda, &B[0], &ldb, &beta, &C[0],
           &ldc);
}

inline int GETRS_NT(int const &N, int const &NRHS, double *A, int const &lda, int *ipiv, double *B,
                    int const &ldb) {
    int info;
    dgetrs_(&charNoTrans, &N, &NRHS, &A[0], &lda, &ipiv[0], &B[0], &ldb, &info);
    return info;
}
inline int GETRS_NT(int const &N, int const &NRHS, float *A, int const &lda, int *ipiv, float *B,
                    int const &ldb) {
    int info;
    sgetrs_(&charNoTrans, &N, &NRHS, &A[0], &lda, &ipiv[0], &B[0], &ldb, &info);
    return info;
}

inline int GETRS_T(int const &N, int const &NRHS, double *A, int const &lda, int *ipiv, double *B,
                   int const &ldb) {
    int info;
    dgetrs_(&charTrans, &N, &NRHS, &A[0], &lda, &ipiv[0], &B[0], &ldb, &info);
    return info;
}
inline int GETRS_T(int const &N, int const &NRHS, float *A, int const &lda, int *ipiv, float *B,
                   int const &ldb) {
    int info;
    sgetrs_(&charTrans, &N, &NRHS, &A[0], &lda, &ipiv[0], &B[0], &ldb, &info);
    return info;
}

inline int GETRF(int const &M, int const &N, double *A, int const &lda, int *ipiv) {
    int info;
    dgetrf_(&M, &N, &A[0], &lda, &ipiv[0], &info);
    return info;
}
inline int GETRF(int const &M, int const &N, float *A, int const &lda, int *ipiv) {
    int info;
    sgetrf_(&M, &N, &A[0], &lda, &ipiv[0], &info);
    return info;
}

} // namespace linalg
} // namespace yabi

#endif
