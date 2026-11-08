/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// Unit tests for the thin BLAS/LAPACK wrappers in linear_algebra.hpp.  Every
// solve the integrator performs funnels through these, so they are verified in
// isolation against hand-computed results.

#include "test_support.hpp"

#include <yabi/linear_algebra.hpp>

#include <vector>

int main() {
    using yabi::linalg::AXPY;
    using yabi::linalg::DGEMM_NT_NT;
    using yabi::linalg::DGEMM_NT_T;
    using yabi::linalg::GEMV_NT;
    using yabi::linalg::GER;
    using yabi::linalg::GETRF;
    using yabi::linalg::GETRS_NT;
    using yabi::linalg::GETRS_T;
    using yabi::linalg::NRM2;
    using yabi::linalg::SCAL;

    // --- AXPY: y = y + alpha*x ---------------------------------------------
    {
        std::vector<double> x = {1.0, 2.0, 3.0}, y = {10.0, 20.0, 30.0};
        double alpha = 2.0;
        AXPY(3, alpha, x.data(), 1, y.data(), 1);
        yabi_test::check(yabi_test::near(y[0], 12.0) && yabi_test::near(y[1], 24.0) &&
                             yabi_test::near(y[2], 36.0),
                         "AXPY y += alpha*x");
    }

    // --- SCAL: x = alpha*x --------------------------------------------------
    {
        std::vector<double> x = {1.0, -2.0, 0.5};
        double alpha = 3.0;
        SCAL(3, alpha, x.data(), 1);
        yabi_test::check(yabi_test::near(x[0], 3.0) && yabi_test::near(x[1], -6.0) &&
                             yabi_test::near(x[2], 1.5),
                         "SCAL x *= alpha");
    }

    // --- NRM2: 2-norm -------------------------------------------------------
    {
        std::vector<double> x = {3.0, 4.0};
        yabi_test::check(yabi_test::near(NRM2(2, x.data(), 1), 5.0), "NRM2 2-norm of (3,4)");
    }

    // --- GEMV_NT: y = alpha*A*x + beta*y ------------------------------------
    {
        // A = [[1,2],[3,4]] column-major, x=(1,1), y=(0,0), alpha=1, beta=0
        std::vector<double> A = {1.0, 3.0, 2.0, 4.0}, x = {1.0, 1.0}, y = {0.0, 0.0};
        double alpha = 1.0, beta = 0.0;
        GEMV_NT(2, 2, alpha, A.data(), 2, x.data(), 1, beta, y.data(), 1);
        yabi_test::check(yabi_test::near(y[0], 3.0) && yabi_test::near(y[1], 7.0),
                         "GEMV_NT y = A*x");
    }

    // --- GER: A = A + alpha*x*y^T -------------------------------------------
    {
        std::vector<double> A = {0.0, 0.0, 0.0, 0.0}, x = {1.0, 2.0}, y = {3.0, 4.0};
        double alpha = 1.0;
        GER(2, 2, alpha, x.data(), 1, y.data(), 1, A.data(), 2);
        // A (column-major) += x y^T = [[3,4],[6,8]]
        yabi_test::check(yabi_test::near(A[0], 3.0) && yabi_test::near(A[1], 6.0) &&
                             yabi_test::near(A[2], 4.0) && yabi_test::near(A[3], 8.0),
                         "GER A += x*y^T");
    }

    // --- DGEMM_NT_NT: C = A*B -----------------------------------------------
    {
        std::vector<double> A = {1.0, 2.0, 3.0, 4.0}, // [[1,3],[2,4]] col-major
            B = {5.0, 6.0, 7.0, 8.0}, C = {0.0, 0.0, 0.0, 0.0}; // [[5,7],[6,8]]
        double alpha = 1.0, beta = 0.0;
        DGEMM_NT_NT(2, 2, 2, alpha, A.data(), 2, B.data(), 2, beta, C.data(), 2);
        // A*B = [[23,31],[34,46]] col-major = {23,34,31,46}
        yabi_test::check(yabi_test::near(C[0], 23.0) && yabi_test::near(C[1], 34.0) &&
                             yabi_test::near(C[2], 31.0) && yabi_test::near(C[3], 46.0),
                         "DGEMM_NT_NT C = A*B");
    }

    // --- DGEMM_NT_T: C = A*B^T ----------------------------------------------
    {
        std::vector<double> A = {1.0, 2.0, 3.0, 4.0}, B = {5.0, 6.0, 7.0, 8.0},
                        C = {0.0, 0.0, 0.0, 0.0};
        double alpha = 1.0, beta = 0.0;
        DGEMM_NT_T(2, 2, 2, alpha, A.data(), 2, B.data(), 2, beta, C.data(), 2);
        // B^T = [[5,6],[7,8]]; A*B^T = [[26,30],[38,44]] col-major = {26,38,30,44}
        yabi_test::check(yabi_test::near(C[0], 26.0) && yabi_test::near(C[1], 38.0) &&
                             yabi_test::near(C[2], 30.0) && yabi_test::near(C[3], 44.0),
                         "DGEMM_NT_T C = A*B^T");
    }

    // --- GETRF + GETRS_NT: LU solve -----------------------------------------
    {
        // A = [[4,2],[3,1]] col-major; solve A x = b, b = (10,8) -> x = (3,-1)
        std::vector<double> A = {4.0, 3.0, 2.0, 1.0}, b = {10.0, 8.0};
        std::vector<int> ipiv(2);
        yabi_test::check(GETRF(2, 2, A.data(), 2, ipiv.data()) == 0, "GETRF factors A");
        yabi_test::check(GETRS_NT(2, 1, A.data(), 2, ipiv.data(), b.data(), 2) == 0,
                         "GETRS_NT solves");
        yabi_test::check(yabi_test::near(b[0], 3.0) && yabi_test::near(b[1], -1.0),
                         "GETRS_NT solution (3,-1)");
    }

    // --- GETRS_T: transposed solve (A^T x = b) ------------------------------
    {
        // A (col-major) = [[4,3],[2,1]]; A^T = [[4,2],[3,1]]. Solve A^T x = b, b=(10,8)
        // 4x0 + 2x1 = 10, 3x0 + x1 = 8 -> x0 = 3, x1 = -1.
        std::vector<double> A = {4.0, 2.0, 3.0, 1.0}, b = {10.0, 8.0};
        std::vector<int> ipiv(2);
        yabi_test::check(GETRF(2, 2, A.data(), 2, ipiv.data()) == 0, "GETRF factors A");
        yabi_test::check(GETRS_T(2, 1, A.data(), 2, ipiv.data(), b.data(), 2) == 0,
                         "GETRS_T solves");
        yabi_test::check(yabi_test::near(b[0], 3.0) && yabi_test::near(b[1], -1.0),
                         "GETRS_T solution (3,-1)");
    }

    // --- GETRF on a singular matrix reports info != 0 -----------------------
    {
        std::vector<double> singular = {1.0, 1.0, 1.0, 1.0}; // rank 1
        std::vector<int> ipiv(2);
        yabi_test::check(GETRF(2, 2, singular.data(), 2, ipiv.data()) != 0,
                         "GETRF reports singular matrix");
    }

    return yabi_test::finish("linear_algebra unit tests");
}
