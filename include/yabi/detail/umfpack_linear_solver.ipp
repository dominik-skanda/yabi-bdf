/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

#ifndef YABI_DETAIL_UMFPACK_LINEAR_SOLVER_IPP
#define YABI_DETAIL_UMFPACK_LINEAR_SOLVER_IPP

#include <yabi/linear_solver.hpp>
#include <umfpack.h>
#include <vector>
#include <cstdlib>
#include <cstring>

namespace yabi {

template <typename Scalar>
class UmfpackLinearSolver : public LinearSolver<Scalar> {
public:
    UmfpackLinearSolver() : n_(0), symbolic_(NULL), numeric_(NULL), has_pattern_(false),
                            symbolic_calls_(0) {}
    virtual ~UmfpackLinearSolver() { release(); }

    bool initialize(unsigned int n) override {
        release();
        n_ = n;
        return true;
    }

    bool supportsCsc() const override { return true; }
    bool prefersSparseFactorization() const override { return true; }

    // Number of symbolic factorizations performed.  With a native CSC pattern
    // this stays at one while the pattern is unchanged (T31 gate).
    unsigned int symbolicFactorizations() const { return symbolic_calls_; }

    // Dense compatibility path: the caller gave no pattern, so build CSC from
    // the dense buffer once and hand it to the native CSC path.
    bool factorize(Scalar *P, int *ipiv, long *scalingD1, long *scalingD2,
                   unsigned int n) override {
        (void)ipiv; (void)scalingD1; (void)scalingD2;
        if (n == 0) return true;
        CscMatrix<Scalar> csc;
        buildCscFromDense(P, n, csc);
        return factorizeCsc(csc, ipiv, scalingD1, scalingD2, n);
    }

    bool factorizeCsc(const CscMatrix<Scalar> &P, int *ipiv, long *scalingD1,
                      long *scalingD2, unsigned int n) override {
        (void)ipiv; (void)scalingD1; (void)scalingD2;
        if (n == 0) return true;
        if (P.col_ptr.size() != (std::size_t)n + 1 || P.row_idx.size() != P.val.size())
            return false;

        double Control[UMFPACK_CONTROL];
        double Info[UMFPACK_INFO];
        umfpack_di_defaults(Control);

        const bool same_pattern = has_pattern_ && n_ == n && CSC_Ap_ == P.col_ptr &&
                                  CSC_Ai_ == P.row_idx;

        if (numeric_) {
            umfpack_di_free_numeric(&numeric_);
            numeric_ = NULL;
        }

        if (!same_pattern) {
            if (symbolic_) {
                umfpack_di_free_symbolic(&symbolic_);
                symbolic_ = NULL;
            }
            CSC_Ap_ = P.col_ptr;
            CSC_Ai_ = P.row_idx;
            CSC_Ax_.assign(P.val.size(), 0.0);
            if (CSC_Ap_.empty())
                return false;
            int status = umfpack_di_symbolic((int)n, (int)n, CSC_Ap_.data(), CSC_Ai_.data(),
                                             CSC_Ax_.data(), &symbolic_, Control, Info);
            if (status != UMFPACK_OK)
                return false;
            ++symbolic_calls_;
            has_pattern_ = true;
        }

        CSC_Ax_.resize(P.val.size());
        for (std::size_t k = 0; k < P.val.size(); ++k)
            CSC_Ax_[k] = (double)P.val[k];

        int status = umfpack_di_numeric(CSC_Ap_.data(), CSC_Ai_.data(), CSC_Ax_.data(), symbolic_,
                                        &numeric_, Control, Info);
        return status == UMFPACK_OK;
    }

    int solve(Scalar *P, int *ipiv, long *scalingD1, long *scalingD2,
              Scalar *b, unsigned int n, unsigned int nrhs) override {
        (void)P; (void)ipiv; (void)scalingD1; (void)scalingD2;
        if (!numeric_) return -1;
        return solveCsc(CscMatrix<Scalar>(), ipiv, scalingD1, scalingD2, b, n, nrhs);
    }

    int solveCsc(const CscMatrix<Scalar> &P, int *ipiv, long *scalingD1, long *scalingD2,
                 Scalar *b, unsigned int n, unsigned int nrhs) override {
        (void)P; (void)ipiv; (void)scalingD1; (void)scalingD2;
        if (!numeric_ || n != n_) return -1;
        if (CSC_Ap_.empty()) return -1;

        double Control[UMFPACK_CONTROL];
        double Info[UMFPACK_INFO];
        umfpack_di_defaults(Control);

        std::vector<double> x(n);
        std::vector<double> bb(n);
        for (unsigned int rhs = 0; rhs < nrhs; ++rhs) {
            for (unsigned int i = 0; i < n; ++i)
                bb[i] = (double)b[(std::size_t)rhs * n + i];
            int status = umfpack_di_solve(UMFPACK_A, CSC_Ap_.data(), CSC_Ai_.data(),
                                          CSC_Ax_.data(), x.data(), bb.data(), numeric_, Control,
                                          Info);
            if (status != UMFPACK_OK)
                return status;
            for (unsigned int i = 0; i < n; ++i)
                b[(std::size_t)rhs * n + i] = (Scalar)x[i];
        }
        return 0;
    }

private:
    void release() {
        if (numeric_) { umfpack_di_free_numeric(&numeric_); numeric_ = NULL; }
        if (symbolic_) { umfpack_di_free_symbolic(&symbolic_); symbolic_ = NULL; }
        has_pattern_ = false;
        CSC_Ap_.clear();
        CSC_Ai_.clear();
        CSC_Ax_.clear();
    }

    static void buildCscFromDense(const Scalar *P, unsigned int n, CscMatrix<Scalar> &csc) {
        csc.n = n;
        csc.col_ptr.assign(n + 1, 0);
        csc.row_idx.clear();
        csc.val.clear();
        for (unsigned int j = 0; j < n; ++j) {
            csc.col_ptr[j] = (int)csc.val.size();
            for (unsigned int i = 0; i < n; ++i) {
                Scalar v = P[(std::size_t)j * n + i];
                if (v != (Scalar)0.0) {
                    csc.row_idx.push_back((int)i);
                    csc.val.push_back(v);
                }
            }
        }
        csc.col_ptr[n] = (int)csc.val.size();
    }

    unsigned int n_;
    void *symbolic_;
    void *numeric_;
    bool has_pattern_;
    unsigned int symbolic_calls_;
    std::vector<int> CSC_Ap_, CSC_Ai_;
    std::vector<double> CSC_Ax_;
};

} // namespace yabi

#endif
