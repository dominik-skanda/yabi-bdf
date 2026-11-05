/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

#ifndef YABI_DETAIL_KLU_LINEAR_SOLVER_IPP
#define YABI_DETAIL_KLU_LINEAR_SOLVER_IPP

#include <yabi/linear_solver.hpp>
#include <klu.h>
#include <vector>
#include <cstdlib>

namespace yabi {

template <typename Scalar>
class KluLinearSolver : public LinearSolver<Scalar> {
public:
    KluLinearSolver() : n_(0), symbolic_(NULL), numeric_(NULL), has_pattern_(false),
                        symbolic_calls_(0) {
        klu_defaults(&Common_);
    }
    virtual ~KluLinearSolver() { release(); }

    bool initialize(unsigned int n) override {
        release();
        n_ = n;
        return true;
    }

    bool supportsCsc() const override { return true; }
    bool prefersSparseFactorization() const override { return true; }

    unsigned int symbolicFactorizations() const { return symbolic_calls_; }

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
        if (P.col_ptr.empty())
            return false;

        if (numeric_) {
            klu_free_numeric(&numeric_, &Common_);
            numeric_ = NULL;
        }

        const bool same_pattern = has_pattern_ && n_ == n && CSC_Ap_ == P.col_ptr &&
                                  CSC_Ai_ == P.row_idx;

        if (!same_pattern) {
            if (symbolic_) {
                klu_free_symbolic(&symbolic_, &Common_);
                symbolic_ = NULL;
            }
            CSC_Ap_ = P.col_ptr;
            CSC_Ai_ = P.row_idx;
            symbolic_ = klu_analyze((int)n, CSC_Ap_.data(), CSC_Ai_.data(), &Common_);
            if (!symbolic_)
                return false;
            ++symbolic_calls_;
            has_pattern_ = true;
        }

        CSC_Ax_.assign(P.val.size(), 0.0);
        for (std::size_t k = 0; k < P.val.size(); ++k)
            CSC_Ax_[k] = (double)P.val[k];

        numeric_ = klu_factor(CSC_Ap_.data(), CSC_Ai_.data(), CSC_Ax_.data(), symbolic_, &Common_);
        return numeric_ != NULL;
    }

    int solve(Scalar *P, int *ipiv, long *scalingD1, long *scalingD2,
              Scalar *b, unsigned int n, unsigned int nrhs) override {
        (void)P; (void)ipiv; (void)scalingD1; (void)scalingD2;
        return solveCsc(CscMatrix<Scalar>(), ipiv, scalingD1, scalingD2, b, n, nrhs);
    }

    int solveCsc(const CscMatrix<Scalar> &P, int *ipiv, long *scalingD1, long *scalingD2,
                 Scalar *b, unsigned int n, unsigned int nrhs) override {
        (void)P; (void)ipiv; (void)scalingD1; (void)scalingD2;
        if (!numeric_ || !symbolic_ || n != n_) return -1;
        std::vector<double> bb((std::size_t)n * nrhs);
        for (std::size_t i = 0; i < bb.size(); ++i)
            bb[i] = (double)b[i];
        int status = klu_solve(symbolic_, numeric_, (int)n, (int)nrhs, bb.data(), &Common_);
        if (status != 1) return status;
        for (std::size_t i = 0; i < bb.size(); ++i)
            b[i] = (Scalar)bb[i];
        return 0;
    }

private:
    void release() {
        if (numeric_) { klu_free_numeric(&numeric_, &Common_); numeric_ = NULL; }
        if (symbolic_) { klu_free_symbolic(&symbolic_, &Common_); symbolic_ = NULL; }
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
    klu_symbolic *symbolic_;
    klu_numeric *numeric_;
    klu_common Common_;
    bool has_pattern_;
    unsigned int symbolic_calls_;
    std::vector<int> CSC_Ap_, CSC_Ai_;
    std::vector<double> CSC_Ax_;
};

} // namespace yabi

#endif
