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
    KluLinearSolver() : n_(0), symbolic_(NULL), numeric_(NULL) {
        klu_defaults(&Common_);
    }
    virtual ~KluLinearSolver() {
        if (numeric_) klu_free_numeric(&numeric_, &Common_);
        if (symbolic_) klu_free_symbolic(&symbolic_, &Common_);
    }

    bool initialize(unsigned int n) override {
        n_ = n;
        if (numeric_) { klu_free_numeric(&numeric_, &Common_); numeric_ = NULL; }
        if (symbolic_) { klu_free_symbolic(&symbolic_, &Common_); symbolic_ = NULL; }
        return true;
    }

    bool factorize(Scalar *P, int *ipiv, long *scalingD1, long *scalingD2,
                   unsigned int n) override {
        (void)ipiv; (void)scalingD1; (void)scalingD2;
        if (n == 0) return true;
        std::vector<int> Ap(n + 1, 0);
        for (unsigned int j = 0; j < n; j++) {
            for (unsigned int i = 0; i < n; i++) {
                if (P[j * n + i] != 0.0) Ap[j + 1]++;
            }
        }
        for (unsigned int j = 0; j < n; j++) Ap[j + 1] += Ap[j];
        std::vector<int> Ai(Ap[n]);
        std::vector<double> Ax(Ap[n]);
        std::vector<int> Ai_cnt(n, 0);
        for (unsigned int j = 0; j < n; j++) {
            for (unsigned int i = 0; i < n; i++) {
                if (P[j * n + i] != 0.0) {
                    int idx = Ap[j] + Ai_cnt[j]++;
                    Ai[idx] = (int)i;
                    Ax[idx] = (double)P[j * n + i];
                }
            }
        }
        if (numeric_) { klu_free_numeric(&numeric_, &Common_); numeric_ = NULL; }
        if (symbolic_) { klu_free_symbolic(&symbolic_, &Common_); symbolic_ = NULL; }
        symbolic_ = klu_analyze((int)n, &Ap[0], &Ai[0], &Common_);
        if (!symbolic_) return false;
        numeric_ = klu_factor(&Ap[0], &Ai[0], &Ax[0], symbolic_, &Common_);
        if (!numeric_) return false;
        CSC_Ap_ = Ap; CSC_Ai_ = Ai; CSC_Ax_ = Ax;
        return true;
    }

    int solve(Scalar *P, int *ipiv, long *scalingD1, long *scalingD2,
              Scalar *b, unsigned int n, unsigned int nrhs) override {
        (void)P; (void)ipiv; (void)scalingD1; (void)scalingD2;
        if (!numeric_ || !symbolic_) return -1;
        std::vector<double> bb(n);
        for (unsigned int i = 0; i < n; i++) bb[i] = (double)b[i];
        int status = klu_solve(symbolic_, numeric_, (int)n, (int)nrhs, &bb[0], &Common_);
        if (status != 1) return status;
        for (unsigned int i = 0; i < n; i++) b[i] = (Scalar)bb[i];
        return 0;
    }

private:
    unsigned int n_;
    klu_symbolic *symbolic_;
    klu_numeric *numeric_;
    klu_common Common_;
    std::vector<int> CSC_Ap_, CSC_Ai_;
    std::vector<double> CSC_Ax_;
};

} // namespace yabi

#endif
