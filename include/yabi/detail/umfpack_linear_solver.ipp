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
    UmfpackLinearSolver() : n_(0), symbolic_(NULL), numeric_(NULL), owns_(true) {}
    virtual ~UmfpackLinearSolver() {
        if (numeric_) umfpack_di_free_numeric(&numeric_);
        if (symbolic_) umfpack_di_free_symbolic(&symbolic_);
    }

    bool initialize(unsigned int n) override {
        n_ = n;
        if (numeric_) { umfpack_di_free_numeric(&numeric_); numeric_ = NULL; }
        if (symbolic_) { umfpack_di_free_symbolic(&symbolic_); symbolic_ = NULL; }
        return true;
    }

    bool factorize(Scalar *P, int *ipiv, long *scalingD1, long *scalingD2,
                   unsigned int n) override {
        (void)ipiv; (void)scalingD1; (void)scalingD2;
        if (n == 0) return true;
        // UMFPACK expects compressed column form; convert dense P (column-major in yabi? P stored as P[j*numOfSpecies + i] etc but passed as array)
        // yabi uses column-major style storage consistent with BLAS; build CSC from dense
        // For generality convert dense to CSC (all entries considered; for sparse benefit need pattern)
        std::vector<int> Ap(n + 1, 0);
        // count non-zeros (all for now, but interface ready)
        for (unsigned int j = 0; j < n; j++) {
            for (unsigned int i = 0; i < n; i++) {
                Scalar v = P[j * n + i];
                if (v != 0.0) Ap[j + 1]++;
            }
        }
        for (unsigned int j = 0; j < n; j++) Ap[j + 1] += Ap[j];
        std::vector<int> Ai(Ap[n]);
        std::vector<double> Ax(Ap[n]);
        std::vector<int> Ai_cnt(n, 0);
        for (unsigned int j = 0; j < n; j++) {
            for (unsigned int i = 0; i < n; i++) {
                Scalar v = P[j * n + i];
                if (v != 0.0) {
                    int idx = Ap[j] + Ai_cnt[j]++;
                    Ai[idx] = (int)i;
                    Ax[idx] = (double)v;
                }
            }
        }
        if (numeric_) { umfpack_di_free_numeric(&numeric_); numeric_ = NULL; }
        if (symbolic_) { umfpack_di_free_symbolic(&symbolic_); symbolic_ = NULL; }
        double Control[UMFPACK_CONTROL];
        double Info[UMFPACK_INFO];
        umfpack_di_defaults(Control);
        int status = umfpack_di_symbolic((int)n, (int)n, &Ap[0], &Ai[0], &Ax[0], &symbolic_, Control, Info);
        if (status != UMFPACK_OK) return false;
        status = umfpack_di_numeric(&Ap[0], &Ai[0], &Ax[0], symbolic_, &numeric_, Control, Info);
        if (status != UMFPACK_OK) return false;
        // store CSC copy if needed? numeric uses Ax copy
        CSC_Ap_ = Ap; CSC_Ai_ = Ai; CSC_Ax_ = Ax;
        return true;
    }

    int solve(Scalar *P, int *ipiv, long *scalingD1, long *scalingD2,
              Scalar *b, unsigned int n, unsigned int nrhs) override {
        (void)P; (void)ipiv; (void)scalingD1; (void)scalingD2;
        if (!numeric_) return -1;
        if (nrhs != 1) {
            // simple path: solve one by one by copying
        }
        std::vector<double> x(n);
        std::vector<double> bb(n);
        for (unsigned int i = 0; i < n; i++) bb[i] = (double)b[i];
        double Control[UMFPACK_CONTROL];
        double Info[UMFPACK_INFO];
        umfpack_di_defaults(Control);
        int status = umfpack_di_solve(UMFPACK_A, &CSC_Ap_[0], &CSC_Ai_[0], &CSC_Ax_[0], &x[0], &bb[0], numeric_, Control, Info);
        if (status != UMFPACK_OK) return status;
        for (unsigned int i = 0; i < n; i++) b[i] = (Scalar)x[i];
        return 0;
    }

private:
    unsigned int n_;
    void *symbolic_;
    void *numeric_;
    bool owns_;
    std::vector<int> CSC_Ap_, CSC_Ai_;
    std::vector<double> CSC_Ax_;
};

} // namespace yabi

#endif
