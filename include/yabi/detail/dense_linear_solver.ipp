/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

#ifndef YABI_DETAIL_DENSE_LINEAR_SOLVER_IPP
#define YABI_DETAIL_DENSE_LINEAR_SOLVER_IPP

#include <yabi/linear_algebra.hpp>
#include <yabi/linear_solver.hpp>

namespace yabi {

template <typename Scalar>
class DenseLUSolver : public LinearSolver<Scalar> {
public:
    DenseLUSolver() {}
    virtual ~DenseLUSolver() {}

    bool initialize(unsigned int n) override {
        (void)n;
        return true;
    }

    bool factorize(Scalar *P, int *ipiv, long *scalingD1, long *scalingD2,
                   unsigned int n) override {
        (void)scalingD1;
        (void)scalingD2;
        int info = linalg::GETRF((int)n, (int)n, P, (int)n, ipiv);
        return info == 0;
    }

    int solve(Scalar *P, int *ipiv, long *scalingD1, long *scalingD2,
              Scalar *b, unsigned int n, unsigned int nrhs) override {
        (void)scalingD1;
        (void)scalingD2;
        if (nrhs == 1) {
            return linalg::GETRS_NT((int)n, 1, P, (int)n, ipiv, b, (int)n);
        }
        // Fallback: use nrhs==1 path by repeated calls not needed; call with nrhs
        // For compatibility call GETRS_NT (LAPACK supports nrhs)
        // linalg has nrhs param
        return linalg::GETRS_NT((int)n, (int)nrhs, P, (int)n, ipiv, b, (int)n);
    }
};

} // namespace yabi

#endif
