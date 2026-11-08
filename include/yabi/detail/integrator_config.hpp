/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  Copyright (C) 2010-2012 Dominik Skanda
 *
 *  SPDX-License-Identifier: Apache-2.0
 */
#ifndef YABI_DETAIL_INTEGRATOR_CONFIG_HPP
#define YABI_DETAIL_INTEGRATOR_CONFIG_HPP

#include <limits>

namespace yabi {
namespace detail {

// Tunable integrator configuration.  Defaults reproduce the historical
// constructor values exactly; the public set*() methods overwrite them.
template <typename Scalar> struct IntegratorConfig {
    Scalar START_STEP_SIZE_SAFETY = (Scalar)0.90;
    Scalar NEWTON_STEP_REDUCTION = (Scalar)0.25;
    Scalar STEP_NOT_ACCEPTED_REDUCTION = (Scalar)0.25;
    Scalar STEP_NOT_ACCEPTED_SAVETY_STEP = (Scalar)0.50;

    Scalar ORDER_STAY_SAVETY_STEP = (Scalar)0.90;
    Scalar ORDER_UP_SAVETY_STEP = (Scalar)0.90;
    Scalar ORDER_DOWN_SAVETY_STEP = (Scalar)0.90;
    unsigned int allowedSTEPS = 10;

    Scalar NEWTON_SAFETY = (Scalar)-1.00; // negative: rate is estimated automatically
    Scalar NEWTON_CONVERGED = (Scalar)0.01;
    Scalar ACCEPT_NEWTON_CONVERGENCE_RATE = (Scalar)0.30;
    unsigned int allowedNEWTON = 10;
    unsigned int MAXNEWTONITERATION = 3;
    // Opt-in damped Newton: backtrack (halve) the Newton correction while the
    // BDF residual fails to decrease, up to NEWTON_LINE_SEARCH_MAX times.  Off by
    // default (preserves the historical full-step behaviour bit-for-bit).
    bool NEWTON_LINE_SEARCH = false;
    unsigned int NEWTON_LINE_SEARCH_MAX = 6;

    unsigned int MAXORDER = 6;
    unsigned int MINORDER = 1;
    Scalar MACHINEPRECISION = std::numeric_limits<Scalar>::epsilon();
    bool HITSOLUTION = false;
    unsigned int ERROR_WEIGHTS_INDEX = 1;
    unsigned int MINTAYLOR = 1;

    bool updateEveryP = false;

    Scalar RESTRICT_ORDER_1_RATIO = (Scalar)3.000;
    Scalar RESTRICT_ORDER_2_RATIO = (Scalar)2.414;
    Scalar RESTRICT_ORDER_3_RATIO = (Scalar)1.618;
    Scalar RESTRICT_ORDER_4_RATIO = (Scalar)1.280;
    Scalar RESTRICT_ORDER_5_RATIO = (Scalar)1.127;
    Scalar RESTRICT_ORDER_6_RATIO = (Scalar)1.064;
};

// Step/iteration/linear-algebra counters exposed by the diagnostics API.
struct IntegratorStatistics {
    unsigned int acceptedSteps = 0;
    unsigned int rejectedSteps = 0;
    unsigned int rejectedNewtonSteps = 0;
    unsigned int updateJacobian = 0;
    unsigned int updateP = 0;
    unsigned int newtonIterations = 0;
};

} // namespace detail
} // namespace yabi

#endif
