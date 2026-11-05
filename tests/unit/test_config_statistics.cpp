/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

#include "test_support.hpp"

#include <yabi/detail/integrator_config.hpp>

#include <limits>

int main() {
    yabi::detail::IntegratorConfig<double> config;

    yabi_test::check(config.START_STEP_SIZE_SAFETY == 0.90, "config.START_STEP_SIZE_SAFETY");
    yabi_test::check(config.NEWTON_STEP_REDUCTION == 0.25, "config.NEWTON_STEP_REDUCTION");
    yabi_test::check(config.STEP_NOT_ACCEPTED_REDUCTION == 0.25,
                     "config.STEP_NOT_ACCEPTED_REDUCTION");
    yabi_test::check(config.STEP_NOT_ACCEPTED_SAVETY_STEP == 0.50,
                     "config.STEP_NOT_ACCEPTED_SAVETY_STEP");
    yabi_test::check(config.ORDER_STAY_SAVETY_STEP == 0.90, "config.ORDER_STAY_SAVETY_STEP");
    yabi_test::check(config.ORDER_UP_SAVETY_STEP == 0.90, "config.ORDER_UP_SAVETY_STEP");
    yabi_test::check(config.ORDER_DOWN_SAVETY_STEP == 0.90, "config.ORDER_DOWN_SAVETY_STEP");
    yabi_test::check(config.allowedSTEPS == 10, "config.allowedSTEPS");
    yabi_test::check(config.NEWTON_SAFETY == -1.00, "config.NEWTON_SAFETY");
    yabi_test::check(config.NEWTON_CONVERGED == 0.01, "config.NEWTON_CONVERGED");
    yabi_test::check(config.ACCEPT_NEWTON_CONVERGENCE_RATE == 0.30,
                     "config.ACCEPT_NEWTON_CONVERGENCE_RATE");
    yabi_test::check(config.allowedNEWTON == 10, "config.allowedNEWTON");
    yabi_test::check(config.MAXNEWTONITERATION == 3, "config.MAXNEWTONITERATION");
    yabi_test::check(config.MAXORDER == 6, "config.MAXORDER");
    yabi_test::check(config.MINORDER == 1, "config.MINORDER");
    yabi_test::check(config.MACHINEPRECISION == std::numeric_limits<double>::epsilon(),
                     "config.MACHINEPRECISION");
    yabi_test::check(config.HITSOLUTION == false, "config.HITSOLUTION");
    yabi_test::check(config.ERROR_WEIGHTS_INDEX == 1, "config.ERROR_WEIGHTS_INDEX");
    yabi_test::check(config.MINTAYLOR == 1, "config.MINTAYLOR");
    yabi_test::check(config.updateEveryP == false, "config.updateEveryP");
    yabi_test::check(config.RESTRICT_ORDER_1_RATIO == 3.000, "config.RESTRICT_ORDER_1_RATIO");
    yabi_test::check(config.RESTRICT_ORDER_2_RATIO == 2.414, "config.RESTRICT_ORDER_2_RATIO");
    yabi_test::check(config.RESTRICT_ORDER_3_RATIO == 1.618, "config.RESTRICT_ORDER_3_RATIO");
    yabi_test::check(config.RESTRICT_ORDER_4_RATIO == 1.280, "config.RESTRICT_ORDER_4_RATIO");
    yabi_test::check(config.RESTRICT_ORDER_5_RATIO == 1.127, "config.RESTRICT_ORDER_5_RATIO");
    yabi_test::check(config.RESTRICT_ORDER_6_RATIO == 1.064, "config.RESTRICT_ORDER_6_RATIO");

    // Config is plain mutable state, independent of the integrator.
    config.MAXORDER = 4;
    config.HITSOLUTION = true;
    yabi_test::check(config.MAXORDER == 4, "config.MAXORDER writable");
    yabi_test::check(config.HITSOLUTION == true, "config.HITSOLUTION writable");

    yabi::detail::IntegratorStatistics stats;
    yabi_test::check(stats.acceptedSteps == 0 && stats.rejectedSteps == 0 &&
                         stats.rejectedNewtonSteps == 0 && stats.updateJacobian == 0 &&
                         stats.updateP == 0 && stats.newtonIterations == 0,
                     "statistics zero-initialised");

    return yabi_test::finish("config/statistics unit tests");
}
