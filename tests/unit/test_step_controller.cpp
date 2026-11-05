/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

#include "test_support.hpp"

#include <yabi/detail/integrator_config.hpp>
#include <yabi/detail/nordsieck_state.hpp>
#include <yabi/detail/step_controller.hpp>

int main() {
    yabi::detail::NordsieckState<double> nordsieck;
    yabi::detail::IntegratorConfig<double> config;
    yabi::detail::StepController<double> controller(nordsieck, config);

    // Mode 0: max(existing weight, |y|, atol).
    double weights0[2] = {0.1, 5.0};
    const double y0[2] = {-2.0, 1.0};
    const double atol0[2] = {0.5, 0.3};
    yabi_test::check(
        yabi::detail::StepController<double>::calculateErrorWeights(2, 0, weights0, y0, atol0, 0.5),
        "calculateErrorWeights mode 0");
    yabi_test::check(yabi_test::near(weights0[0], 2.0) && yabi_test::near(weights0[1], 5.0),
                     "mode 0 raises to max(|y|, atol, weight)");

    // Mode 1: |y| + atol/rtol.
    double weights1[2] = {0.0, 0.0};
    const double y1[2] = {-2.0, 1.0};
    const double atol1[2] = {0.5, 0.3};
    yabi_test::check(
        yabi::detail::StepController<double>::calculateErrorWeights(2, 1, weights1, y1, atol1, 0.5),
        "calculateErrorWeights mode 1");
    yabi_test::check(yabi_test::near(weights1[0], 3.0) && yabi_test::near(weights1[1], 1.6),
                     "mode 1 uses |y| + atol/rtol");

    return yabi_test::finish("StepController unit tests");
}
