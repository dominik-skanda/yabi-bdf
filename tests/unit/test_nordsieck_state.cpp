/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

#include "test_support.hpp"

#include <yabi/detail/nordsieck_state.hpp>

int main() {
    yabi::detail::NordsieckState<double> nordsieck;
    nordsieck.initialize(2);
    std::vector<std::vector<double>> no_derivs;

    yabi_test::check(nordsieck.z_predictor.size() == 14, "z_predictor size n*7");
    yabi_test::check(nordsieck.z_corrector.size() == 14, "z_corrector size n*7");
    yabi_test::check(nordsieck.z_corrector_old.size() == 14, "z_corrector_old size n*7");
    yabi_test::check(nordsieck.corrector.size() == 2, "corrector size n");
    yabi_test::check(nordsieck.corrector_old.size() == 2, "corrector_old size n");
    yabi_test::check(nordsieck.solution_Vector.size() == 2, "solution_Vector size n");
    yabi_test::check(nordsieck.corrector[0] == 1.0 && nordsieck.corrector[1] == 1.0,
                     "initialize sets unit initial correction");
    yabi_test::check(nordsieck.z_predictor[0] == 0.0 && nordsieck.z_corrector[0] == 0.0,
                     "initialize zeroes the history");

    // l_i for order 1: {1, h/h0}.
    nordsieck.order_corrector = 1;
    nordsieck.step_size_h = 0.5;
    nordsieck.stepSizeSet_new[0] = 0.5;
    yabi_test::check(nordsieck.calculate_l_vector(), "calculate_l_vector order 1");
    yabi_test::check(yabi_test::near(nordsieck.l_i[0], 1.0) &&
                         yabi_test::near(nordsieck.l_i[1], 1.0),
                     "l_i order 1 values");

    // l_i for order 2: {1, 3, 2} with h=0.5, h0=0.5, h1=0.25.
    nordsieck.order_corrector = 2;
    nordsieck.stepSizeSet_new[0] = 0.5;
    nordsieck.stepSizeSet_new[1] = 0.25;
    yabi_test::check(nordsieck.calculate_l_vector(), "calculate_l_vector order 2");
    yabi_test::check(yabi_test::near(nordsieck.l_i[0], 1.0) &&
                         yabi_test::near(nordsieck.l_i[1], 3.0) &&
                         yabi_test::near(nordsieck.l_i[2], 2.0),
                     "l_i order 2 values");

    // Predictor for order 1: z_pred[:,0] = z_corr[:,0] + z_corr[:,1],
    // z_pred[:,1] = z_corr[:,1] (constant+linear extrapolation).
    nordsieck.order_corrector = 1;
    nordsieck.z_corrector[0] = 1.0; // col 0, species 0
    nordsieck.z_corrector[1] = 2.0; // col 0, species 1
    nordsieck.z_corrector[2] = 0.5; // col 1, species 0
    nordsieck.z_corrector[3] = -1.0;
    yabi_test::check(nordsieck.calculate_next_predictor(2, 0, no_derivs, no_derivs),
                     "calculate_next_predictor order 1");
    yabi_test::check(yabi_test::near(nordsieck.z_predictor[0], 1.5) &&
                         yabi_test::near(nordsieck.z_predictor[1], 1.0) &&
                         yabi_test::near(nordsieck.z_predictor[2], 0.5) &&
                         yabi_test::near(nordsieck.z_predictor[3], -1.0),
                     "predictor constant+linear extrapolation");

    // Corrector: z_corr = z_pred + corrector * l_i^T.
    nordsieck.order_corrector = 1;
    nordsieck.z_predictor[0] = 1.0;
    nordsieck.z_predictor[1] = 2.0;
    nordsieck.z_predictor[2] = 3.0;
    nordsieck.z_predictor[3] = 4.0;
    nordsieck.corrector[0] = 0.5;
    nordsieck.corrector[1] = -0.5;
    nordsieck.l_i[0] = 1.0;
    nordsieck.l_i[1] = 0.0;
    yabi_test::check(nordsieck.calculate_next_corrector(2, 0, no_derivs, no_derivs, no_derivs),
                     "calculate_next_corrector");
    yabi_test::check(yabi_test::near(nordsieck.z_corrector[0], 1.5) &&
                         yabi_test::near(nordsieck.z_corrector[1], 1.5) &&
                         yabi_test::near(nordsieck.z_corrector[2], 3.0) &&
                         yabi_test::near(nordsieck.z_corrector[3], 4.0),
                     "corrector adds correction*l_i to predictor");

    // Dense output: at t_i[1] only the 0-th coefficient contributes; one step
    // later the first coefficient contributes with weight 1.
    nordsieck.order_corrector = 1;
    nordsieck.step_size_h = 1.0;
    nordsieck.t_i[1] = 5.0;
    nordsieck.z_corrector[0] = 1.0;
    nordsieck.z_corrector[1] = 2.0;
    nordsieck.z_corrector[2] = 3.0;
    nordsieck.z_corrector[3] = 4.0;
    yabi_test::check(nordsieck.calculate_output_vector(2, 0, no_derivs, no_derivs, 5.0),
                     "calculate_output_vector at t_i[1]");
    yabi_test::check(yabi_test::near(nordsieck.solution_Vector[0], 1.0) &&
                         yabi_test::near(nordsieck.solution_Vector[1], 2.0),
                     "dense output at t_i[1]");
    yabi_test::check(nordsieck.calculate_output_vector(2, 0, no_derivs, no_derivs, 6.0),
                     "calculate_output_vector one step ahead");
    yabi_test::check(yabi_test::near(nordsieck.solution_Vector[0], 4.0) &&
                         yabi_test::near(nordsieck.solution_Vector[1], 6.0),
                     "dense output one step ahead");

    return yabi_test::finish("NordsieckState unit tests");
}
