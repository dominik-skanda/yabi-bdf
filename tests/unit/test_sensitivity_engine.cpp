/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

#include "engine_fixture.hpp"
#include "test_support.hpp"

int main() {
    yabi_test::EngineFixture f;

    f.sensitivity.initialize(3);
    yabi_test::check(f.sensitivity.forward_solution_Vector.size() == 3,
                     "SensitivityEngine sizes forward solution vector");
    yabi_test::check(f.sensitivity.correctorVector_tape.size() == 3,
                     "SensitivityEngine sizes replay corrector vector");
    yabi_test::check(f.sensitivity.forward_solution_Vector[0] == 0.0,
                     "forward solution starts at zero");

    f.sensitivity.forward_solution_Vector[0] = 7.0;
    f.sensitivity.forward_solution_Vector[2] = -3.0;
    f.sensitivity.zeroForwardSolution();
    yabi_test::check(f.sensitivity.forward_solution_Vector[0] == 0.0 &&
                         f.sensitivity.forward_solution_Vector[2] == 0.0,
                     "zeroForwardSolution clears every entry");

    // A new tape invalidates the forward replay and the previous order.
    f.sensitivity.getTAPE = true;
    f.sensitivity.lastForwardOrder = 5;
    f.sensitivity.beginNewTape();
    yabi_test::check(!f.sensitivity.getTAPE && f.sensitivity.lastForwardOrder == 0,
                     "beginNewTape resets replay state");

    // Per-tape replay buffers are released on tape reset.
    yabi::detail::ReplayTape<double> node;
    f.sensitivity.adjointTapePointer = &node;
    f.sensitivity.z_predictor_tape.push_back(1.0);
    f.sensitivity.z_corrector_tape.push_back(2.0);
    f.sensitivity.P_tape.push_back(3.0);
    f.sensitivity.ip_tape.push_back(4);
    f.sensitivity.scaling_D1_tape.push_back(5);
    f.sensitivity.scaling_D2_tape.push_back(6);
    f.sensitivity.resetForwardBuffers();
    yabi_test::check(f.sensitivity.adjointTapePointer == nullptr,
                     "resetForwardBuffers clears cursor");
    yabi_test::check(f.sensitivity.z_predictor_tape.empty() &&
                         f.sensitivity.z_corrector_tape.empty() && f.sensitivity.P_tape.empty() &&
                         f.sensitivity.ip_tape.empty() && f.sensitivity.scaling_D1_tape.empty() &&
                         f.sensitivity.scaling_D2_tape.empty(),
                     "resetForwardBuffers releases replay buffers");

    // Releasing with no simultaneous derivatives is a safe no-op.
    yabi_test::check(f.sensitivity.freeSimultaneousDerivatives(),
                     "freeSimultaneousDerivatives with no derivatives");
    yabi_test::check(f.sensitivity.numOfDerivatives == 0 &&
                         f.sensitivity.z_predictor_derivatives == nullptr,
                     "freeSimultaneousDerivatives keeps empty state null");

    return yabi_test::finish("SensitivityEngine unit tests");
}
