/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

#include "engine_fixture.hpp"
#include "test_support.hpp"

int main() {
    yabi_test::EngineFixture f;

    yabi_test::check(f.adjoint.solution_adjoint_species.empty(),
                     "AdjointEngine species starts empty");
    yabi_test::check(f.adjoint.solution_adjoint_parameter.empty(),
                     "AdjointEngine parameter starts empty");
    yabi_test::check(f.adjoint.stored_adjoint_order == 0, "AdjointEngine order starts at zero");

    // Simulate a completed reverse sweep and check release() owns the cleanup.
    f.adjoint.solution_adjoint_species.assign(f.numOfSpecies, std::vector<double>(2, 0.0));
    for (unsigned int i = 0; i < f.numOfSpecies; i++) {
        f.adjoint.solution_adjoint_species[i][0] = 1.0;
        f.adjoint.solution_adjoint_species[i][1] = 2.0;
    }
    f.adjoint.solution_adjoint_parameter.assign(f.sizeOfParameter, std::vector<double>(2, 0.0));
    for (unsigned int i = 0; i < f.sizeOfParameter; i++) {
        f.adjoint.solution_adjoint_parameter[i][0] = 3.0;
        f.adjoint.solution_adjoint_parameter[i][1] = 4.0;
    }
    f.adjoint.stored_adjoint_order = 2;

    f.adjoint.release();
    yabi_test::check(f.adjoint.solution_adjoint_species.empty(), "release clears species arrays");
    yabi_test::check(f.adjoint.solution_adjoint_parameter.empty(),
                     "release clears parameter arrays");
    yabi_test::check(f.adjoint.stored_adjoint_order == 0, "release clears stored order");

    // Idempotent and safe on already-empty state.
    f.adjoint.release();
    yabi_test::check(f.adjoint.solution_adjoint_species.empty(), "release is idempotent");

    return yabi_test::finish("AdjointEngine unit tests");
}
