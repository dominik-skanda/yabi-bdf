/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

#include "engine_fixture.hpp"
#include "test_support.hpp"

int main() {
    yabi_test::EngineFixture f;

    yabi_test::check(f.adjoint.solution_adjoint_species == nullptr,
                     "AdjointEngine species starts null");
    yabi_test::check(f.adjoint.solution_adjoint_parameter == nullptr,
                     "AdjointEngine parameter starts null");
    yabi_test::check(f.adjoint.stored_adjoint_order == 0, "AdjointEngine order starts at zero");

    // Simulate a completed reverse sweep and check release() owns the cleanup.
    f.adjoint.solution_adjoint_species = new double *[f.numOfSpecies];
    for (unsigned int i = 0; i < f.numOfSpecies; i++)
        f.adjoint.solution_adjoint_species[i] = new double[2]{1.0, 2.0};
    f.adjoint.solution_adjoint_parameter = new double *[f.sizeOfParameter];
    for (unsigned int i = 0; i < f.sizeOfParameter; i++)
        f.adjoint.solution_adjoint_parameter[i] = new double[2]{3.0, 4.0};
    f.adjoint.stored_adjoint_order = 2;

    f.adjoint.release();
    yabi_test::check(f.adjoint.solution_adjoint_species == nullptr,
                     "release clears species arrays");
    yabi_test::check(f.adjoint.solution_adjoint_parameter == nullptr,
                     "release clears parameter arrays");
    yabi_test::check(f.adjoint.stored_adjoint_order == 0, "release clears stored order");

    // Idempotent and safe on already-empty state.
    f.adjoint.release();
    yabi_test::check(f.adjoint.solution_adjoint_species == nullptr, "release is idempotent");

    return yabi_test::finish("AdjointEngine unit tests");
}
