/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

#include "engine_fixture.hpp"
#include "test_support.hpp"

int main() {
    yabi_test::EngineFixture f;

    yabi_test::check(f.newton.P.size() == 4, "NewtonSolver owns P");
    yabi_test::check(f.newton.J.size() == 4, "NewtonSolver owns J");
    yabi_test::check(f.newton.ip.size() == 2, "NewtonSolver owns pivots");
    yabi_test::check(f.newton.scaling_D1.size() == 2 && f.newton.scaling_D2.size() == 2,
                     "NewtonSolver owns scalings");
    yabi_test::check(!f.newton.setJacobian && !f.newton.setNewP &&
                         !f.newton.newtonIteration_Started,
                     "NewtonSolver iteration flags start clear");
    yabi_test::check(f.newton.convergenceIndicator == 0.0 && f.newton.newtonFAIL == 0 &&
                         f.newton.stepFAIL == 0,
                     "NewtonSolver diagnostics start at zero");

    // P = [[2, 1], [1, 3]] (column-major), unit error weights.  The historical
    // scaling sweeps rows: D1 = -ceil(logb(row_sum)) = {-1, -2}, then
    // P <- scalbn(P, D1) per row.
    f.newton.P[0] = 2.0;
    f.newton.P[1] = 1.0;
    f.newton.P[2] = 1.0;
    f.newton.P[3] = 3.0;
    yabi_test::check(f.newton.calculateNewtonScaling(), "calculateNewtonScaling succeeds");
    yabi_test::check(f.newton.scaling_D2[0] == 0 && f.newton.scaling_D2[1] == 0,
                     "unit weights leave D2 zero");
    yabi_test::check(f.newton.scaling_D1[0] == -1 && f.newton.scaling_D1[1] == -2,
                     "row scalings from row sums");
    yabi_test::check(yabi_test::near(f.newton.P[0], 1.0) && yabi_test::near(f.newton.P[1], 0.25) &&
                         yabi_test::near(f.newton.P[2], 0.5) &&
                         yabi_test::near(f.newton.P[3], 0.75),
                     "P scaled by row scalings");

    return yabi_test::finish("NewtonSolver unit tests");
}
