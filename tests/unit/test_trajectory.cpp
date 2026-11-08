/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// Unit tests for the Trajectory record/interpolate used by the continuous
// adjoint and the higher-order forward sensitivities.  The interpolation is the
// dense-output polynomial z[j]*theta^j (theta = (t - t_right)/h), verified
// against hand-built constant/linear/quadratic trajectories and multi-step
// binary search.

#include "test_support.hpp"

#include <yabi/detail/continuous_adjoint.hpp>

#include <vector>

int main() {
    using yabi::detail::Trajectory;

    // --- single constant step (order 0) ------------------------------------
    {
        Trajectory<double> traj;
        traj.record(1.0, 1.0, 0, std::vector<double>{5.0}); // y(t)=5 on [0,1]
        double y;
        yabi_test::check(traj.interpolate(0.0, &y, 1) && yabi_test::near(y, 5.0),
                         "constant step at left edge");
        yabi_test::check(traj.interpolate(0.5, &y, 1) && yabi_test::near(y, 5.0),
                         "constant step mid-interval");
        yabi_test::check(traj.interpolate(1.0, &y, 1) && yabi_test::near(y, 5.0),
                         "constant step at right edge");
        yabi_test::check(!traj.interpolate(1.01, &y, 1), "reject past the right edge");
        yabi_test::check(!traj.interpolate(-0.01, &y, 1), "reject before the left edge");
    }

    // --- single linear step (order 1), y(t) = 2t + 1 ------------------------
    {
        Trajectory<double> traj;
        traj.record(1.0, 1.0, 1, std::vector<double>{3.0, 2.0}); // z0=3, z1=2
        double y;
        yabi_test::check(traj.interpolate(0.0, &y, 1) && yabi_test::near(y, 1.0),
                         "linear at t=0");
        yabi_test::check(traj.interpolate(0.5, &y, 1) && yabi_test::near(y, 2.0),
                         "linear at t=0.5");
        yabi_test::check(traj.interpolate(1.0, &y, 1) && yabi_test::near(y, 3.0),
                         "linear at t=1");
    }

    // --- single quadratic step (order 2), y(t) = t^2 ------------------------
    {
        Trajectory<double> traj;
        // z0=y(1)=1, z1=y'(1)*h=2, z2=y''(1)*h^2/2=1
        traj.record(1.0, 1.0, 2, std::vector<double>{1.0, 2.0, 1.0});
        double y;
        yabi_test::check(traj.interpolate(0.0, &y, 1) && yabi_test::near(y, 0.0),
                         "quadratic at t=0");
        yabi_test::check(traj.interpolate(0.5, &y, 1) && yabi_test::near(y, 0.25),
                         "quadratic at t=0.5");
        yabi_test::check(traj.interpolate(1.0, &y, 1) && yabi_test::near(y, 1.0),
                         "quadratic at t=1");
    }

    // --- two-component linear (n=2), y0=t, y1=2t ----------------------------
    {
        Trajectory<double> traj;
        // z = [z0_0, z0_1, z1_0, z1_1] = [1, 2, 1, 2] at t_right=1, h=1
        traj.record(1.0, 1.0, 1, std::vector<double>{1.0, 2.0, 1.0, 2.0});
        double y[2];
        yabi_test::check(traj.interpolate(0.5, y, 2), "two-component interpolate succeeds");
        yabi_test::check(yabi_test::near(y[0], 0.5) && yabi_test::near(y[1], 1.0),
                         "two-component linear values");
    }

    // --- multi-step binary search over a continuous y = 2t + 1 --------------
    {
        Trajectory<double> traj;
        traj.record(1.0, 1.0, 1, std::vector<double>{3.0, 2.0}); // [0,1]
        traj.record(2.0, 1.0, 1, std::vector<double>{5.0, 2.0}); // [1,2]
        double y;
        yabi_test::check(traj.interpolate(0.25, &y, 1) && yabi_test::near(y, 1.5),
                         "multi-step finds step 1");
        yabi_test::check(traj.interpolate(1.75, &y, 1) && yabi_test::near(y, 4.5),
                         "multi-step finds step 2");
        yabi_test::check(traj.interpolate(2.0, &y, 1) && yabi_test::near(y, 5.0),
                         "multi-step at the final right edge");
        yabi_test::check(!traj.interpolate(2.01, &y, 1), "multi-step reject past the end");
    }

    // --- clear/empty --------------------------------------------------------
    {
        Trajectory<double> traj;
        traj.record(1.0, 1.0, 0, std::vector<double>{1.0});
        yabi_test::check(traj.size() == 1, "size after record");
        traj.clear();
        yabi_test::check(traj.empty() && traj.size() == 0, "clear empties the trajectory");
        double y;
        yabi_test::check(!traj.interpolate(0.5, &y, 1), "empty trajectory rejects interpolate");
    }

    return yabi_test::finish("Trajectory unit tests");
}
