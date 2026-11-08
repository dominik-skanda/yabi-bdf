/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// Unit tests for the continuous-adjoint RHS wrappers (TangentRHS and
// ContinuousAdjointRHS) with a mock linear RHS f = A y (A = [[1,2],[3,4]],
// column-major).  The wrappers must compute A*ydot (tangent) and A^T*lambda
// (adjoint) by interpolating the state from a recorded trajectory.

#include "test_support.hpp"

#include <yabi/detail/continuous_adjoint.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace {

// f = A y, A = [[1,2],[3,4]] (column-major).  Forward(1) = A dx (Jacobian is A),
// adjointProduct = A^T lambda.
class LinearRHS : public yabi::AbstractRHS<double> {
  public:
    std::size_t Domain() const override { return 2; }
    std::size_t Range() const override { return 2; }

    std::vector<double> Forward(std::size_t p, const std::vector<double> &x) const override {
        (void)p;
        return {1.0 * x[0] + 2.0 * x[1], 3.0 * x[0] + 4.0 * x[1]};
    }

    bool adjointProduct(const std::vector<double> &x, const std::vector<double> &lambda,
                        std::vector<double> &result) const override {
        (void)x;
        result = {1.0 * lambda[0] + 3.0 * lambda[1], 2.0 * lambda[0] + 4.0 * lambda[1]}; // A^T
        return true;
    }

    std::vector<double> Reverse(std::size_t, const std::vector<double> &) const override {
        return {};
    }
    std::size_t size_order() const override { return 1; }
    void capacity_order(std::size_t) override {}
    AbstractRHS<double> *clone() const override { return new LinearRHS(); }
};

// A constant trajectory y = (2, 3) over [0, 1].
yabi::detail::Trajectory<double> constantTrajectory() {
    yabi::detail::Trajectory<double> t;
    t.record(1.0, 1.0, 0, std::vector<double>{2.0, 3.0});
    return t;
}

} // namespace

int main() {
    LinearRHS rhs;
    yabi::detail::Trajectory<double> traj = constantTrajectory();
    const std::vector<double> no_params;

    // --- TangentRHS: Forward(0) = A * ydot ----------------------------------
    {
        yabi::detail::TangentRHS<double> tangent(rhs, traj, 2, 1.0, no_params);
        std::vector<double> x = {1.0, 0.0, 0.5}; // (ydot=(1,0), t=0.5)
        std::vector<double> out = tangent.Forward(0, x);
        // A*(1,0) = (1, 3)
        yabi_test::check(out.size() == 2 && yabi_test::near(out[0], 1.0) &&
                             yabi_test::near(out[1], 3.0),
                         "TangentRHS = A ydot");
    }

    // --- ContinuousAdjointRHS: Forward(0) = A^T lambda ----------------------
    {
        yabi::detail::ContinuousAdjointRHS<double> adj(rhs, traj, 2, 1.0, no_params);
        // x = (lambda, tau) with lambda=(1,0), tau=0.25 -> t = T - tau = 0.75
        std::vector<double> x = {1.0, 0.0, 0.25};
        std::vector<double> out = adj.Forward(0, x);
        // A^T*(1,0) = (1, 2)
        yabi_test::check(out.size() == 2 && yabi_test::near(out[0], 1.0) &&
                             yabi_test::near(out[1], 2.0),
                         "ContinuousAdjointRHS = A^T lambda");
    }

    // --- clone preserves the wrappers --------------------------------------
    {
        yabi::detail::TangentRHS<double> tangent(rhs, traj, 2, 1.0, no_params);
        std::unique_ptr<yabi::AbstractRHS<double>> copy(tangent.clone());
        std::vector<double> x = {0.0, 1.0, 0.5}; // ydot=(0,1)
        std::vector<double> out = copy->Forward(0, x);
        // A*(0,1) = (2, 4)
        yabi_test::check(out.size() == 2 && yabi_test::near(out[0], 2.0) &&
                             yabi_test::near(out[1], 4.0),
                         "cloned TangentRHS = A ydot");
    }

    return yabi_test::finish("continuous-adjoint RHS wrappers");
}
