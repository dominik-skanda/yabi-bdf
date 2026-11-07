/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  Copyright (C) 2010-2012 Dominik Skanda
 *
 *  SPDX-License-Identifier: Apache-2.0
 */
#ifndef YABI_DETAIL_CONTINUOUS_ADJOINT_HPP
#define YABI_DETAIL_CONTINUOUS_ADJOINT_HPP

#include <cstddef>
#include <vector>

#include <yabi/abstract_rhs.hpp>

namespace yabi {
namespace detail {

// Forward trajectory for the continuous adjoint: the Nordsieck history of every
// accepted step, so the state y(t) can be reconstructed by dense-output
// interpolation during the backward adjoint pass.  One step covers the interval
// [t_right - h, t_right] with the polynomial y(t) = sum_j z[n*j+i] * theta^j,
// theta = (t - t_right)/h — the same interpolant `calculate_output_vector` uses.
template <typename Scalar> struct TrajectoryStep {
    Scalar t_right = 0;
    Scalar h = 0;
    unsigned int order = 1;
    std::vector<Scalar> z; // Nordsieck coefficients, size (order+1)*n (or 7*n)
};

template <typename Scalar> class Trajectory {
  public:
    void clear() { steps_.clear(); }

    // Record one accepted step.  `z` is the Nordsieck coefficient array
    // (size 7*n, but only columns 0..order are meaningful) anchored at t_right.
    void record(Scalar t_right, Scalar h, unsigned int order, const std::vector<Scalar> &z) {
        TrajectoryStep<Scalar> s;
        s.t_right = t_right;
        s.h = h;
        s.order = order;
        s.z = z;
        steps_.push_back(std::move(s));
    }

    std::size_t size() const { return steps_.size(); }
    bool empty() const { return steps_.empty(); }

    // Evaluate y(t) (n components) at the state time t.  Returns false if t is
    // outside the recorded interval.
    bool interpolate(Scalar t, Scalar *y, std::size_t n) const {
        // Steps are recorded in ascending time order; scan backwards so the
        // earliest step covering t is found (steps may abut at a shared node).
        for (auto it = steps_.rbegin(); it != steps_.rend(); ++it) {
            const TrajectoryStep<Scalar> &s = *it;
            if (t <= s.t_right && t >= s.t_right - s.h) {
                const Scalar theta = (t - s.t_right) / s.h;
                for (std::size_t i = 0; i < n; ++i) {
                    Scalar acc = 0;
                    Scalar p = 1; // theta^j
                    for (unsigned int j = 0; j <= s.order; ++j) {
                        acc += s.z[n * j + i] * p;
                        p *= theta;
                    }
                    y[i] = acc;
                }
                return true;
            }
        }
        return false;
    }

  private:
    std::vector<TrajectoryStep<Scalar>> steps_;
};

// The continuous-adjoint ODE as a time-dependent AbstractRHS, so the existing
// BDF integrator can integrate it backward in reversed time.
//
//   d(lambda)/d tau = J(y(T - tau))^T lambda,   tau in [0, T]
//
// The state y(T - tau) is looked up from a recorded Trajectory, and the
// Jacobian-transpose product J(y)^T v is taken from AbstractRHS::Reverse(1, v)
// (the first `n` components).  This keeps the adjoint backend-agnostic: CppAD
// supplies Reverse via AD, the pure UserODERHS supplies it via the analytic
// Jacobian.
template <typename Scalar> class ContinuousAdjointRHS : public AbstractRHS<Scalar> {
  public:
    ContinuousAdjointRHS(const AbstractRHS<Scalar> &base, const Trajectory<Scalar> &trajectory,
                         std::size_t n, Scalar T, const std::vector<Scalar> &parameter)
        : base_(base.clone()), trajectory_(trajectory), n_(n), T_(T), parameter_(parameter) {}

    std::size_t Domain() const override { return n_ + 1; } // (lambda, tau)
    std::size_t Range() const override { return n_; }

    std::vector<Scalar> Forward(std::size_t p, const std::vector<Scalar> &x) const override {
        // x = (lambda[0..n), tau)
        const Scalar tau = x[n_];
        const Scalar t = T_ - tau;
        std::vector<Scalar> y(n_);
        trajectory_.interpolate(t, y.data(), n_);

        // Rebuild the forward argument (state, [time], [parameters]).
        std::vector<Scalar> arg(base_->Domain());
        std::size_t k = 0;
        for (std::size_t i = 0; i < n_; ++i)
            arg[k++] = y[i];
        if (k < base_->Domain())
            arg[k++] = t; // time-dependent
        for (std::size_t i = 0; i < parameter_.size(); ++i)
            arg[k++] = parameter_[i];

        // Direction: lambda for the value, or the state direction for the
        // Newton Jacobian column (tau direction is always 0 in the integrator).
        std::vector<Scalar> dir(x.begin(), x.begin() + n_);

        std::vector<Scalar> Jt_dir;
        if (!base_->adjointProduct(arg, dir, Jt_dir)) {
            // Should not happen: adjointProduct is required for the continuous
            // adjoint and checked up front by continuousAdjoint().
            throw std::runtime_error("ContinuousAdjointRHS: adjointProduct unsupported");
        }
        (void)p; // the adjoint is linear in lambda, so Forward(0) and Forward(1)
                 // both reduce to J(y)^T applied to the state direction.
        return Jt_dir;
    }

    std::vector<Scalar> Reverse(std::size_t, const std::vector<Scalar> &) const override {
        return {}; // unused: the adjoint RHS is only integrated forward
    }

    std::size_t size_order() const override { return 1; }
    void capacity_order(std::size_t) override {}

    AbstractRHS<Scalar> *clone() const override {
        return new ContinuousAdjointRHS(*base_, trajectory_, n_, T_, parameter_);
    }

  private:
    std::unique_ptr<AbstractRHS<Scalar>> base_; // cloned forward RHS (owns)
    Trajectory<Scalar> trajectory_;
    std::size_t n_;
    Scalar T_;
    std::vector<Scalar> parameter_;
};

} // namespace detail
} // namespace yabi

#endif
