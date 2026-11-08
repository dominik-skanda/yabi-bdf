/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  Copyright (C) 2010-2012 Dominik Skanda
 *
 *  SPDX-License-Identifier: Apache-2.0
 */
#ifndef YABI_DETAIL_CONTINUOUS_ADJOINT_HPP
#define YABI_DETAIL_CONTINUOUS_ADJOINT_HPP

#include <algorithm>
#include <cstddef>
#include <memory>
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
        // Steps are recorded in ascending t_right order, so locate the interval
        // [t_right - h, t_right] containing t with a binary search (O(log n)
        // instead of a linear backward scan).
        auto it = std::lower_bound(steps_.begin(), steps_.end(), t,
                                   [](const TrajectoryStep<Scalar> &s, Scalar val) {
                                       return s.t_right < val;
                                   });
        if (it == steps_.end())
            --it; // t is at/after the last step's right edge
        const TrajectoryStep<Scalar> &s = *it;
        if (t > s.t_right || t < s.t_right - s.h)
            return false;
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

  private:
    std::vector<TrajectoryStep<Scalar>> steps_;
};

// The continuous-adjoint ODE as a time-dependent AbstractRHS, so the existing
// BDF integrator can integrate it backward in reversed time.
//
//   d(lambda)/d tau = J(y(T - tau))^T lambda,   tau in [0, T]
//   d(mu)/d tau     = (d f/d p)^T lambda,       mu(T) = 0  =>  mu(0) = d g/d p
//
// The state y(T - tau) is looked up from a recorded Trajectory.  The
// Jacobian-transpose product J(y)^T v comes from AbstractRHS::adjointProduct,
// and the parameter-adjoint (d f/d p)^T v from adjointParameterProduct — both
// backend-agnostic (CppAD via AD, pure UserODERHS via the analytic Jacobian).
template <typename Scalar> class ContinuousAdjointRHS : public AbstractRHS<Scalar> {
  public:
    ContinuousAdjointRHS(const AbstractRHS<Scalar> &base, const Trajectory<Scalar> &trajectory,
                         std::size_t n, Scalar T, const std::vector<Scalar> &parameter)
        : base_(base.clone()), trajectory_(trajectory), n_(n), p_(parameter.size()), T_(T),
          parameter_(parameter) {}

    std::size_t Domain() const override { return n_ + p_ + 1; } // (lambda, mu, tau)
    std::size_t Range() const override { return n_ + p_; }

    std::vector<Scalar> Forward(std::size_t p, const std::vector<Scalar> &x) const override {
        (void)p; // the augmented adjoint is linear in (lambda, mu), so Forward(0)
                 // and Forward(1) both reduce to J(y)^T / (d f/dp)^T applied to
                 // the state direction.
        // x = (lambda[0..n), mu[n..n+p), tau)
        const Scalar tau = x[n_ + p_];
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
        for (std::size_t i = 0; i < p_; ++i)
            arg[k++] = parameter_[i];

        // Direction: the first n_ entries (lambda for Forward(0), the state
        // direction for Forward(1)); mu and tau directions are always 0 in the
        // Newton Jacobian sweep.
        std::vector<Scalar> dir(x.begin(), x.begin() + n_);

        std::vector<Scalar> Jt;
        if (!base_->adjointProduct(arg, dir, Jt)) {
            // Should not happen: adjointProduct is required for the continuous
            // adjoint and checked up front by continuousAdjoint().
            throw std::runtime_error("ContinuousAdjointRHS: adjointProduct unsupported");
        }
        std::vector<Scalar> pt;
        if (p_ > 0)
            base_->adjointParameterProduct(arg, dir, pt);

        // Augmented RHS value: (J^T dir, (d f/dp)^T dir).
        std::vector<Scalar> out(n_ + p_);
        for (std::size_t i = 0; i < n_; ++i)
            out[i] = Jt[i];
        for (std::size_t i = 0; i < p_; ++i)
            out[n_ + i] = (i < pt.size()) ? pt[i] : Scalar(0);
        return out;
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
    std::size_t p_;
    Scalar T_;
    std::vector<Scalar> parameter_;
};

// The tangent ODE as a time-dependent AbstractRHS, so the existing BDF can
// integrate the forward tangent ydot' = J(y(t)) ydot (needed by the second-order
// continuous adjoint).  The state y(t) is looked up from a recorded Trajectory;
// J(y) ydot is the forward directional derivative Forward(1, ydot).
template <typename Scalar> class TangentRHS : public AbstractRHS<Scalar> {
  public:
    TangentRHS(const AbstractRHS<Scalar> &base, const Trajectory<Scalar> &trajectory,
               std::size_t n, Scalar T, const std::vector<Scalar> &parameter)
        : base_(base.clone()), trajectory_(trajectory), n_(n), T_(T), parameter_(parameter) {
        (void)T_; // the tangent integrates in *forward* time, so T is unused
    }

    std::size_t Domain() const override { return n_ + 1; } // (ydot, t)
    std::size_t Range() const override { return n_; }

    std::vector<Scalar> Forward(std::size_t p, const std::vector<Scalar> &x) const override {
        (void)p; // linear in ydot, so Forward(0) and Forward(1) coincide
        const Scalar t = x[n_];
        std::vector<Scalar> y(n_);
        trajectory_.interpolate(t, y.data(), n_);

        std::vector<Scalar> arg(base_->Domain());
        std::size_t k = 0;
        for (std::size_t i = 0; i < n_; ++i)
            arg[k++] = y[i];
        if (k < base_->Domain())
            arg[k++] = t;
        for (std::size_t i = 0; i < parameter_.size(); ++i)
            arg[k++] = parameter_[i];

        base_->Forward(0, arg);
        std::vector<Scalar> dx(base_->Domain(), Scalar(0));
        for (std::size_t i = 0; i < n_; ++i)
            dx[i] = x[i]; // state direction = ydot
        return base_->Forward(1, dx); // J(y) * ydot
    }

    std::vector<Scalar> Reverse(std::size_t, const std::vector<Scalar> &) const override {
        return {};
    }
    std::size_t size_order() const override { return 1; }
    void capacity_order(std::size_t) override {}
    AbstractRHS<Scalar> *clone() const override {
        return new TangentRHS(*base_, trajectory_, n_, T_, parameter_);
    }

  private:
    std::unique_ptr<AbstractRHS<Scalar>> base_;
    Trajectory<Scalar> trajectory_;
    std::size_t n_;
    Scalar T_;
    std::vector<Scalar> parameter_;
};

// The second-order continuous adjoint as a time-dependent AbstractRHS.  In
// reversed time tau it integrates the augmented pair
//   d lambda/d tau = J^T lambda
//   d gamma /d tau = J^T gamma + Hess(lambda^T f) * ydot
// with gamma(T) = 0, so gamma(0) = H g * s (the Hessian-vector product).  Both
// y(T-tau) and the tangent ydot(T-tau) come from recorded Trajectories; J^T is
// AbstractRHS::adjointProduct and the Hessian term is hessianVectorProduct.
template <typename Scalar> class SecondOrderAdjointRHS : public AbstractRHS<Scalar> {
  public:
    SecondOrderAdjointRHS(const AbstractRHS<Scalar> &base, const Trajectory<Scalar> &trajectory,
                          const Trajectory<Scalar> &tangent_trajectory, std::size_t n, Scalar T,
                          const std::vector<Scalar> &parameter)
        : base_(base.clone()), trajectory_(trajectory), tangent_(tangent_trajectory), n_(n),
          T_(T), parameter_(parameter) {}

    std::size_t Domain() const override { return 2 * n_ + 1; } // (lambda, gamma, tau)
    std::size_t Range() const override { return 2 * n_; }

    std::vector<Scalar> Forward(std::size_t p, const std::vector<Scalar> &x) const override {
        (void)p; // linear in (lambda, gamma)
        const Scalar tau = x[2 * n_];
        const Scalar t = T_ - tau;

        std::vector<Scalar> y(n_), yd(n_);
        trajectory_.interpolate(t, y.data(), n_);
        tangent_.interpolate(t, yd.data(), n_);

        std::vector<Scalar> arg(base_->Domain());
        std::size_t k = 0;
        for (std::size_t i = 0; i < n_; ++i)
            arg[k++] = y[i];
        if (k < base_->Domain())
            arg[k++] = t;
        for (std::size_t i = 0; i < parameter_.size(); ++i)
            arg[k++] = parameter_[i];

        std::vector<Scalar> dir_lam(x.begin(), x.begin() + n_);
        std::vector<Scalar> dir_gam(x.begin() + n_, x.begin() + 2 * n_);

        std::vector<Scalar> Jt_lam, Jt_gam, Hv;
        if (!base_->adjointProduct(arg, dir_lam, Jt_lam) ||
            !base_->adjointProduct(arg, dir_gam, Jt_gam) ||
            !base_->hessianVectorProduct(arg, dir_lam, yd, Hv)) {
            throw std::runtime_error("SecondOrderAdjointRHS: Hessian/adjointhook unsupported");
        }

        std::vector<Scalar> out(2 * n_);
        for (std::size_t i = 0; i < n_; ++i) {
            out[i] = Jt_lam[i];        // d lambda/d tau
            out[n_ + i] = Jt_gam[i] + Hv[i]; // d gamma/d tau
        }
        return out;
    }

    std::vector<Scalar> Reverse(std::size_t, const std::vector<Scalar> &) const override {
        return {};
    }
    std::size_t size_order() const override { return 1; }
    void capacity_order(std::size_t) override {}
    AbstractRHS<Scalar> *clone() const override {
        return new SecondOrderAdjointRHS(*base_, trajectory_, tangent_, n_, T_, parameter_);
    }

  private:
    std::unique_ptr<AbstractRHS<Scalar>> base_;
    Trajectory<Scalar> trajectory_;
    Trajectory<Scalar> tangent_;
    std::size_t n_;
    Scalar T_;
    std::vector<Scalar> parameter_;
};

// The second-order tangent ODE as a time-dependent AbstractRHS, so the existing
// BDF integrates S2' = J(y) S2 + H(y) S S forward (S is the first-order tangent
// looked up from a recorded Trajectory, H S S via secondOrderDirectional).
template <typename Scalar> class SecondOrderTangentRHS : public AbstractRHS<Scalar> {
  public:
    SecondOrderTangentRHS(const AbstractRHS<Scalar> &base, const Trajectory<Scalar> &trajectory,
                          const Trajectory<Scalar> &tangent_trajectory, std::size_t n, Scalar T,
                          const std::vector<Scalar> &parameter)
        : base_(base.clone()), trajectory_(trajectory), tangent_(tangent_trajectory), n_(n),
          T_(T), parameter_(parameter) {
        (void)T_; // forward time
    }

    std::size_t Domain() const override { return n_ + 1; } // (S2, t)
    std::size_t Range() const override { return n_; }

    std::vector<Scalar> Forward(std::size_t p, const std::vector<Scalar> &x) const override {
        const Scalar t = x[n_];
        std::vector<Scalar> y(n_), s(n_);
        trajectory_.interpolate(t, y.data(), n_);
        tangent_.interpolate(t, s.data(), n_);

        std::vector<Scalar> arg(base_->Domain());
        std::size_t k = 0;
        for (std::size_t i = 0; i < n_; ++i)
            arg[k++] = y[i];
        if (k < base_->Domain())
            arg[k++] = t;
        for (std::size_t i = 0; i < parameter_.size(); ++i)
            arg[k++] = parameter_[i];

        base_->Forward(0, arg);
        std::vector<Scalar> dx(base_->Domain(), Scalar(0));
        for (std::size_t i = 0; i < n_; ++i)
            dx[i] = x[i]; // state direction = S2 (or dS2)
        std::vector<Scalar> Jx = base_->Forward(1, dx); // J * S2

        if (p == 0) {
            std::vector<Scalar> HSS;
            if (!base_->secondOrderDirectional(arg, s, HSS))
                throw std::runtime_error("SecondOrderTangentRHS: secondOrderDirectional unsupported");
            for (std::size_t i = 0; i < n_; ++i)
                Jx[i] += HSS[i]; // S2' = J S2 + H S S
        }
        return Jx;
    }

    std::vector<Scalar> Reverse(std::size_t, const std::vector<Scalar> &) const override {
        return {};
    }
    std::size_t size_order() const override { return 1; }
    void capacity_order(std::size_t) override {}
    AbstractRHS<Scalar> *clone() const override {
        return new SecondOrderTangentRHS(*base_, trajectory_, tangent_, n_, T_, parameter_);
    }

  private:
    std::unique_ptr<AbstractRHS<Scalar>> base_;
    Trajectory<Scalar> trajectory_;
    Trajectory<Scalar> tangent_;
    std::size_t n_;
    Scalar T_;
    std::vector<Scalar> parameter_;
};

// The k-th-order tangent ODE as a time-dependent AbstractRHS (T40), so the
// existing BDF integrates S_k' = J(y) S_k + source_k forward, where
// source_k = orderKDirectional(k, y, [S_1..S_{k-1}, S_k]) is the k-th-order
// nonlinear source (excluding the linear J S_k).  y and S_1..S_{k-1} come from
// recorded Trajectories; S_k is the RHS's own state.
template <typename Scalar> class KthOrderTangentRHS : public AbstractRHS<Scalar> {
  public:
    KthOrderTangentRHS(const AbstractRHS<Scalar> &base, const Trajectory<Scalar> &trajectory,
                       const std::vector<Trajectory<Scalar>> &lower, std::size_t k, std::size_t n,
                       Scalar T, const std::vector<Scalar> &parameter)
        : base_(base.clone()), trajectory_(trajectory), lower_(lower), k_(k), n_(n), T_(T),
          parameter_(parameter) {
        (void)T_; // forward time
    }

    std::size_t Domain() const override { return n_ + 1; } // (S_k, t)
    std::size_t Range() const override { return n_; }

    std::vector<Scalar> Forward(std::size_t p, const std::vector<Scalar> &x) const override {
        const Scalar t = x[n_];
        std::vector<Scalar> y(n_);
        trajectory_.interpolate(t, y.data(), n_);

        std::vector<Scalar> arg(base_->Domain());
        std::size_t j = 0;
        for (std::size_t i = 0; i < n_; ++i)
            arg[j++] = y[i];
        if (j < base_->Domain())
            arg[j++] = t;
        for (std::size_t i = 0; i < parameter_.size(); ++i)
            arg[j++] = parameter_[i];

        base_->Forward(0, arg);
        std::vector<Scalar> dx(base_->Domain(), Scalar(0));
        for (std::size_t i = 0; i < n_; ++i)
            dx[i] = x[i]; // state direction = S_k (or dS_k for the Jacobian sweep)
        std::vector<Scalar> Jx = base_->Forward(1, dx); // J * S_k

        if (p == 0) {
            // directions = [S_1 .. S_{k-1} (from trajectories), S_k (from x)]
            std::vector<std::vector<Scalar>> S(k_);
            for (std::size_t m = 0; m + 1 < k_; ++m) {
                S[m].resize(n_);
                lower_[m].interpolate(t, S[m].data(), n_);
            }
            S[k_ - 1] = std::vector<Scalar>(x.begin(), x.begin() + n_);

            std::vector<Scalar> src;
            if (!base_->orderKDirectional(k_, arg, S, src))
                throw std::runtime_error("KthOrderTangentRHS: orderKDirectional unsupported");
            for (std::size_t i = 0; i < n_; ++i)
                Jx[i] += src[i]; // S_k' = J S_k + source_k
        }
        return Jx;
    }

    std::vector<Scalar> Reverse(std::size_t, const std::vector<Scalar> &) const override {
        return {};
    }
    std::size_t size_order() const override { return 1; }
    void capacity_order(std::size_t) override {}
    AbstractRHS<Scalar> *clone() const override {
        return new KthOrderTangentRHS(*base_, trajectory_, lower_, k_, n_, T_, parameter_);
    }

  private:
    std::unique_ptr<AbstractRHS<Scalar>> base_;
    Trajectory<Scalar> trajectory_;
    std::vector<Trajectory<Scalar>> lower_; // S_1 .. S_{k-1}
    std::size_t k_;
    std::size_t n_;
    Scalar T_;
    std::vector<Scalar> parameter_;
};

} // namespace detail
} // namespace yabi

#endif
