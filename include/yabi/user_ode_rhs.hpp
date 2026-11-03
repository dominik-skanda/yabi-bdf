#pragma once

#include <cstddef>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <yabi/abstract_rhs.hpp>

namespace yabi {

// Pure (CppAD-free) AbstractRHS adapter for an autonomous ODE f(y) with an
// analytic Jacobian J(y) = df/dy (column-major: J[i + n*j] = df_i/dy_j).
//
// Forward(0, x) evaluates f(x) and remembers x as the "point" (matching
// CppAD's stateful ADFun semantics: the core sets the point with Forward(0)
// and then queries J(point) via Forward(1, dx)).
//
// Forward(2, …) (exact second-order / Hessian-in-direction) and Reverse(…)
// (adjoint) are not provided by default and throw — pure mode supports forward
// and simultaneous sensitivity, not the adjoint (which needs an AD tape to
// reverse) and not second order without the Hessian hook (see T28/T29).
template <typename Scalar> class UserODERHS : public AbstractRHS<Scalar> {
  public:
    using ODE = std::function<void(const Scalar *y, Scalar *f)>;
    using Jacobian = std::function<void(const Scalar *y, Scalar *J)>; // n*n, column-major

    UserODERHS(std::size_t n, ODE f, Jacobian jac) : n_(n), f_(std::move(f)), jac_(std::move(jac)) {}

    std::size_t Domain() const override { return n_; }
    std::size_t Range() const override { return n_; }

    std::vector<Scalar> Forward(std::size_t p, const std::vector<Scalar> &x) const override {
        if (p == 0) {
            point_ = x;
            std::vector<Scalar> f(n_);
            f_(x.data(), f.data());
            return f;
        }
        if (p == 1) {
            std::vector<Scalar> J(n_ * n_);
            jac_(point_.data(), J.data());
            std::vector<Scalar> df(n_, Scalar(0));
            for (std::size_t i = 0; i < n_; ++i)
                for (std::size_t j = 0; j < n_; ++j)
                    df[i] += J[i + n_ * j] * x[j];
            return df;
        }
        throw std::runtime_error(
            "UserODERHS: Forward(2) (Hessian-in-direction) not provided — pure mode second-order "
            "sensitivity needs an analytic Hessian hook");
    }

    std::vector<Scalar> Reverse(std::size_t, const std::vector<Scalar> &) const override {
        throw std::runtime_error("UserODERHS: Reverse (adjoint) unavailable without an AD tape");
    }

    std::size_t size_order() const override { return 1; }
    void capacity_order(std::size_t) override {}
    AbstractRHS<Scalar> *clone() const override { return new UserODERHS<Scalar>(n_, f_, jac_); }

  private:
    std::size_t n_;
    ODE f_;
    Jacobian jac_;
    mutable std::vector<Scalar> point_; // last Forward(0) argument (stateful point)
};

} // namespace yabi
