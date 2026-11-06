#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <yabi/abstract_rhs.hpp>
#include <yabi/sparsity_pattern.hpp>

namespace yabi {

// Pure (CppAD-free) AbstractRHS adapter for an autonomous ODE f(y) with an
// analytic Jacobian J(y) = df/dy (column-major: J[i + n*j] = df_i/dy_j).
//
// Forward(0, x) evaluates f(x) and remembers x as the "point" (matching
// CppAD's stateful ADFun semantics: the core sets the point with Forward(0)
// and then queries J(point) via Forward(1, dx)).
//
// Optional sparsity / matrix-free hooks:
//   * setSparsityPattern() supplies the structural pattern of J = df/dy.  It
//     enables the same CSC sparse and matrix-free Newton paths that CppAD
//     tapes get through CppADRHS.
//   * setJacobianVectorProduct() supplies an analytic J(y)*v, so the
//     matrix-free path never forms the full n x n Jacobian.
// Both are additive: without them the backend keeps the historical dense
// behaviour.
//
// Forward(2, …) (exact second-order / Hessian-in-direction) and Reverse(…)
// (adjoint) are not provided by default and throw — pure mode supports forward
// and simultaneous sensitivity, not the adjoint (which needs an AD tape to
// reverse) and not second order without the Hessian hook (see T28/T29).
template <typename Scalar> class UserODERHS : public AbstractRHS<Scalar> {
  public:
    using ODE = std::function<void(const Scalar *y, Scalar *f)>;
    using Jacobian = std::function<void(const Scalar *y, Scalar *J)>; // n*n, column-major
    using JacobianVector =
        std::function<void(const Scalar *y, const Scalar *v, Scalar *Jv)>; // n, J(y)*v

    UserODERHS(std::size_t n, ODE f, Jacobian jac) : n_(n), f_(std::move(f)), jac_(std::move(jac)) {}

    // Structural sparsity of J = df/dy, column-oriented: column_rows[j] lists
    // the rows i for which J(i,j) may be non-zero.  The pattern is the user's
    // structural contract; entries not listed are treated as structurally
    // zero.  Columns are normalised (sorted, duplicate rows removed).
    UserODERHS &setSparsityPattern(SparsityPattern pattern) {
        if (pattern.rows != n_ || pattern.cols != n_ || pattern.column_rows.size() != n_)
            throw std::invalid_argument(
                "UserODERHS: sparsity pattern must be a square n x n column-oriented pattern");
        for (std::size_t j = 0; j < n_; ++j) {
            auto &rows = pattern.column_rows[j];
            std::sort(rows.begin(), rows.end());
            rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
            for (std::size_t row : rows)
                if (row >= n_)
                    throw std::invalid_argument(
                        "UserODERHS: sparsity pattern row index out of range");
        }
        pattern_ = std::move(pattern);
        pattern_valid_ = true;
        return *this;
    }

    // Optional analytic J(y)*v.  v and Jv have n entries.  When set, the
    // matrix-free Newton path uses it directly and never forms J.
    UserODERHS &setJacobianVectorProduct(JacobianVector jv) {
        jv_ = std::move(jv);
        return *this;
    }

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

    AbstractRHS<Scalar> *clone() const override {
        auto *copy = new UserODERHS<Scalar>(n_, f_, jac_);
        copy->pattern_ = pattern_;
        copy->pattern_valid_ = pattern_valid_;
        copy->jv_ = jv_;
        return copy;
    }

    // ---- Optional sparsity / matrix-free hooks (T32b) ----
    bool hasSparsityPattern() const override { return pattern_valid_; }

    bool jacobianSparsity(SparsityPattern &pattern) const override {
        if (!pattern_valid_)
            return false;
        pattern = pattern_;
        return true;
    }

    bool sparseJacobian(const std::vector<Scalar> &x, const SparsityPattern &pattern,
                        std::vector<Scalar> &values) const override {
        if (!pattern_valid_ || pattern != pattern_ || x.size() != n_)
            return false;

        std::vector<Scalar> J(n_ * n_);
        jac_(x.data(), J.data());

        // Same column-major order as CppADRHS::sparseJacobian: columns
        // ascending, rows ascending within a column (the pattern is sorted).
        values.clear();
        values.reserve(pattern_.nnz());
        for (std::size_t j = 0; j < n_; ++j)
            for (std::size_t i : pattern_.column_rows[j])
                values.push_back(J[i + n_ * j]);
        return true;
    }

    bool jacobianVectorProduct(const std::vector<Scalar> &x, const std::vector<Scalar> &v,
                               std::vector<Scalar> &result) const override {
        if (x.size() != n_ || v.size() != n_)
            return false;
        result.assign(n_, Scalar(0));

        if (jv_) {
            jv_(x.data(), v.data(), result.data());
            return true;
        }

        // No analytic product supplied: evaluate J once and multiply.  This is
        // still exact; the hook above simply avoids forming J.
        std::vector<Scalar> J(n_ * n_);
        jac_(x.data(), J.data());
        for (std::size_t i = 0; i < n_; ++i)
            for (std::size_t j = 0; j < n_; ++j)
                result[i] += J[i + n_ * j] * v[j];
        return true;
    }

  private:
    std::size_t n_;
    ODE f_;
    Jacobian jac_;
    SparsityPattern pattern_;
    bool pattern_valid_ = false;
    JacobianVector jv_;
    mutable std::vector<Scalar> point_; // last Forward(0) argument (stateful point)
};

} // namespace yabi
