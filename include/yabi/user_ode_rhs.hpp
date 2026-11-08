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

// Pure (CppAD-free) AbstractRHS adapter for an ODE f(y) with an analytic
// Jacobian J(y) = df/dy (column-major: J[i + n*j] = df_i/dy_j).  Autonomous by
// default; setTimeDerivative(ft) and setParameterDerivative(n, fp) extend it to
// time-dependent f(t, y) and time+parameter f(t, y, p) forms, in which case the
// callbacks receive the full argument (y, [t], [p]).
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
    using TimeDeriv = std::function<void(const Scalar *y, Scalar *ft)>;    // n, df/dt(y)
    using ParamDeriv =
        std::function<void(const Scalar *y, Scalar *fp)>; // n*n_param, column-major df/dp
    // Hessian of lambda^T f (state block) times direction v:
    //   result[i] = sum_{j,k} d^2(lambda^T f)/dy_i dy_j dy_k * v[j] ... = Hess(lambda^T f) * v.
    using HessianVector = std::function<void(const Scalar *y, const Scalar *lambda,
                                             const Scalar *v, Scalar *result)>; // n entries
    // Second-order directional derivative of the vector RHS: H(y) * v * v, n entries.
    using SecondOrderDir = std::function<void(const Scalar *y, const Scalar *v, Scalar *result)>;
    // k-th-order nonlinear source of the vector RHS: result = k! y^(k) - J(x) S_k,
    // n entries.  x is the full Domain() argument; S holds k state directions,
    // flattened column-major as S[m*n + i] = S_{m+1}[i] (m = 0..k-1).
    using OrderKDirectional =
        std::function<void(unsigned k, const Scalar *x, const Scalar *S, Scalar *result)>;

    UserODERHS(std::size_t n, ODE f, Jacobian jac) : n_(n), f_(std::move(f)), jac_(std::move(jac)) {}

    // Mark the RHS as time-dependent and supply df/dt.  The ODE/Jacobian/…/
    // callbacks then receive the full argument (y, [t], [p]) instead of just y,
    // with t at index n and parameters at n+1 …
    UserODERHS &setTimeDerivative(TimeDeriv ft) {
        ft_ = std::move(ft);
        has_time_ = true;
        return *this;
    }

    // Mark the RHS as parameter-dependent and supply df/dp (column-major,
    // n x n_param).  Call setTimeDerivative() first — parameters follow time in
    // the argument layout.
    UserODERHS &setParameterDerivative(std::size_t n_param, ParamDeriv fp) {
        n_param_ = n_param;
        fp_ = std::move(fp);
        return *this;
    }

    // Optional analytic Hessian-in-direction hook for the second-order
    // continuous adjoint (T26): Hess(lambda^T f) * v, n entries.
    UserODERHS &setHessianVectorProduct(HessianVector hv) {
        hv_ = std::move(hv);
        return *this;
    }

    // Optional analytic second-order directional derivative H(y)*v*v for the
    // second-order simultaneous/forward sensitivity (T27), n entries.
    UserODERHS &setSecondOrderDirectional(SecondOrderDir sod) {
        sod_ = std::move(sod);
        return *this;
    }

    // Optional analytic k-th-order nonlinear source (T40) for higher-order
    // simultaneous/forward sensitivity, n entries.  When unset, forwardOrder(k)
    // (k >= 2) fails loudly rather than falling back.
    UserODERHS &setOrderKDirectional(OrderKDirectional okd) {
        okd_ = std::move(okd);
        return *this;
    }

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

    std::size_t Domain() const override { return n_ + (has_time_ ? 1 : 0) + n_param_; }
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
            if (has_time_ && x.size() > n_) {
                std::vector<Scalar> ft(n_);
                ft_(point_.data(), ft.data());
                for (std::size_t i = 0; i < n_; ++i)
                    df[i] += ft[i] * x[n_];
            }
            if (n_param_ > 0 && x.size() > n_ + 1) {
                std::vector<Scalar> fp(n_ * n_param_);
                fp_(point_.data(), fp.data());
                for (std::size_t i = 0; i < n_; ++i)
                    for (std::size_t j = 0; j < n_param_; ++j)
                        df[i] += fp[i + n_ * j] * x[n_ + 1 + j];
            }
            return df;
        }
        throw std::runtime_error(
            "UserODERHS: Forward(2) (Hessian-in-direction) not provided — pure mode second-order "
            "sensitivity needs an analytic Hessian hook");
    }

    std::vector<Scalar> Reverse(std::size_t, const std::vector<Scalar> &) const override {
        throw std::runtime_error("UserODERHS: Reverse (adjoint) unavailable without an AD tape");
    }

    // Continuous-adjoint Jacobian-transpose product J(y)^T lambda via the
    // analytic Jacobian.
    bool adjointProduct(const std::vector<Scalar> &x, const std::vector<Scalar> &lambda,
                        std::vector<Scalar> &result) const override {
        if (x.size() != Domain() || lambda.size() != n_)
            return false;
        std::vector<Scalar> J(n_ * n_);
        jac_(x.data(), J.data());
        result.assign(n_, Scalar(0));
        for (std::size_t k = 0; k < n_; ++k)
            for (std::size_t i = 0; i < n_; ++i)
                result[k] += J[i + n_ * k] * lambda[i]; // J[i + n_*k] = df_i/dy_k
        return true;
    }

    // (d f/d p)^T lambda via the analytic parameter derivative (empty result
    // for autonomous / time-only RHS).
    bool adjointParameterProduct(const std::vector<Scalar> &x, const std::vector<Scalar> &lambda,
                                 std::vector<Scalar> &result) const override {
        if (n_param_ == 0 || !fp_ || x.size() != Domain() || lambda.size() != n_)
            return false;
        std::vector<Scalar> fp(n_ * n_param_);
        fp_(x.data(), fp.data()); // fp[i + n_*j] = df_i/dp_j
        result.assign(n_param_, Scalar(0));
        for (std::size_t j = 0; j < n_param_; ++j)
            for (std::size_t i = 0; i < n_; ++i)
                result[j] += fp[i + n_ * j] * lambda[i];
        return true;
    }

    // Hess(lambda^T f) * v via the analytic hook.
    bool hessianVectorProduct(const std::vector<Scalar> &x, const std::vector<Scalar> &lambda,
                              const std::vector<Scalar> &v,
                              std::vector<Scalar> &result) const override {
        if (!hv_ || x.size() != Domain() || lambda.size() != n_ || v.size() != n_)
            return false;
        result.assign(n_, Scalar(0));
        hv_(x.data(), lambda.data(), v.data(), result.data());
        return true;
    }

    // H(y) * v * v via the analytic hook.
    bool secondOrderDirectional(const std::vector<Scalar> &x, const std::vector<Scalar> &v,
                                std::vector<Scalar> &result) const override {
        if (!sod_ || x.size() != Domain() || v.size() != n_)
            return false;
        result.assign(n_, Scalar(0));
        sod_(x.data(), v.data(), result.data());
        return true;
    }

    // k-th-order nonlinear source k! y^(k) - J S_k via the analytic hook.
    bool orderKDirectional(std::size_t k, const std::vector<Scalar> &x,
                           const std::vector<std::vector<Scalar>> &S,
                           std::vector<Scalar> &result) const override {
        if (k < 2 || !okd_ || x.size() != Domain())
            return false;
        for (const auto &s : S)
            if (s.size() != n_)
                return false;
        std::vector<Scalar> Sflat(k * n_);
        for (std::size_t m = 0; m < k; ++m)
            for (std::size_t i = 0; i < n_; ++i)
                Sflat[m * n_ + i] = S[m][i];
        result.assign(n_, Scalar(0));
        okd_((unsigned)k, x.data(), Sflat.data(), result.data());
        return true;
    }

    std::size_t size_order() const override { return 1; }
    void capacity_order(std::size_t) override {}

    AbstractRHS<Scalar> *clone() const override {
        auto *copy = new UserODERHS<Scalar>(n_, f_, jac_);
        copy->pattern_ = pattern_;
        copy->pattern_valid_ = pattern_valid_;
        copy->jv_ = jv_;
        copy->ft_ = ft_;
        copy->has_time_ = has_time_;
        copy->fp_ = fp_;
        copy->n_param_ = n_param_;
        copy->hv_ = hv_;
        copy->sod_ = sod_;
        copy->okd_ = okd_;
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
        if (!pattern_valid_ || pattern != pattern_ || x.size() != Domain())
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
        if (x.size() != Domain() || v.size() != n_)
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

    // Batched J*V.  With the analytic hook, one hook call per direction; without
    // it, evaluate the analytic Jacobian once and apply it to every column.
    bool jacobianMatrixProduct(const std::vector<Scalar> &x,
                               const std::vector<Scalar> &directions, unsigned int columns,
                               std::vector<Scalar> &result) const override {
        if (x.size() != Domain())
            return false;
        if (columns == 0) {
            result.clear();
            return directions.empty();
        }
        if (directions.size() != n_ * (std::size_t)columns)
            return false;

        result.assign(n_ * (std::size_t)columns, Scalar(0));
        if (jv_) {
            for (unsigned int c = 0; c < columns; ++c)
                jv_(x.data(), directions.data() + (std::size_t)c * n_,
                    result.data() + (std::size_t)c * n_);
            return true;
        }

        std::vector<Scalar> J(n_ * n_);
        jac_(x.data(), J.data());
        for (unsigned int c = 0; c < columns; ++c)
            for (std::size_t i = 0; i < n_; ++i) {
                Scalar sum = Scalar(0);
                for (std::size_t j = 0; j < n_; ++j)
                    sum += J[i + n_ * j] * directions[(std::size_t)c * n_ + j];
                result[(std::size_t)c * n_ + i] = sum;
            }
        return true;
    }

  private:
    std::size_t n_;
    ODE f_;
    Jacobian jac_;
    SparsityPattern pattern_;
    bool pattern_valid_ = false;
    JacobianVector jv_;
    TimeDeriv ft_;
    bool has_time_ = false;
    ParamDeriv fp_;
    std::size_t n_param_ = 0;
    HessianVector hv_;
    SecondOrderDir sod_;
    OrderKDirectional okd_;
    mutable std::vector<Scalar> point_; // last Forward(0) argument (stateful point)
};

} // namespace yabi
