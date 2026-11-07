#pragma once

#include <cppad/cppad.hpp>

#include <algorithm>
#include <cstddef>
#include <set>
#include <vector>

#include <yabi/abstract_rhs.hpp>
#include <yabi/sparsity_pattern.hpp>

namespace yabi {

// Concrete AbstractRHS adapter over a CppAD::ADFun tape.  Non-owning by
// default (wraps a user ADFun); owns the tape when constructed from a raw
// pointer (used by clone()).
//
// When CppAD provides the sparsity hooks, the state Jacobian pattern is
// detected once with forward sparsity propagation and the values are evaluated
// with SparseJacobianForward.  CppAD colours the pattern internally, so the
// cost is proportional to the number of colours (typically the bandwidth),
// not to the number of state variables.
template <typename Scalar> class CppADRHS : public AbstractRHS<Scalar> {
  public:
    explicit CppADRHS(CppAD::ADFun<Scalar> &fun) : fun_(&fun), own_(false) {}
    explicit CppADRHS(CppAD::ADFun<Scalar> *fun) : fun_(fun), own_(true) {}
    ~CppADRHS() override {
        if (own_)
            delete fun_;
    }

    std::size_t Domain() const override { return fun_->Domain(); }
    std::size_t Range() const override { return fun_->Range(); }

    std::vector<Scalar> Forward(std::size_t p, const std::vector<Scalar> &x) const override {
        CppAD::vector<Scalar> xv(x.size());
        for (std::size_t i = 0; i < x.size(); ++i)
            xv[i] = x[i];
        CppAD::vector<Scalar> yv = fun_->Forward(p, xv);
        return std::vector<Scalar>(yv.begin(), yv.end());
    }

    std::vector<Scalar> Reverse(std::size_t p, const std::vector<Scalar> &w) const override {
        CppAD::vector<Scalar> wv(w.size());
        for (std::size_t i = 0; i < w.size(); ++i)
            wv[i] = w[i];
        CppAD::vector<Scalar> dwv = fun_->Reverse(p, wv);
        return std::vector<Scalar>(dwv.begin(), dwv.end());
    }

    std::size_t size_order() const override { return fun_->size_order(); }
    void capacity_order(std::size_t n) override { fun_->capacity_order(n); }

    bool supports_taylor() const override { return true; }

    bool adjointProduct(const std::vector<Scalar> &x, const std::vector<Scalar> &lambda,
                        std::vector<Scalar> &result) const override {
        if (x.size() != fun_->Domain() || lambda.size() != fun_->Range())
            return false;
        CppAD::vector<Scalar> xv(x.size());
        for (std::size_t i = 0; i < x.size(); ++i)
            xv[i] = x[i];
        fun_->Forward(0, xv); // set the point (state, time, parameters)
        CppAD::vector<Scalar> lamv(lambda.size());
        for (std::size_t i = 0; i < lambda.size(); ++i)
            lamv[i] = lambda[i];
        CppAD::vector<Scalar> dwv = fun_->Reverse(1, lamv); // (J^T lambda, d f/dt . lambda, d f/dp^T lambda)
        if (dwv.size() < fun_->Range()) {
            result.clear();
            return false;
        }
        result.assign(dwv.begin(), dwv.begin() + fun_->Range());
        return true;
    }

    bool adjointParameterProduct(const std::vector<Scalar> &x, const std::vector<Scalar> &lambda,
                                 std::vector<Scalar> &result) const override {
        // Parameter block is (Domain - Range - 1) entries (present only for
        // RHS_type 2, i.e. time + parameter).
        const std::size_t p = fun_->Domain() - fun_->Range() - 1;
        if (p == 0 || x.size() != fun_->Domain() || lambda.size() != fun_->Range())
            return false;
        CppAD::vector<Scalar> xv(x.size());
        for (std::size_t i = 0; i < x.size(); ++i)
            xv[i] = x[i];
        fun_->Forward(0, xv);
        CppAD::vector<Scalar> lamv(lambda.size());
        for (std::size_t i = 0; i < lambda.size(); ++i)
            lamv[i] = lambda[i];
        CppAD::vector<Scalar> dwv = fun_->Reverse(1, lamv); // (J^T lam, df/dt.lam, df/dp^T lam)
        if (dwv.size() < fun_->Domain()) {
            result.clear();
            return false;
        }
        const std::size_t offset = fun_->Range() + 1; // skip state + time
        result.assign(dwv.begin() + offset, dwv.end());
        return true;
    }

    bool taylorForward(std::size_t p, const std::vector<Scalar> &x,
                       std::vector<Scalar> &y) const override {
        if (x.size() != fun_->Domain() * (p + 1))
            return false;
        CppAD::vector<Scalar> xv(x.size());
        for (std::size_t i = 0; i < x.size(); ++i)
            xv[i] = x[i];
        CppAD::vector<Scalar> yv = fun_->Forward(p, xv);
        if (yv.size() != fun_->Range() * (p + 1)) {
            y.clear();
            return false;
        }
        y.assign(yv.begin(), yv.end());
        return true;
    }

    AbstractRHS<Scalar> *clone() const override {
        CppAD::ADFun<Scalar> *copy = new CppAD::ADFun<Scalar>();
        *copy = *fun_;
        return new CppADRHS<Scalar>(copy);
    }

    bool hasSparsityPattern() const override { return true; }

    bool jacobianSparsity(SparsityPattern &pattern) const override {
        const std::size_t domain = fun_->Domain();
        const std::size_t range = fun_->Range();

        if (domain == 0 || range == 0)
            return false;

        // R = I_domain, transposed detection: s[j] is the set of rows i for
        // which J(i,j) may be non-zero, i.e. exactly a CSC column.
        CppAD::vector<std::set<std::size_t>> r(domain);
        for (std::size_t j = 0; j < domain; ++j)
            r[j].insert(j);
        CppAD::vector<std::set<std::size_t>> s =
            fun_->ForSparseJac(domain, r, /*transpose=*/true);

        const std::size_t state = range; // yabi state size == RHS range
        pattern.rows = state;
        pattern.cols = state;
        pattern.column_rows.assign(state, {});

        // Row-oriented pattern over the full domain, for SparseJacobianForward.
        sparse_row_pattern_.resize(range);
        for (std::size_t i = 0; i < range; ++i)
            sparse_row_pattern_[i].clear();

        for (std::size_t j = 0; j < domain; ++j)
            for (std::size_t i : s[j]) {
                if (i >= range)
                    continue;
                sparse_row_pattern_[i].insert(j);
                if (j < state) {
                    if (i < state)
                        pattern.column_rows[j].push_back(i);
                }
            }

        // Flatten the state block in column-major order: the same order as the
        // values returned by SparseJacobianForward and as CscMatrix::val.
        sparse_row_.clear();
        sparse_col_.clear();
        for (std::size_t j = 0; j < state; ++j)
            for (std::size_t i : pattern.column_rows[j]) {
                sparse_row_.push_back(i);
                sparse_col_.push_back(j);
            }

        sparse_pattern_ = pattern;
        sparse_ready_ = true;

        // The colouring cache is only valid for one pattern/row/col set.
        sparse_work_.clear();

        return true;
    }

    bool jacobianVectorProduct(const std::vector<Scalar> &x, const std::vector<Scalar> &v,
                               std::vector<Scalar> &result) const override {
        const std::size_t n = fun_->Domain();
        const std::size_t m = fun_->Range();
        if (x.size() != n || v.size() != n)
            return false;
        // One CppAD call.  For q=1 and a full xq, CppAD's layout is
        // interleaved per variable/component: xq[2*j] = x_j, xq[2*j+1] = v_j,
        // and the result returns yq[2*i] = f_i, yq[2*i+1] = (J(x)*v)_i.
        CppAD::vector<Scalar> xv(2 * n);
        for (std::size_t i = 0; i < n; ++i) {
            xv[2 * i] = x[i];
            xv[2 * i + 1] = v[i];
        }
        CppAD::vector<Scalar> yv = fun_->Forward(1, xv);
        if (yv.size() != 2 * m)
            return false;
        result.resize(m);
        for (std::size_t i = 0; i < m; ++i)
            result[i] = yv[2 * i + 1];
        return true;
    }

    // Batched product with one CppAD multi-direction forward sweep.  For a
    // single column the historical interleaved Forward(1) path is used so the
    // integrator's single-direction usage is unchanged.
    bool jacobianMatrixProduct(const std::vector<Scalar> &x,
                               const std::vector<Scalar> &directions, unsigned int columns,
                               std::vector<Scalar> &result) const override {
        const std::size_t n = fun_->Domain();
        const std::size_t m = fun_->Range();
        if (x.size() != n)
            return false;
        if (columns == 0) {
            result.clear();
            return directions.empty();
        }
        if (directions.size() != n * (std::size_t)columns)
            return false;

        if (columns == 1) {
            std::vector<Scalar> direction(directions.begin(), directions.end());
            return jacobianVectorProduct(x, direction, result);
        }

        CppAD::vector<Scalar> xv(n);
        for (std::size_t i = 0; i < n; ++i)
            xv[i] = x[i];
        fun_->Forward(0, xv);

        // Layout: xq[columns*j + c] is direction c of variable j.
        CppAD::vector<Scalar> xq(n * (std::size_t)columns);
        for (std::size_t j = 0; j < n; ++j)
            for (unsigned int c = 0; c < columns; ++c)
                xq[(std::size_t)columns * j + c] = directions[(std::size_t)c * n + j];

        CppAD::vector<Scalar> yq = fun_->Forward(1, (std::size_t)columns, xq);
        if (yq.size() != m * (std::size_t)columns)
            return false;

        // Layout: yq[columns*i + c] is component i of direction c.
        result.assign(m * (std::size_t)columns, Scalar(0));
        for (std::size_t i = 0; i < m; ++i)
            for (unsigned int c = 0; c < columns; ++c)
                result[(std::size_t)c * m + i] = yq[(std::size_t)columns * i + c];
        return true;
    }

    bool sparseJacobian(const std::vector<Scalar> &x, const SparsityPattern &pattern,
                        std::vector<Scalar> &values) const override {
        if (!sparse_ready_ || sparse_pattern_ != pattern) {
            SparsityPattern detected;
            if (!jacobianSparsity(detected) || detected != pattern)
                return false;
        }

        values.assign(sparse_row_.size(), (Scalar)0.0);
        if (values.empty())
            return true;

        CppAD::vector<Scalar> xv(x.size());
        for (std::size_t i = 0; i < x.size(); ++i)
            xv[i] = x[i];

        CppAD::vector<Scalar> jac(values.size());
        fun_->SparseJacobianForward(xv, sparse_row_pattern_, sparse_row_, sparse_col_, jac,
                                    sparse_work_);
        for (std::size_t k = 0; k < values.size(); ++k)
            values[k] = jac[k];
        return true;
    }

  private:
    CppAD::ADFun<Scalar> *fun_;
    bool own_;

    // Detection/colouring cache.  The patterns depend only on the operation
    // sequence, which is fixed for an ADFun, so they are computed once.
    mutable bool sparse_ready_ = false;
    mutable SparsityPattern sparse_pattern_;
    mutable CppAD::vector<std::set<std::size_t>> sparse_row_pattern_;
    mutable std::vector<std::size_t> sparse_row_;
    mutable std::vector<std::size_t> sparse_col_;
    mutable CppAD::sparse_jacobian_work sparse_work_;
};

} // namespace yabi
