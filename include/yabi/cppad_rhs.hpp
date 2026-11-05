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
