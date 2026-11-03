#pragma once

#include <cppad/cppad.hpp>

#include <cstddef>
#include <vector>

#include <yabi/abstract_rhs.hpp>

namespace yabi {

// Concrete AbstractRHS adapter over a CppAD::ADFun tape.  Non-owning by
// default (wraps a user ADFun); owns the tape when constructed from a raw
// pointer (used by clone()).
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

  private:
    CppAD::ADFun<Scalar> *fun_;
    bool own_;
};

} // namespace yabi
