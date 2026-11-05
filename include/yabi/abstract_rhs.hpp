#pragma once

#include <cstddef>
#include <vector>

#include <yabi/sparsity_pattern.hpp>

namespace yabi {

// Type-erased right-hand-side / AD-tape interface.  The integrator depends on
// this, not on CppAD; CppAD is one concrete adapter (see cppad_rhs.hpp).
//
//   Forward(p, x)  returns the p-th Taylor coefficient of f at direction x:
//                  p=0 -> f(x), p=1 -> J(x)*dx (directional derivative),
//                  p=2 -> the exact second-order coefficient (Hessian-in-direction).
//   Reverse(p, w)  returns the p-th order adjoint: w has Range()*p entries and
//                  the result has Domain()*p entries.
//   clone()        returns an owned deep copy (used by the replay tape pool).
//
// The sparsity hooks are optional.  They let a backend expose the state
// Jacobian pattern and evaluate only its structural entries, so the integrator
// can keep J/P in CSC and avoid n dense directional sweeps.  A backend that
// does not implement them makes the integrator fall back to the dense path
// unchanged.
template <typename Scalar> class AbstractRHS {
  public:
    virtual ~AbstractRHS() {}
    virtual std::size_t Domain() const = 0;
    virtual std::size_t Range() const = 0;
    virtual std::vector<Scalar> Forward(std::size_t p, const std::vector<Scalar> &x) const = 0;
    virtual std::vector<Scalar> Reverse(std::size_t p, const std::vector<Scalar> &w) const = 0;
    virtual std::size_t size_order() const = 0;
    virtual void capacity_order(std::size_t n) = 0;
    virtual AbstractRHS *clone() const = 0;

    // Sparsity of the state Jacobian: the (Range x Range) block df_i/dy_j,
    // with i,j < Range().  x in sparseJacobian() is the full Domain() argument
    // (state, time, parameters), exactly as passed to Forward().
    virtual bool hasSparsityPattern() const { return false; }
    virtual bool jacobianSparsity(SparsityPattern &pattern) const {
        (void)pattern;
        return false;
    }
    virtual bool sparseJacobian(const std::vector<Scalar> &x, const SparsityPattern &pattern,
                                std::vector<Scalar> &values) const {
        (void)x;
        (void)pattern;
        (void)values;
        return false;
    }
};

} // namespace yabi
