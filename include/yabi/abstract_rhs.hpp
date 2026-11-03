#pragma once

#include <cstddef>
#include <vector>

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
};

} // namespace yabi
