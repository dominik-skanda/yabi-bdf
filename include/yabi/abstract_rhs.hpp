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

    // Matrix-free Jacobian-vector product J(x)*v.  x is the full Domain()
    // argument and v has Domain() entries (state, then time, then parameters).
    // The default implementation is stateful: it sets the point with
    // Forward(0, x) and then evaluates Forward(1, v), which is exactly the
    // existing AbstractRHS contract.  Backends may override it with a
    // one-evaluation implementation.
    virtual bool jacobianVectorProduct(const std::vector<Scalar> &x,
                                       const std::vector<Scalar> &v,
                                       std::vector<Scalar> &result) const {
        if (v.size() != Domain() || x.size() != Domain())
            return false;
        Forward(0, x);
        result = Forward(1, v);
        return result.size() == Range();
    }

    // Batched Jacobian-matrix product J(x)*V.  `directions` and `result` are
    // column-major Domain() x columns and Range() x columns respectively, so
    // they match the dense P/J layout already used by the integrator.  The
    // default is exact and calls jacobianVectorProduct() once per column; a
    // sparse/block/matrix-free backend may override it with a batched product.
    // This is the extension point for future sparse/matrix-free sensitivity
    // linearizations; the replay engines still use explicit J/H today.
    virtual bool jacobianMatrixProduct(const std::vector<Scalar> &x,
                                       const std::vector<Scalar> &directions,
                                       unsigned int columns,
                                       std::vector<Scalar> &result) const {
        const std::size_t domain = Domain();
        const std::size_t range = Range();
        if (x.size() != domain)
            return false;
        if (columns == 0) {
            result.clear();
            return directions.empty();
        }
        if (directions.size() != domain * (std::size_t)columns)
            return false;

        result.assign(range * (std::size_t)columns, Scalar(0));
        std::vector<Scalar> direction(domain);
        std::vector<Scalar> product;
        for (unsigned int c = 0; c < columns; ++c) {
            for (std::size_t i = 0; i < domain; ++i)
                direction[i] = directions[(std::size_t)c * domain + i];
            if (!jacobianVectorProduct(x, direction, product) || product.size() != range)
                return false;
            for (std::size_t i = 0; i < range; ++i)
                result[(std::size_t)c * range + i] = product[i];
        }
        return true;
    }
};

} // namespace yabi
