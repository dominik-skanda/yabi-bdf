/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

#ifndef YABI_MATRIX_FREE_OPERATOR_HPP
#define YABI_MATRIX_FREE_OPERATOR_HPP

#include <cstddef>

namespace yabi {

// A matrix-free representation of the Newton matrix P = I - (h/l1)*J, already
// scaled by the Newton row/column scalings D1 and D2.
//
// The operator is deliberately independent of CppAD: a backend may implement
// apply() with an AD directional derivative, an analytic Jacobian, or any other
// matrix-free product.  Krylov solvers use it instead of a stored matrix.
template <typename Scalar> class MatrixFreeOperator {
  public:
    virtual ~MatrixFreeOperator() {
    }

    // y = P x for the current Newton matrix.  x and y both have n entries.
    // Return false on an evaluation failure (e.g. a non-finite AD result).
    virtual bool apply(const Scalar *x, Scalar *y) const = 0;

    // Batched y = P X for `columns` column-major vectors of length n.  The
    // default is an exact column loop; a block/GPU/sparse backend can override
    // it with one kernel.  `x` and `y` have n*columns entries.
    virtual bool applyBlock(const Scalar *x, unsigned int n, unsigned int columns,
                            Scalar *y) const {
        if (n == 0 && columns != 0)
            return false;
        for (unsigned int c = 0; c < columns; ++c)
            if (!apply(x + (std::size_t)c * n, y + (std::size_t)c * n))
                return false;
        return true;
    }
};

} // namespace yabi

#endif
