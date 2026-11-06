/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

#ifndef YABI_MATRIX_FREE_OPERATOR_HPP
#define YABI_MATRIX_FREE_OPERATOR_HPP

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
};

} // namespace yabi

#endif
