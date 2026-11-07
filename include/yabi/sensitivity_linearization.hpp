/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

#ifndef YABI_SENSITIVITY_LINEARIZATION_HPP
#define YABI_SENSITIVITY_LINEARIZATION_HPP

#include <cstddef>

#include <yabi/matrix_free_operator.hpp>

namespace yabi {

// Extension point for a sensitivity linearization whose state operator is
// sparse, block or matrix-free rather than an explicitly stored dense J/H.
//
// The existing forward/simultaneous/adjoint engines still record and replay an
// explicit P/ip and explicit J/H; this interface is deliberately not wired into
// them yet.  It documents the contract a future sparse/matrix-free sensitivity
// solve implements, so that work does not have to reopen the integrator API.
//
// Layout is column-major, matching the integrator's P/J storage:
//   X and Y are rows() x columns, X[c*rows()+i].
template <typename Scalar> class SensitivityLinearization {
  public:
    virtual ~SensitivityLinearization() {}

    // Number of state rows in X/Y.
    virtual unsigned int rows() const = 0;

    // Y = L * X for `columns` sensitivity directions.  Return false on an
    // evaluation failure; Y is then unspecified.
    virtual bool apply(const Scalar *X, unsigned int columns, Scalar *Y) const = 0;
};

// Adapter from the matrix-free Newton operator to the sensitivity contract.
// The operator is borrowed and must outlive this object and every apply().
template <typename Scalar>
class MatrixFreeSensitivityLinearization : public SensitivityLinearization<Scalar> {
  public:
    MatrixFreeSensitivityLinearization(const MatrixFreeOperator<Scalar> *op, unsigned int rows)
        : op_(op), rows_(rows) {}

    unsigned int rows() const override { return rows_; }

    bool apply(const Scalar *X, unsigned int columns, Scalar *Y) const override {
        if (!op_)
            return false;
        return op_->applyBlock(X, rows_, columns, Y);
    }

  private:
    const MatrixFreeOperator<Scalar> *op_;
    unsigned int rows_;
};

} // namespace yabi

#endif
