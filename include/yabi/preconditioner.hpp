/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

#ifndef YABI_PRECONDITIONER_HPP
#define YABI_PRECONDITIONER_HPP

#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

#include <yabi/csc_matrix.hpp>
#include <yabi/matrix_free_operator.hpp>

namespace yabi {

// Application-supplied preconditioner for the matrix-free Krylov solve.
//
// The integrator intentionally does not implement ILU/ILUT, AMG, block or
// physics-based preconditioners.  This interface is the seam through which a
// user adapts an external package (AMGCL, hypre, Trilinos, a hand-written
// block solver, ...) without touching the Newton or Krylov code.
//
// Two modes are supported:
//
//   * requiresAssembledMatrix() == true
//       The Newton solver assembles the scaled CSC Newton matrix P and passes
//       it to update().  The built-in Jacobi/Direct preconditioners are in
//       this category.  A sparse ILU/AMG adapter normally belongs here.
//
//   * requiresAssembledMatrix() == false
//       The Newton solver may remain fully matrix-free: no sparsity detection,
//       no CSC J/P assembly.  update() receives a null P and the current
//       scaled P*x operator, which a J-free/block/operator-based
//       preconditioner can query directly.
//
// `apply()` is called inside the Krylov iteration and must evaluate
// z = M^{-1} r for the n-vector r.  It may use mutable workspace but must not
// mutate the operator or matrix it was given (they are borrowed).
template <typename Scalar> class Preconditioner {
  public:
    virtual ~Preconditioner() {}

    // True when update() needs the assembled scaled CSC Newton matrix.
    virtual bool requiresAssembledMatrix() const = 0;

    // Prepare/reuse the preconditioner for the current Newton matrix.
    //   * P        scaled CSC P = D1*(I - (h/l1)*J)*D2, or null on the J-free path
    //   * op       scaled P*x operator (non-owning, valid until the next update)
    //   * diagonal scaled Newton diagonal when assembled, otherwise empty
    //   * n        number of state unknowns
    // Called once per Jacobian/P refresh and before every Krylov solve.
    virtual bool update(const CscMatrix<Scalar> *P, const MatrixFreeOperator<Scalar> *op,
                        const std::vector<Scalar> &diagonal, unsigned int n) = 0;

    // z = M^{-1} r.  Both arrays have n entries and may alias.
    virtual bool apply(const Scalar *r, Scalar *z) = 0;
};

// std::function adapter for one-off/user-provided preconditioners.  The update
// callback may be empty, in which case update() succeeds and the apply callback
// is used as-is.
template <typename Scalar> class FunctionPreconditioner : public Preconditioner<Scalar> {
  public:
    using Update = std::function<bool(const CscMatrix<Scalar> *,
                                      const MatrixFreeOperator<Scalar> *,
                                      const std::vector<Scalar> &, unsigned int)>;
    using Apply = std::function<bool(const Scalar *, Scalar *, unsigned int)>;

    FunctionPreconditioner(bool requires_assembled_matrix, Apply apply)
        : requires_assembled_matrix_(requires_assembled_matrix), apply_(std::move(apply)) {}

    FunctionPreconditioner(bool requires_assembled_matrix, Update update, Apply apply)
        : requires_assembled_matrix_(requires_assembled_matrix), update_(std::move(update)),
          apply_(std::move(apply)) {}

    bool requiresAssembledMatrix() const override { return requires_assembled_matrix_; }

    bool update(const CscMatrix<Scalar> *P, const MatrixFreeOperator<Scalar> *op,
                const std::vector<Scalar> &diagonal, unsigned int n) override {
        n_ = n;
        return update_ ? update_(P, op, diagonal, n) : true;
    }

    bool apply(const Scalar *r, Scalar *z) override {
        return apply_ ? apply_(r, z, n_) : false;
    }

  private:
    bool requires_assembled_matrix_;
    Update update_;
    Apply apply_;
    unsigned int n_ = 0;
};

} // namespace yabi

#endif
