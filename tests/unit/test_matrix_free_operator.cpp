/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// Unit test for the MatrixFreeOperator default applyBlock (batched P·X), which
// loops one apply() per column, plus its edge cases.

#include "test_support.hpp"

#include <yabi/matrix_free_operator.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace {

// y = A x with A = [[1,2],[3,4]] (the operator is a constant linear map).
class MockOperator : public yabi::MatrixFreeOperator<double> {
  public:
    bool apply(const double *x, double *y) const override {
        y[0] = 1.0 * x[0] + 2.0 * x[1];
        y[1] = 3.0 * x[0] + 4.0 * x[1];
        return true;
    }
};

} // namespace

int main() {
    MockOperator op;

    // --- applyBlock batches columns of A ------------------------------------
    {
        double x[4] = {1.0, 0.0, 0.0, 1.0}; // columns e0, e1
        double y[4] = {0.0, 0.0, 0.0, 0.0};
        yabi_test::check(op.applyBlock(x, 2, 2, y), "applyBlock succeeds");
        // A e0 = (1,3), A e1 = (2,4) -> y = {1,3,2,4}
        yabi_test::check(yabi_test::near(y[0], 1.0) && yabi_test::near(y[1], 3.0) &&
                             yabi_test::near(y[2], 2.0) && yabi_test::near(y[3], 4.0),
                         "applyBlock applies A per column");
    }

    // --- edge cases ----------------------------------------------------------
    {
        double x[4] = {0.0, 0.0, 0.0, 0.0}, y[4] = {0.0, 0.0, 0.0, 0.0};
        yabi_test::check(!op.applyBlock(x, 0, 2, y), "applyBlock rejects n==0 with columns!=0");
        yabi_test::check(op.applyBlock(x, 2, 0, y), "applyBlock accepts columns==0");
    }

    return yabi_test::finish("MatrixFreeOperator unit tests");
}
