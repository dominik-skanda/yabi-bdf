/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// Unit tests for the AbstractRHS *default* method implementations: the
// jacobianVectorProduct default (Forward(0) then Forward(1)), the batched
// jacobianMatrixProduct default (one jacobianVectorProduct per column), and the
// "return false" stubs for the hooks a backend does not override.

#include "test_support.hpp"

#include <yabi/abstract_rhs.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace {

// A minimal linear RHS f(y) = A y with A = [[1,2],[3,4]] (column-major).
// Forward(0) = A*x, Forward(1) = A*dx (so the Jacobian is the constant A).
class LinearRHS : public yabi::AbstractRHS<double> {
  public:
    std::size_t Domain() const override { return 2; }
    std::size_t Range() const override { return 2; }

    std::vector<double> Forward(std::size_t p, const std::vector<double> &x) const override {
        if (p == 0) {
            return {1.0 * x[0] + 2.0 * x[1], 3.0 * x[0] + 4.0 * x[1]};
        }
        if (p == 1) {
            return {1.0 * x[0] + 2.0 * x[1], 3.0 * x[0] + 4.0 * x[1]};
        }
        return {};
    }

    std::vector<double> Reverse(std::size_t, const std::vector<double> &) const override {
        return {};
    }
    std::size_t size_order() const override { return 1; }
    void capacity_order(std::size_t) override {}
    AbstractRHS<double> *clone() const override { return new LinearRHS(); }
};

} // namespace

int main() {
    LinearRHS rhs;

    // --- default jacobianVectorProduct == Forward(0) + Forward(1) -----------
    {
        std::vector<double> x = {2.0, 3.0}, v = {1.0, 0.0}, result;
        yabi_test::check(rhs.jacobianVectorProduct(x, v, result), "jacobianVectorProduct succeeds");
        // A * v = (1*1 + 2*0, 3*1 + 4*0) = (1, 3)
        yabi_test::check(result.size() == 2 && yabi_test::near(result[0], 1.0) &&
                             yabi_test::near(result[1], 3.0),
                         "jacobianVectorProduct = A v");
    }

    // --- default jacobianMatrixProduct = one J·v per column -----------------
    {
        std::vector<double> x = {0.0, 0.0};
        // two directions e0, e1 (column-major: dir[2*c + i])
        std::vector<double> directions = {1.0, 0.0, 0.0, 1.0};
        std::vector<double> result;
        yabi_test::check(rhs.jacobianMatrixProduct(x, directions, 2, result),
                         "jacobianMatrixProduct succeeds");
        // columns: A e0 = (1,3), A e1 = (2,4) -> result = {1,3,2,4}
        yabi_test::check(result.size() == 4 && yabi_test::near(result[0], 1.0) &&
                             yabi_test::near(result[1], 3.0) && yabi_test::near(result[2], 2.0) &&
                             yabi_test::near(result[3], 4.0),
                         "jacobianMatrixProduct = A V");
    }

    // --- default jacobianVectorProduct rejects wrong-size inputs ------------
    {
        std::vector<double> result;
        yabi_test::check(!rhs.jacobianVectorProduct({1.0}, {1.0, 0.0}, result),
                         "jacobianVectorProduct rejects wrong-size x");
        yabi_test::check(!rhs.jacobianVectorProduct({1.0, 0.0}, {1.0}, result),
                         "jacobianVectorProduct rejects wrong-size v");
    }

    // --- every un-overridden hook returns false -----------------------------
    {
        std::vector<double> x = {0.0, 0.0}, v = {1.0, 0.0}, lambda = {1.0, 0.0}, result;
        std::vector<std::vector<double>> S = {{1.0, 0.0}, {0.0, 1.0}};
        yabi::SparsityPattern pattern;

        yabi_test::check(!rhs.supports_taylor(), "supports_taylor defaults to false");
        yabi_test::check(!rhs.taylorForward(1, x, result), "taylorForward defaults to false");
        yabi_test::check(!rhs.adjointProduct(x, lambda, result), "adjointProduct defaults to false");
        yabi_test::check(!rhs.adjointParameterProduct(x, lambda, result),
                         "adjointParameterProduct defaults to false");
        yabi_test::check(!rhs.hessianVectorProduct(x, lambda, v, result),
                         "hessianVectorProduct defaults to false");
        yabi_test::check(!rhs.secondOrderDirectional(x, v, result),
                         "secondOrderDirectional defaults to false");
        yabi_test::check(!rhs.orderKDirectional(2, x, S, result),
                         "orderKDirectional defaults to false");
        yabi_test::check(!rhs.hasSparsityPattern(), "hasSparsityPattern defaults to false");
        yabi_test::check(!rhs.jacobianSparsity(pattern), "jacobianSparsity defaults to false");
        yabi_test::check(!rhs.sparseJacobian(x, pattern, result),
                         "sparseJacobian defaults to false");
    }

    return yabi_test::finish("AbstractRHS default methods");
}
