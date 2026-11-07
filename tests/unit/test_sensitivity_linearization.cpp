/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// T32d — batched Jacobian products and sensitivity-linearization interfaces.

#include "test_support.hpp"

#include <yabi/abstract_rhs.hpp>
#include <yabi/matrix_free_operator.hpp>
#include <yabi/sensitivity_linearization.hpp>
#include <yabi/user_ode_rhs.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

using Vec = std::vector<double>;

// Minimal AbstractRHS used to exercise the default jacobianMatrixProduct()
// column loop (no override).
class DefaultBatchRHS : public yabi::AbstractRHS<double> {
  public:
    std::size_t Domain() const override { return 2; }
    std::size_t Range() const override { return 2; }

    std::vector<double> Forward(std::size_t p, const std::vector<double> &x) const override {
        if (p == 0) {
            point_ = x;
            return {x[0] * x[0] + x[1], x[0] - 3.0 * x[1]};
        }
        if (p == 1) {
            ++derivative_calls;
            return {2.0 * point_[0] * x[0] + x[1], x[0] - 3.0 * x[1]};
        }
        throw std::runtime_error("DefaultBatchRHS: unsupported order");
    }

    std::vector<double> Reverse(std::size_t, const std::vector<double> &) const override {
        throw std::runtime_error("DefaultBatchRHS: no reverse");
    }
    std::size_t size_order() const override { return 1; }
    void capacity_order(std::size_t) override {}
    yabi::AbstractRHS<double> *clone() const override { return new DefaultBatchRHS(*this); }

    mutable unsigned int derivative_calls = 0;

  private:
    mutable std::vector<double> point_;
};

// y' = f(y) with an analytic J, used for the pure batch contract.
void ode(const double *x, double *f) {
    f[0] = -x[0] + x[1] * x[2];
    f[1] = x[0] - 2.0 * x[1];
    f[2] = x[0] * x[1] - x[2];
}

void jacobian(const double *x, double *J) {
    const std::size_t n = 3;
    J[0 + n * 0] = -1.0;
    J[1 + n * 0] = x[2];
    J[2 + n * 0] = x[1];

    J[0 + n * 1] = 1.0;
    J[1 + n * 1] = -2.0;
    J[2 + n * 1] = x[0];

    J[0 + n * 2] = 0.0;
    J[1 + n * 2] = 0.0;
    J[2 + n * 2] = -1.0;
}

void explicitProduct(const Vec &x, const Vec &directions, unsigned int columns, Vec &expected) {
    const std::size_t n = 3;
    Vec J(n * n, 0.0);
    jacobian(x.data(), J.data());
    expected.assign(n * columns, 0.0);
    for (unsigned int c = 0; c < columns; ++c)
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                expected[(std::size_t)c * n + i] +=
                    J[i + n * j] * directions[(std::size_t)c * n + j];
}

class DenseOperator : public yabi::MatrixFreeOperator<double> {
  public:
    explicit DenseOperator(std::vector<double> matrix) : matrix_(std::move(matrix)) {
    }

    bool apply(const double *x, double *y) const override {
        const std::size_t n = (std::size_t)std::sqrt((double)matrix_.size());
        for (std::size_t i = 0; i < n; ++i) {
            double sum = 0.0;
            for (std::size_t j = 0; j < n; ++j)
                sum += matrix_[j * n + i] * x[j];
            y[i] = sum;
        }
        return true;
    }

  private:
    std::vector<double> matrix_;
};

void testDefaultBatchLoop() {
    DefaultBatchRHS rhs;
    Vec x = {1.5, -0.5};
    Vec directions = {1.0, 0.0, 0.0, 1.0, 2.0, -1.0};

    Vec result;
    yabi_test::check(rhs.jacobianMatrixProduct(x, directions, 3, result),
                     "default batch product succeeds");
    yabi_test::check(rhs.derivative_calls == 3,
                     "default batch product calls the directional product per column");
    yabi_test::check(result.size() == 6, "default batch product returns columns*range entries");

    const double expected[] = {2.0 * 1.5, 1.0, 1.0, -3.0, 2.0 * 1.5 * 2.0 - 1.0, 2.0 + 3.0};
    double max_diff = 0.0;
    for (std::size_t i = 0; i < 6; ++i)
        max_diff = std::max(max_diff, std::fabs(result[i] - expected[i]));
    yabi_test::check(max_diff < 1e-15, "default batch product matches explicit J*V");

    Vec empty;
    yabi_test::check(rhs.jacobianMatrixProduct(x, empty, 0, empty),
                     "zero columns is a valid empty batch");
    yabi_test::check(empty.empty(), "zero columns returns an empty result");
}

void testUserODERHSBatch() {
    const std::size_t n = 3;
    Vec x = {1.5, -0.5, 2.0};
    Vec directions = {0.25, -1.0, 0.75, 1.0, 2.0, -0.5, -2.0, 0.5, 1.5};

    Vec expected;
    explicitProduct(x, directions, 3, expected);

    unsigned int jacobian_calls = 0;
    yabi::UserODERHS<double> rhs(n, ode,
                                 [&](const double *point, double *J) {
                                     ++jacobian_calls;
                                     jacobian(point, J);
                                 });

    Vec result;
    yabi_test::check(rhs.jacobianMatrixProduct(x, directions, 3, result),
                     "UserODERHS batch product succeeds");
    yabi_test::check(jacobian_calls == 1,
                     "UserODERHS evaluates the analytic Jacobian once for a batch");
    double max_diff = 0.0;
    for (std::size_t i = 0; i < result.size(); ++i)
        max_diff = std::max(max_diff, std::fabs(result[i] - expected[i]));
    yabi_test::check(max_diff < 1e-15, "UserODERHS batch matches explicit J*V");

    // With the J*v hook the batch delegates once per direction.
    unsigned int hook_calls = 0;
    yabi::UserODERHS<double> hooked(n, ode, jacobian);
    hooked.setJacobianVectorProduct([&](const double *point, const double *v, double *out) {
        ++hook_calls;
        Vec J(n * n, 0.0);
        jacobian(point, J.data());
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = 0.0;
            for (std::size_t j = 0; j < n; ++j)
                out[i] += J[i + n * j] * v[j];
        }
    });

    Vec hooked_result;
    yabi_test::check(hooked.jacobianMatrixProduct(x, directions, 3, hooked_result),
                     "UserODERHS batch with J*v hook succeeds");
    yabi_test::check(hook_calls == 3, "the J*v hook is called once per column");
    max_diff = 0.0;
    for (std::size_t i = 0; i < hooked_result.size(); ++i)
        max_diff = std::max(max_diff, std::fabs(hooked_result[i] - expected[i]));
    yabi_test::check(max_diff < 1e-15, "J*v batch matches explicit J*V");

    // Validation.
    Vec bad_x = {1.0, 2.0};
    yabi_test::check(!rhs.jacobianMatrixProduct(bad_x, directions, 3, result),
                     "batch rejects a wrong-size point");
    Vec bad_directions(5, 0.0);
    yabi_test::check(!rhs.jacobianMatrixProduct(x, bad_directions, 3, result),
                     "batch rejects a wrong-size direction block");
    Vec empty;
    yabi_test::check(rhs.jacobianMatrixProduct(x, empty, 0, result),
                     "UserODERHS accepts an empty batch");
    yabi_test::check(result.empty(), "UserODERHS empty batch returns an empty result");
    yabi_test::check(!rhs.jacobianMatrixProduct(bad_x, empty, 0, result),
                     "empty batch still validates the point");
}

void testMatrixFreeBlockProduct() {
    DenseOperator op({2.0, -1.0, 0.5, 3.0});
    const double X[] = {1.0, -2.0, 0.25, 4.0};
    double Y[4] = {0.0, 0.0, 0.0, 0.0};

    yabi_test::check(op.applyBlock(X, 2, 2, Y), "default block product succeeds");
    double expected[4] = {0.0, 0.0, 0.0, 0.0};
    for (unsigned int c = 0; c < 2; ++c)
        op.apply(X + 2 * c, expected + 2 * c);

    double max_diff = 0.0;
    for (std::size_t i = 0; i < 4; ++i)
        max_diff = std::max(max_diff, std::fabs(Y[i] - expected[i]));
    yabi_test::check(max_diff < 1e-15, "default block product matches per-column apply");
    yabi_test::check(op.applyBlock(X, 2, 0, Y), "zero-column block product succeeds");
    yabi_test::check(!op.applyBlock(X, 0, 2, Y), "zero-size block product is rejected");
}

void testSensitivityAdapter() {
    DenseOperator op({2.0, -1.0, 0.5, 3.0});
    yabi::MatrixFreeSensitivityLinearization<double> linearization(&op, 2);
    yabi_test::check(linearization.rows() == 2, "sensitivity adapter reports its row count");

    const double X[] = {1.0, -2.0, 0.25, 4.0};
    double Y[4] = {0.0, 0.0, 0.0, 0.0};
    yabi_test::check(linearization.apply(X, 2, Y), "sensitivity adapter apply succeeds");
    double expected[4] = {0.0, 0.0, 0.0, 0.0};
    for (unsigned int c = 0; c < 2; ++c)
        op.apply(X + 2 * c, expected + 2 * c);

    double max_diff = 0.0;
    for (std::size_t i = 0; i < 4; ++i)
        max_diff = std::max(max_diff, std::fabs(Y[i] - expected[i]));
    yabi_test::check(max_diff < 1e-15, "sensitivity adapter matches the operator");
    yabi_test::check(linearization.apply(X, 0, Y), "sensitivity adapter accepts zero columns");

    yabi::MatrixFreeSensitivityLinearization<double> invalid(nullptr, 2);
    yabi_test::check(!invalid.apply(X, 2, Y), "a null operator is rejected");
    yabi_test::check(invalid.rows() == 2, "a null adapter still reports its configured rows");
}

void testUserODERHSCloneBatch() {
    yabi::UserODERHS<double> rhs(3, ode, jacobian);
    std::unique_ptr<yabi::AbstractRHS<double>> copy(rhs.clone());

    Vec x = {1.5, -0.5, 2.0};
    Vec directions = {0.25, -1.0, 0.75, 1.0, 2.0, -0.5};
    Vec expected;
    explicitProduct(x, directions, 2, expected);

    Vec result;
    yabi_test::check(copy->jacobianMatrixProduct(x, directions, 2, result),
                     "cloned UserODERHS supports the batch product");
    double max_diff = 0.0;
    for (std::size_t i = 0; i < result.size(); ++i)
        max_diff = std::max(max_diff, std::fabs(result[i] - expected[i]));
    yabi_test::check(max_diff < 1e-15, "cloned batch matches explicit J*V");
}

} // namespace

int main() {
    testDefaultBatchLoop();
    testUserODERHSBatch();
    testMatrixFreeBlockProduct();
    testSensitivityAdapter();
    testUserODERHSCloneBatch();
    return yabi_test::finish("sensitivity linearization unit tests");
}
