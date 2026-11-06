/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

#include "test_support.hpp"

#include <yabi/sparsity_pattern.hpp>
#include <yabi/user_ode_rhs.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

using Vec = std::vector<double>;

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

// Deliberately unsorted and duplicated: setSparsityPattern must normalise it.
yabi::SparsityPattern makePattern() {
    yabi::SparsityPattern pattern;
    pattern.rows = 3;
    pattern.cols = 3;
    pattern.column_rows = {{1, 0, 0}, {1}, {2, 0}};
    return pattern;
}

void testPatternValidation() {
    yabi::UserODERHS<double> rhs(3, ode, jacobian);
    yabi_test::check(!rhs.hasSparsityPattern(), "no pattern before setSparsityPattern");

    rhs.setSparsityPattern(makePattern());
    yabi_test::check(rhs.hasSparsityPattern(), "pattern present after setSparsityPattern");

    yabi::SparsityPattern out;
    yabi_test::check(rhs.jacobianSparsity(out), "jacobianSparsity succeeds");
    const std::vector<std::vector<std::size_t>> expected = {{0, 1}, {1}, {0, 2}};
    yabi_test::check(out.column_rows == expected, "pattern is sorted and de-duplicated");

    bool threw = false;
    try {
        yabi::SparsityPattern bad = makePattern();
        bad.rows = 2;
        yabi::UserODERHS<double> invalid(3, ode, jacobian);
        invalid.setSparsityPattern(bad);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    yabi_test::check(threw, "a non-square pattern is rejected");

    threw = false;
    try {
        yabi::SparsityPattern bad = makePattern();
        bad.column_rows[0].push_back(7);
        yabi::UserODERHS<double> invalid(3, ode, jacobian);
        invalid.setSparsityPattern(bad);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    yabi_test::check(threw, "an out-of-range row index is rejected");
}

void testSparseJacobian() {
    yabi::UserODERHS<double> rhs(3, ode, jacobian);
    rhs.setSparsityPattern(makePattern());

    Vec x = {1.5, -0.5, 2.0};
    yabi::SparsityPattern pattern;
    yabi_test::check(rhs.jacobianSparsity(pattern), "pattern for extraction");

    Vec values;
    yabi_test::check(rhs.sparseJacobian(x, pattern, values), "sparseJacobian succeeds");
    yabi_test::check(values.size() == pattern.nnz(), "sparseJacobian returns exactly nnz values");

    Vec J(9, 0.0);
    jacobian(x.data(), J.data());
    std::size_t k = 0;
    double max_diff = 0.0;
    for (std::size_t j = 0; j < 3; ++j)
        for (std::size_t i : pattern.column_rows[j]) {
            max_diff = std::max(max_diff, std::fabs(values[k] - J[i + 3 * j]));
            ++k;
        }
    yabi_test::check(k == values.size(), "column-major extraction visits every value");
    yabi_test::check(max_diff < 1e-15, "sparse values match the analytic Jacobian");

    // A different pattern is not silently accepted.
    yabi::SparsityPattern other = makePattern();
    other.column_rows[1].clear();
    yabi_test::check(!rhs.sparseJacobian(x, other, values),
                     "sparseJacobian rejects a mismatched pattern");
}

void testJacobianVectorProduct() {
    Vec x = {1.5, -0.5, 2.0};
    Vec v = {0.25, -1.0, 0.75};

    Vec J(9, 0.0);
    jacobian(x.data(), J.data());
    Vec expected(3, 0.0);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            expected[i] += J[i + 3 * j] * v[j];

    // Default: exact J*v from the analytic Jacobian.
    yabi::UserODERHS<double> fallback(3, ode, jacobian);
    Vec product;
    yabi_test::check(fallback.jacobianVectorProduct(x, v, product),
                     "default jacobianVectorProduct succeeds");
    double max_diff = 0.0;
    for (std::size_t i = 0; i < 3; ++i)
        max_diff = std::max(max_diff, std::fabs(product[i] - expected[i]));
    yabi_test::check(max_diff < 1e-15, "default J*v is exact");

    // Analytic hook: the matrix-free path must use it instead of forming J.
    unsigned int calls = 0;
    Vec seen(3, 0.0);
    yabi::UserODERHS<double> hooked(3, ode, jacobian);
    hooked.setJacobianVectorProduct([&](const double *y, const double *d, double *out) {
        ++calls;
        for (std::size_t i = 0; i < 3; ++i)
            seen[i] = 2.0 * d[i]; // distinct from J*v, so the hook is observable
        (void)y;
        for (std::size_t i = 0; i < 3; ++i)
            out[i] = seen[i];
    });

    Vec hook_product;
    yabi_test::check(hooked.jacobianVectorProduct(x, v, hook_product),
                     "analytic jacobianVectorProduct succeeds");
    yabi_test::check(calls == 1, "analytic J*v hook is used exactly once");
    max_diff = 0.0;
    for (std::size_t i = 0; i < 3; ++i)
        max_diff = std::max(max_diff, std::fabs(hook_product[i] - 2.0 * v[i]));
    yabi_test::check(max_diff < 1e-15, "analytic J*v hook result is returned");

    // Wrong-size inputs are rejected rather than read out of bounds.
    Vec bad(2, 0.0);
    yabi_test::check(!hooked.jacobianVectorProduct(bad, v, hook_product),
                     "J*v rejects a wrong-size point");
    yabi_test::check(!hooked.jacobianVectorProduct(x, bad, hook_product),
                     "J*v rejects a wrong-size direction");
}

void testClonePreservesHooks() {
    unsigned int calls = 0;
    yabi::UserODERHS<double> rhs(3, ode, jacobian);
    rhs.setSparsityPattern(makePattern());
    rhs.setJacobianVectorProduct([&](const double *y, const double *v, double *out) {
        ++calls;
        Vec J(9, 0.0);
        jacobian(y, J.data());
        for (std::size_t i = 0; i < 3; ++i) {
            out[i] = 0.0;
            for (std::size_t j = 0; j < 3; ++j)
                out[i] += J[i + 3 * j] * v[j];
        }
    });

    std::unique_ptr<yabi::AbstractRHS<double>> copy(rhs.clone());
    yabi_test::check(copy->hasSparsityPattern(), "clone preserves the sparsity pattern");
    yabi::SparsityPattern pattern;
    yabi_test::check(copy->jacobianSparsity(pattern), "clone exposes the sparsity pattern");

    Vec x = {1.5, -0.5, 2.0};
    Vec v = {0.25, -1.0, 0.75};
    Vec product;
    yabi_test::check(copy->jacobianVectorProduct(x, v, product),
                     "clone preserves the analytic J*v hook");
    yabi_test::check(calls == 1, "clone routes through the copied hook");

    // The cloned pattern is normalised, so compare against the normalised form.
    yabi::SparsityPattern normalised = makePattern();
    for (auto &rows : normalised.column_rows) {
        std::sort(rows.begin(), rows.end());
        rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    }
    yabi_test::check(pattern == normalised, "clone pattern equals the normalised source");
}

} // namespace

int main() {
    testPatternValidation();
    testSparseJacobian();
    testJacobianVectorProduct();
    testClonePreservesHooks();
    return yabi_test::finish("UserODERHS sparsity/J*v unit tests");
}
