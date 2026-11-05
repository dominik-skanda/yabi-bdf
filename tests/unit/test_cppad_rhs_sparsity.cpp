/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

#include "test_support.hpp"

#include <yabi/cppad_rhs.hpp>
#include <yabi/sparsity_pattern.hpp>

#include <cppad/cppad.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace {

CppAD::ADFun<double> makeTridiagonalTape(std::size_t n) {
    CppAD::vector<CppAD::AD<double>> x(n);
    for (std::size_t i = 0; i < n; ++i)
        x[i] = 1.0 + 0.01 * (double)i;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(n);
    for (std::size_t i = 0; i < n; ++i) {
        CppAD::AD<double> left = i > 0 ? x[i - 1] : CppAD::AD<double>(0.0);
        CppAD::AD<double> right = i + 1 < n ? x[i + 1] : CppAD::AD<double>(0.0);
        f[i] = 0.01 * (left - 2.0 * x[i] + right) - 0.1 * x[i];
    }
    CppAD::ADFun<double> fun;
    fun.Dependent(x, f);
    fun.optimize();
    return fun;
}

// A tape whose domain is state + time + parameter.  The state Jacobian hooks
// must return only the state block, ignoring the time/parameter columns.
CppAD::ADFun<double> makeParameterizedTape(std::size_t n) {
    CppAD::vector<CppAD::AD<double>> x(n + 2); // state, time, parameter
    for (std::size_t i = 0; i < n; ++i)
        x[i] = 1.0 + 0.01 * (double)i;
    x[n] = 0.0;
    x[n + 1] = 1.0;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(n);
    for (std::size_t i = 0; i < n; ++i) {
        CppAD::AD<double> left = i > 0 ? x[i - 1] : CppAD::AD<double>(0.0);
        CppAD::AD<double> right = i + 1 < n ? x[i + 1] : CppAD::AD<double>(0.0);
        f[i] = x[n + 1] * (0.01 * (left - 2.0 * x[i] + right) - 0.1 * x[i]);
    }
    CppAD::ADFun<double> fun;
    fun.Dependent(x, f);
    fun.optimize();
    return fun;
}

void testTridiagonalDetection() {
    const std::size_t n = 50;
    CppAD::ADFun<double> fun = makeTridiagonalTape(n);
    yabi::CppADRHS<double> rhs(fun);

    yabi_test::check(rhs.hasSparsityPattern(), "CppADRHS advertises a sparsity pattern");
    yabi_test::check(rhs.Domain() == n && rhs.Range() == n, "CppADRHS reports n x n");

    yabi::SparsityPattern pattern;
    yabi_test::check(rhs.jacobianSparsity(pattern), "jacobianSparsity succeeds");
    yabi_test::check(pattern.rows == n && pattern.cols == n, "pattern is n x n");
    yabi_test::check(pattern.nnz() == 3 * n - 2, "tridiagonal nnz = 3n-2");

    bool structure_ok = true;
    for (std::size_t j = 0; j < n; ++j) {
        std::vector<std::size_t> expected;
        if (j > 0)
            expected.push_back(j - 1);
        expected.push_back(j);
        if (j + 1 < n)
            expected.push_back(j + 1);
        structure_ok = structure_ok && pattern.column_rows[j] == expected;
    }
    yabi_test::check(structure_ok, "pattern columns are the tridiagonal rows");

    // Values: every structural entry must equal the dense Forward(1) entry.
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i)
        x[i] = 1.0 + 0.01 * (double)i;

    std::vector<double> sparse;
    yabi_test::check(rhs.sparseJacobian(x, pattern, sparse), "sparseJacobian succeeds");
    yabi_test::check(sparse.size() == pattern.nnz(), "sparseJacobian returns only nnz values");

    std::vector<double> dx(n, 0.0);
    double max_diff = 0.0;
    std::size_t k = 0;
    for (std::size_t j = 0; j < n; ++j) {
        dx[j] = 1.0;
        std::vector<double> df = rhs.Forward(1, dx);
        dx[j] = 0.0;
        for (std::size_t i : pattern.column_rows[j]) {
            max_diff = std::max(max_diff, std::fabs(sparse[k] - df[i]));
            ++k;
        }
    }
    yabi_test::check(max_diff < 1e-14, "sparse Jacobian matches dense Forward(1) entries");

    // A second evaluation must reuse the cached pattern and stay identical.
    std::vector<double> sparse2;
    yabi_test::check(rhs.sparseJacobian(x, pattern, sparse2), "sparseJacobian reuses the pattern");
    yabi_test::check(sparse2 == sparse, "cached pattern evaluation is deterministic");
}

void testStateBlockOnly() {
    const std::size_t n = 8;
    CppAD::ADFun<double> fun = makeParameterizedTape(n);
    yabi::CppADRHS<double> rhs(fun);

    yabi_test::check(rhs.Domain() == n + 2 && rhs.Range() == n, "parameterised tape dimensions");

    yabi::SparsityPattern pattern;
    yabi_test::check(rhs.jacobianSparsity(pattern), "parameterised jacobianSparsity");
    yabi_test::check(pattern.rows == n && pattern.cols == n,
                     "state Jacobian excludes time/parameter columns");

    // The multiplier is a parameter, so the state structure is tilediagonal.
    yabi_test::check(pattern.nnz() == 3 * n - 2, "parameterised tridiagonal nnz");

    std::vector<double> x(n + 2);
    for (std::size_t i = 0; i < n; ++i)
        x[i] = 1.0 + 0.01 * (double)i;
    x[n] = 3.0;     // time
    x[n + 1] = 2.0; // parameter

    std::vector<double> sparse;
    yabi_test::check(rhs.sparseJacobian(x, pattern, sparse), "parameterised sparseJacobian");

    // Compare on the state block against dense Forward(1) directions.
    std::vector<double> dx(n + 2, 0.0);
    double max_diff = 0.0;
    std::size_t k = 0;
    for (std::size_t j = 0; j < n; ++j) {
        dx[j] = 1.0;
        std::vector<double> df = rhs.Forward(1, dx);
        dx[j] = 0.0;
        for (std::size_t i : pattern.column_rows[j]) {
            max_diff = std::max(max_diff, std::fabs(sparse[k] - df[i]));
            ++k;
        }
    }
    yabi_test::check(max_diff < 1e-14, "parameterised sparse state block matches dense");
}

} // namespace

int main() {
    testTridiagonalDetection();
    testStateBlockOnly();
    return yabi_test::finish("CppADRHS sparsity unit tests");
}
