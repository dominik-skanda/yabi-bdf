/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

#include "test_support.hpp"

#include <yabi/sparsity_pattern.hpp>

#include <cstddef>
#include <vector>

int main() {
    yabi::SparsityPattern empty;
    yabi_test::check(empty.empty(), "default pattern is empty");
    yabi_test::check(empty.nnz() == 0, "default pattern has zero nnz");

    // Column-oriented: column j lists the rows that may be non-zero.
    yabi::SparsityPattern tridiagonal;
    tridiagonal.rows = 4;
    tridiagonal.cols = 4;
    tridiagonal.column_rows = {{0, 1}, {0, 1, 2}, {1, 2, 3}, {2, 3}};

    yabi_test::check(!tridiagonal.empty(), "tridiagonal pattern is non-empty");
    yabi_test::check(tridiagonal.nnz() == 10, "tridiagonal nnz = 10");
    yabi_test::check(tridiagonal.column_rows[1] == std::vector<std::size_t>({0, 1, 2}),
                     "column 1 is column-oriented");

    yabi::SparsityPattern copy = tridiagonal;
    yabi_test::check(copy == tridiagonal, "copy compares equal");
    yabi_test::check(!(copy != tridiagonal), "operator!= agrees with operator==");

    copy.column_rows[2].push_back(0);
    yabi_test::check(copy != tridiagonal, "different structure compares unequal");

    yabi::SparsityPattern scalar;
    scalar.rows = 1;
    scalar.cols = 1;
    scalar.column_rows = {{0}};
    yabi_test::check(scalar.nnz() == 1 && !scalar.empty(), "1x1 pattern with one entry");

    // rows/cols describe the advertised shape even before entries are filled.
    yabi::SparsityPattern shapeOnly;
    shapeOnly.rows = 3;
    shapeOnly.cols = 3;
    yabi_test::check(shapeOnly.empty(), "shape without entries is empty");

    return yabi_test::finish("SparsityPattern unit tests");
}
