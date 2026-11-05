/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

#ifndef YABI_SPARSITY_PATTERN_HPP
#define YABI_SPARSITY_PATTERN_HPP

#include <algorithm>
#include <cstddef>
#include <vector>

namespace yabi {

// Column-oriented sparsity pattern of the state Jacobian J = df/dy.
//
// `column_rows[j]` holds the row indices i for which J(i,j) may be non-zero.
// Keeping the pattern column-oriented matches compressed sparse column (CSC)
// storage, so a detected pattern can be used directly as the structure of J
// and P without transposing or scanning a dense buffer.
struct SparsityPattern {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<std::vector<std::size_t>> column_rows;

    std::size_t nnz() const {
        std::size_t total = 0;
        for (const auto &rows_in_column : column_rows)
            total += rows_in_column.size();
        return total;
    }

    bool empty() const {
        return rows == 0 || cols == 0 || nnz() == 0;
    }

    bool operator==(const SparsityPattern &other) const {
        return rows == other.rows && cols == other.cols && column_rows == other.column_rows;
    }
    bool operator!=(const SparsityPattern &other) const {
        return !(*this == other);
    }
};

} // namespace yabi

#endif
