/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

#ifndef YABI_CSC_MATRIX_HPP
#define YABI_CSC_MATRIX_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <yabi/sparsity_pattern.hpp>

namespace yabi {

// Square matrix in compressed sparse column (CSC) form.
//
//   col_ptr[j] .. col_ptr[j + 1] - 1  are the indices in row_idx/val of the
//                                     entries of column j
//   row_idx[k], val[k]                are the row and value of entry k
//
// Entries are kept sorted by row inside each column.  The structure is fixed
// when the pattern is assigned; re-evaluating the Jacobian only rewrites
// `val`, so a sparse direct solver can reuse its symbolic factorization.
template <typename Scalar> struct CscMatrix {
    std::size_t n = 0;
    std::vector<int> col_ptr;
    std::vector<int> row_idx;
    std::vector<Scalar> val;

    void clear() {
        n = 0;
        col_ptr.clear();
        row_idx.clear();
        val.clear();
    }

    bool empty() const {
        return n == 0 || val.empty();
    }
    std::size_t nnz() const {
        return val.size();
    }

    bool hasSamePattern(const CscMatrix &other) const {
        return n == other.n && col_ptr == other.col_ptr && row_idx == other.row_idx;
    }

    // Build the CSC structure from a column-oriented sparsity pattern.
    // `add_diagonal` is used for P = I - c*J: the identity contributes a
    // structural entry on every diagonal even when J has no diagonal entry.
    bool assignPattern(const SparsityPattern &pattern, bool add_diagonal) {
        if (pattern.rows != pattern.cols)
            return false;
        n = pattern.cols;
        col_ptr.assign(n + 1, 0);
        row_idx.clear();
        val.clear();

        std::size_t entries = 0;
        for (std::size_t j = 0; j < n; ++j) {
            col_ptr[j] = (int)entries;
            std::vector<std::size_t> rows = pattern.column_rows[j];
            std::sort(rows.begin(), rows.end());
            rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
            for (std::size_t row : rows) {
                if (row >= n)
                    return false;
                row_idx.push_back((int)row);
                ++entries;
            }
            if (add_diagonal)
                if (!std::binary_search(rows.begin(), rows.end(), j)) {
                    row_idx.push_back((int)j);
                    ++entries;
                }
        }
        col_ptr[n] = (int)entries;

        // `add_diagonal` inserts the diagonal in-place, which can break the
        // row ordering; re-sort each column once.
        if (add_diagonal) {
            for (std::size_t j = 0; j < n; ++j) {
                std::vector<std::size_t> order;
                order.reserve((std::size_t)(col_ptr[j + 1] - col_ptr[j]));
                for (int k = col_ptr[j]; k < col_ptr[j + 1]; ++k)
                    order.push_back((std::size_t)k);
                std::sort(order.begin(), order.end(),
                          [&](std::size_t a, std::size_t b) { return row_idx[a] < row_idx[b]; });
                std::vector<int> new_rows;
                new_rows.reserve(order.size());
                for (std::size_t k : order)
                    new_rows.push_back(row_idx[k]);
                for (std::size_t k = 0; k < new_rows.size(); ++k)
                    row_idx[(std::size_t)col_ptr[j] + k] = new_rows[k];
            }
        }

        val.assign((std::size_t)entries, (Scalar)0.0);
        return true;
    }

    void zeroValues() {
        std::fill(val.begin(), val.end(), (Scalar)0.0);
    }

    // Expand to column-major dense storage (the historical P/J layout).
    void toDense(std::vector<Scalar> &dense) const {
        dense.assign(n * n, (Scalar)0.0);
        for (std::size_t j = 0; j < n; ++j)
            for (int k = col_ptr[j]; k < col_ptr[j + 1]; ++k)
                dense[j * n + (std::size_t)row_idx[(std::size_t)k]] = val[(std::size_t)k];
    }

    Scalar value(std::size_t row, std::size_t col) const {
        for (int k = col_ptr[col]; k < col_ptr[col + 1]; ++k)
            if ((std::size_t)row_idx[(std::size_t)k] == row)
                return val[(std::size_t)k];
        return (Scalar)0.0;
    }

    // Diagonal in natural (column) order.  Missing structural diagonal entries
    // are reported as zero, matching value(i, i).
    void extractDiagonal(std::vector<Scalar> &diagonal) const {
        diagonal.assign(n, (Scalar)0.0);
        for (std::size_t j = 0; j < n; ++j)
            for (int k = col_ptr[j]; k < col_ptr[j + 1]; ++k)
                if ((std::size_t)row_idx[(std::size_t)k] == j) {
                    diagonal[j] = val[(std::size_t)k];
                    break;
                }
    }

    // D2 scales columns (as the historical Newton scaling does).
    void scaleColumns(const std::vector<long> &scaling_D2) {
        for (std::size_t j = 0; j < n; ++j)
            for (int k = col_ptr[j]; k < col_ptr[j + 1]; ++k)
                val[(std::size_t)k] = (Scalar)std::scalbn(val[(std::size_t)k], (int)scaling_D2[j]);
    }

    // D1 scales rows.
    void scaleRows(const std::vector<long> &scaling_D1) {
        for (std::size_t k = 0; k < val.size(); ++k)
            val[k] = (Scalar)std::scalbn(val[k], (int)scaling_D1[(std::size_t)row_idx[k]]);
    }

    void rowSums(std::vector<Scalar> &sums) const {
        sums.assign(n, (Scalar)0.0);
        for (std::size_t k = 0; k < val.size(); ++k)
            sums[(std::size_t)row_idx[k]] += std::fabs(val[k]);
    }
};

} // namespace yabi

#endif
