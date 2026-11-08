/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// Property-based (deterministic-PRNG) checks for the sparse-matrix invariants:
// over many random matrices, the CSC storage equals the dense matrix, and the
// CSC->CSR transpose (ExternalSparseSolver's handoff) equals the matrix
// transpose.  A fixed seed keeps it reproducible.

#include "test_support.hpp"

#include <yabi/csc_matrix.hpp>
#include <yabi/external_sparse_solver.hpp>
#include <yabi/sparsity_pattern.hpp>

#include <random>
#include <vector>

namespace {

class RecordingSolver : public yabi::ExternalSparseSolver<double> {
  public:
    std::vector<int> ptr, col;
    std::vector<double> val;

  private:
    bool buildCsc(std::vector<int> &csr_ptr, std::vector<int> &csr_col,
                  std::vector<double> &csr_val) override {
        ptr = csr_ptr;
        col = csr_col;
        val = csr_val;
        return true;
    }
    bool solveOne(const std::vector<double> &, std::vector<double> &) override { return true; }
};

} // namespace

int main() {
    std::mt19937 rng(12345); // fixed seed
    std::uniform_real_distribution<double> value(-1.0, 1.0);
    std::bernoulli_distribution nonzero(0.5);

    const int trials = 200;
    for (int trial = 0; trial < trials; ++trial) {
        const int n = 2 + (trial % 7); // n in [2, 8]

        // Random dense matrix + column-oriented pattern.
        std::vector<std::vector<double>> dense(n, std::vector<double>(n, 0.0));
        yabi::SparsityPattern pattern;
        pattern.rows = n;
        pattern.cols = n;
        pattern.column_rows.resize(n);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                if (nonzero(rng)) {
                    pattern.column_rows[j].push_back(i);
                    dense[i][j] = value(rng);
                }

        yabi::CscMatrix<double> csc;
        if (!csc.assignPattern(pattern, false)) {
            yabi_test::check(false, "property: assignPattern failed");
            continue;
        }
        // Fill the CSC values from the dense matrix (rows are sorted by column).
        for (int j = 0; j < n; ++j)
            for (int k = csc.col_ptr[j]; k < csc.col_ptr[j + 1]; ++k)
                csc.val[(std::size_t)k] = dense[(std::size_t)csc.row_idx[k]][(std::size_t)j];

        // CSC == dense.
        std::vector<double> expanded;
        csc.toDense(expanded);
        bool dense_ok = true;
        for (int i = 0; i < n && dense_ok; ++i)
            for (int j = 0; j < n && dense_ok; ++j)
                if (!yabi_test::near(expanded[(std::size_t)j * n + i], dense[i][j], 1e-12))
                    dense_ok = false;
        yabi_test::check(dense_ok, "property: CSC == dense");

        // CSC->CSR == transpose (only for non-empty matrices; the empty case is
        // covered in test_external_sparse_solver).
        if (!csc.empty()) {
            RecordingSolver solver;
            if (!solver.factorizeCsc(csc, nullptr, nullptr, nullptr, (unsigned)n)) {
                yabi_test::check(false, "property: factorizeCsc failed");
                continue;
            }
            bool transpose_ok = true;
            for (int i = 0; i < n && transpose_ok; ++i)
                for (int k = solver.ptr[i]; k < solver.ptr[i + 1] && transpose_ok; ++k) {
                    const int j = solver.col[(std::size_t)k];
                    if (!yabi_test::near(solver.val[(std::size_t)k], dense[i][j], 1e-12))
                        transpose_ok = false;
                }
            yabi_test::check(transpose_ok, "property: CSC->CSR == transpose");
        }
    }

    return yabi_test::finish("property-based sparse checks");
}
