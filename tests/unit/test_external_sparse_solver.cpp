/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// Unit tests for the reusable ExternalSparseSolver base (T36d): the CSC->CSR
// handoff, the factorize/solve plumbing, the nrhs loop, and the interface flags.
// A mock derived solver records what the base hands it, so the shared half is
// verified in isolation — independently of AMGCL/Ginkgo/cuDSS.

#include "test_support.hpp"

#include <yabi/csc_matrix.hpp>
#include <yabi/external_sparse_solver.hpp>
#include <yabi/sparsity_pattern.hpp>

#include <vector>

namespace {

class MockSolver : public yabi::ExternalSparseSolver<double> {
  public:
    std::vector<int> built_ptr, built_col;
    std::vector<double> built_val;
    int solve_calls = 0;

  private:
    bool buildCsc(std::vector<int> &csr_ptr, std::vector<int> &csr_col,
                  std::vector<double> &csr_val) override {
        built_ptr = csr_ptr;
        built_col = csr_col;
        built_val = csr_val;
        return true;
    }

    bool solveOne(const std::vector<double> &rhs, std::vector<double> &x) override {
        ++solve_calls;
        for (std::size_t i = 0; i < rhs.size(); ++i)
            x[i] = 2.0 * rhs[i]; // deterministic transform
        return true;
    }
};

} // namespace

int main() {
    // --- interface flags ----------------------------------------------------
    MockSolver s;
    yabi_test::check(s.supportsCsc(), "supportsCsc is true");
    yabi_test::check(s.prefersSparseFactorization(), "prefersSparseFactorization is true");
    yabi_test::check(!s.factorize(nullptr, nullptr, nullptr, nullptr, 0),
                     "dense factorize is rejected (sparse-only)");
    yabi_test::check(s.solve(nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0) == -1,
                     "dense solve is rejected (sparse-only)");

    // --- CSC -> CSR handoff --------------------------------------------------
    yabi::SparsityPattern pattern;
    pattern.rows = 3;
    pattern.cols = 3;
    pattern.column_rows = {{0, 1}, {0, 1, 2}, {1, 2}}; // 7 nonzeros

    yabi::CscMatrix<double> J;
    yabi_test::check(J.assignPattern(pattern, false), "assign the CSC pattern");
    J.val = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}; // column-major

    yabi_test::check(s.factorizeCsc(J, nullptr, nullptr, nullptr, 3), "factorizeCsc succeeds");

    // Transpose: CSC(columns) -> CSR(rows).
    //   row 0: {0,1}={1,3};  row 1: {0,1,2}={2,4,6};  row 2: {1,2}={5,7}
    yabi_test::check(s.built_ptr == std::vector<int>({0, 2, 5, 7}), "CSR row pointers");
    yabi_test::check(s.built_col == std::vector<int>({0, 1, 0, 1, 2, 1, 2}), "CSR column indices");
    yabi_test::check(s.built_val == std::vector<double>({1.0, 3.0, 2.0, 4.0, 6.0, 5.0, 7.0}),
                     "CSR values are the transpose");

    // --- solveCsc loops nrhs ------------------------------------------------
    double b[6] = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0}; // two RHS, size 3 each
    yabi_test::check(s.solveCsc(J, nullptr, nullptr, nullptr, b, 3, 2) == 0, "solveCsc succeeds");
    yabi_test::check(s.solve_calls == 2, "solveCsc called solveOne once per RHS");
    yabi_test::check(b[0] == 2.0 && b[1] == 2.0 && b[2] == 2.0 && b[3] == 4.0 && b[4] == 4.0 &&
                         b[5] == 4.0,
                     "solveCsc writes each RHS result back");

    // --- edge cases ----------------------------------------------------------
    yabi::CscMatrix<double> empty;
    yabi_test::check(!s.factorizeCsc(empty, nullptr, nullptr, nullptr, 0), "empty matrix rejected");
    yabi_test::check(!s.factorizeCsc(J, nullptr, nullptr, nullptr, 4), "wrong n rejected");

    return yabi_test::finish("ExternalSparseSolver unit tests");
}
