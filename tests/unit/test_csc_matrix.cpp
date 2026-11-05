/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

#include "test_support.hpp"

#include <yabi/csc_matrix.hpp>
#include <yabi/sparsity_pattern.hpp>

#include <cmath>
#include <vector>

int main() {
    // --- structure assignment without diagonal insertion -------------------
    yabi::SparsityPattern tridiagonal;
    tridiagonal.rows = 3;
    tridiagonal.cols = 3;
    tridiagonal.column_rows = {{0, 1}, {0, 1, 2}, {1, 2}};

    yabi::CscMatrix<double> J;
    yabi_test::check(J.assignPattern(tridiagonal, false), "assign tridiagonal without diagonal");
    yabi_test::check(J.n == 3, "CscMatrix dimension");
    yabi_test::check(J.nnz() == 7, "tridiagonal CSC nnz");
    yabi_test::check(J.col_ptr == std::vector<int>({0, 2, 5, 7}), "CSC column pointers");
    yabi_test::check(J.row_idx == std::vector<int>({0, 1, 0, 1, 2, 1, 2}),
                     "CSC row indices sorted");
    yabi_test::check(J.val == std::vector<double>(7, 0.0), "CSC values start at zero");
    yabi_test::check(!J.empty(), "non-empty CSC");

    // Every purely structural entry reads back as zero until written.
    yabi_test::check(J.value(1, 1) == 0.0, "structural entry starts at zero");
    yabi_test::check(J.value(2, 0) == 0.0, "missing entry reads as zero");

    // --- diagonal insertion for P = I - c*J --------------------------------
    yabi::SparsityPattern offDiagonal;
    offDiagonal.rows = 3;
    offDiagonal.cols = 3;
    offDiagonal.column_rows = {{1}, {0}, {2}};

    yabi::CscMatrix<double> P;
    yabi_test::check(P.assignPattern(offDiagonal, true), "assign with explicit diagonal");
    yabi_test::check(P.nnz() == 5, "diagonal adds the two missing entries");
    yabi_test::check(P.col_ptr == std::vector<int>({0, 2, 4, 5}), "diagonal CSC column pointers");
    yabi_test::check(P.row_idx == std::vector<int>({0, 1, 0, 1, 2}),
                     "diagonal CSC rows stay sorted");

    // --- value access, dense expansion, pattern comparison -----------------
    P.val = {10.0, 11.0, 12.0, 13.0, 14.0};
    yabi_test::check(P.value(1, 0) == 11.0, "value reads a CSC entry");
    yabi_test::check(P.value(2, 1) == 0.0, "value returns zero for a hole");

    std::vector<double> dense;
    P.toDense(dense);
    yabi_test::check(dense.size() == 9, "dense expansion is n*n");
    yabi_test::check(dense[0] == 10.0 && dense[1 * 3 + 0] == 12.0 && dense[2 * 3 + 2] == 14.0,
                     "dense expansion is column-major");
    yabi_test::check(dense[0 * 3 + 2] == 0.0, "dense expansion leaves holes at zero");

    yabi::CscMatrix<double> same = P;
    yabi_test::check(P.hasSamePattern(same), "copied matrix has same pattern");
    same.val[0] = -1.0;
    yabi_test::check(P.hasSamePattern(same), "pattern ignores values");
    same.row_idx[0] = 2;
    yabi_test::check(!P.hasSamePattern(same), "different row indices differ");

    // --- Newton scaling: D2 scales columns, D1 scales rows -----------------
    yabi::SparsityPattern full;
    full.rows = 2;
    full.cols = 2;
    full.column_rows = {{0, 1}, {0, 1}};
    yabi::CscMatrix<double> S;
    yabi_test::check(S.assignPattern(full, false), "assign 2x2 full pattern");
    S.val = {2.0, 1.0, 1.0, 3.0}; // column-major [[2,1],[1,3]]

    std::vector<long> d2 = {1, 0}; // column 0 * 2
    S.scaleColumns(d2);
    yabi_test::check(S.val[0] == 4.0 && S.val[1] == 2.0 && S.val[2] == 1.0 && S.val[3] == 3.0,
                     "scaleColumns applies D2 per column");

    std::vector<long> d1 = {-1, -2}; // row 0 * 0.5, row 1 * 0.25
    S.scaleRows(d1);
    yabi_test::check(S.val[0] == 2.0 && S.val[1] == 0.5 && S.val[2] == 0.5 && S.val[3] == 0.75,
                     "scaleRows applies D1 per row");

    std::vector<double> rowSums;
    S.rowSums(rowSums);
    yabi_test::check(rowSums.size() == 2, "rowSums size");
    yabi_test::check(yabi_test::near(rowSums[0], 2.5) && yabi_test::near(rowSums[1], 1.25),
                     "rowSums uses absolute values");

    S.zeroValues();
    yabi_test::check(S.val == std::vector<double>(4, 0.0), "zeroValues clears every entry");

    S.clear();
    yabi_test::check(S.empty() && S.n == 0 && S.nnz() == 0, "clear resets the matrix");

    // --- invalid patterns are rejected ------------------------------------
    yabi::SparsityPattern rectangular;
    rectangular.rows = 2;
    rectangular.cols = 3;
    rectangular.column_rows = {{0}, {1}, {0}};
    yabi::CscMatrix<double> bad;
    yabi_test::check(!bad.assignPattern(rectangular, false), "reject non-square pattern");

    yabi::SparsityPattern outOfRange;
    outOfRange.rows = 2;
    outOfRange.cols = 2;
    outOfRange.column_rows = {{0}, {2}};
    yabi_test::check(!bad.assignPattern(outOfRange, false), "reject out-of-range row index");

    return yabi_test::finish("CscMatrix unit tests");
}
