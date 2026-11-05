// T31 — native sparse infrastructure.
//
// Proves end-to-end that:
//   * CppAD sparsity propagation detects the state Jacobian pattern,
//   * SparseJacobianForward reproduces the dense Forward(1) Jacobian entries,
//   * the integrator assembles/scales/factorizes J/P in CSC (no dense J scan),
//   * a sparse integration reproduces the dense LAPACK integration, and
//   * the CSC solver interface is exercised on every factorization.
#include <yabi/bdf_integrator.hpp>
#include <yabi/cppad_rhs.hpp>
#include <yabi/csc_matrix.hpp>
#include <yabi/detail/dense_linear_solver.ipp>
#include <yabi/sparsity_pattern.hpp>

#include <cppad/cppad.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const std::string &what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << std::endl;
        ++failures;
    }
}

// Tridiagonal diffusion + decay, the structure shared by the heat-1D and
// Polanyi-Wigner benchmarks.
CppAD::ADFun<double> makeTape(std::size_t n) {
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
    CppAD::ADFun<double> rhs;
    rhs.Dependent(x, f);
    rhs.optimize();
    return rhs;
}

// Compare every structural sparse Jacobian entry against the dense Forward(1)
// sweep that the historical path performs.
void checkJacobianValues(std::size_t n) {
    CppAD::ADFun<double> fun = makeTape(n);
    yabi::CppADRHS<double> rhs(fun);

    check(rhs.hasSparsityPattern(), "CppADRHS exposes a sparsity pattern");
    yabi::SparsityPattern pattern;
    check(rhs.jacobianSparsity(pattern), "jacobianSparsity succeeds");

    const std::size_t expected_nnz = 3 * n - 2;
    check(pattern.rows == n && pattern.cols == n, "pattern is n x n");
    check(pattern.nnz() == expected_nnz,
          "tridiagonal nnz = 3n-2 (got " + std::to_string(pattern.nnz()) + ")");

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
    check(structure_ok, "pattern columns are the tridiagonal rows");

    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i)
        x[i] = 1.0 + 0.01 * (double)i;

    std::vector<double> sparse;
    check(rhs.sparseJacobian(x, pattern, sparse), "sparseJacobian succeeds");
    check(sparse.size() == expected_nnz, "sparseJacobian returns only nnz values");

    // Dense reference: one Forward(1) per column.
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
    check(max_diff < 1e-14, "sparse Jacobian matches dense Forward(1) entries");
}

struct RunResult {
    std::vector<double> y;
    bool ok = false;
    unsigned int jacobian_evaluations = 0;
};

RunResult integrate(std::size_t n, bool sparse) {
    CppAD::ADFun<double> fun = makeTape(n);
    yabi::BDFIntegrator<double> integrator(fun);
    integrator.setSparseAssembly(sparse);

    std::vector<double> y0(n);
    for (std::size_t i = 0; i < n; ++i)
        y0[i] = 1.0 + 0.01 * (double)i;
    std::vector<double> abs_err(n, 1e-8);

    RunResult result;
    if (!integrator.resetIntegration(y0.data(), 0.1, 1e-6, abs_err.data(), false, false, nullptr,
                                     nullptr, nullptr, nullptr, 0))
        return result;
    if (!integrator.setAlgorithmSettings(5, 1e-16, true, 1))
        return result;
    result.ok = integrator.performIntegration(0.1, 100000);
    result.y.resize(n);
    integrator.getSolutionVector(result.y.data());
    result.jacobian_evaluations = integrator.getJacobianEvaluations();
    return result;
}

void checkIntegration(std::size_t n, const char *name) {
    RunResult dense = integrate(n, false);
    RunResult sparse = integrate(n, true);

    check(dense.ok, std::string(name) + ": dense integration succeeds");
    check(sparse.ok, std::string(name) + ": sparse integration succeeds");
    check(dense.jacobian_evaluations == sparse.jacobian_evaluations,
          std::string(name) + ": identical Jacobian evaluation count");

    double max_diff = 0.0;
    for (std::size_t i = 0; i < n && dense.y.size() == sparse.y.size(); ++i)
        max_diff = std::max(max_diff, std::fabs(dense.y[i] - sparse.y[i]));
    check(max_diff == 0.0, std::string(name) + ": sparse result is bit-identical to dense");
    std::cout << "  " << name << ": n=" << n << ", max|dense-sparse|=" << max_diff
              << ", jacobian evaluations=" << sparse.jacobian_evaluations << "\n";
}

void checkCscReferenceSolver() {
    yabi::DenseLUSolver<double> solver;
    check(solver.supportsCsc(), "DenseLUSolver supports CSC reference path");
    check(!solver.prefersSparseFactorization(), "DenseLUSolver is not a production sparse solver");

    yabi::SparsityPattern pattern;
    pattern.rows = 3;
    pattern.cols = 3;
    pattern.column_rows = {{0, 1}, {0, 1, 2}, {1, 2}};
    yabi::CscMatrix<double> P;
    check(P.assignPattern(pattern, true), "CscMatrix assigns pattern with diagonal");
    check(P.nnz() == 7, "CscMatrix has explicit diagonal (tridiagonal 3x3 -> 7)");
    check(P.value(0, 0) == 0.0 && P.value(1, 1) == 0.0 && P.value(2, 2) == 0.0,
          "CscMatrix diagonal entries start at zero");

    P.val = {4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0};
    int ipiv[3] = {0, 0, 0};
    long d1[3] = {0, 0, 0};
    long d2[3] = {0, 0, 0};
    check(solver.factorizeCsc(P, ipiv, d1, d2, 3), "factorizeCsc succeeds");
    check(solver.cscFactorizations() == 1, "factorizeCsc counted");

    double b[3] = {5.0, 6.0, 7.0};
    check(solver.solveCsc(P, ipiv, d1, d2, b, 3, 1) == 0, "solveCsc succeeds");
    check(std::fabs(b[0] - 1.0357142857142858) < 1e-14 &&
              std::fabs(b[1] - 0.8571428571428571) < 1e-14 &&
              std::fabs(b[2] - 1.5357142857142858) < 1e-14,
          "solveCsc solves P*x=b");
}

} // namespace

int main() {
    std::cout << "T31 native sparse infrastructure\n";

    checkJacobianValues(50);
    checkCscReferenceSolver();

    // heat-1D n=50: nnz 148/2500 (5.9%)
    checkIntegration(50, "heat-1D");
    // Polanyi-Wigner / semi-discretized PDE n=20: nnz 58/400 (14.5%)
    checkIntegration(20, "polanyi-wigner");

    if (failures != 0) {
        std::cerr << failures << " sparse check(s) failed" << std::endl;
        return 1;
    }
    std::cout << "PASS: native sparse Jacobian, CSC assembly and integration\n";
    return 0;
}
