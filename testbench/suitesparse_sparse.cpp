// T31 — SuiteSparse symbolic reuse and native CSC integration.
//
// Not part of the numbered CppAD glob; CMake registers it only when both
// UMFPACK and KLU are available.  It proves the second T31 gate item:
// pattern-stable factorizations reuse the symbolic analysis.
#include <yabi/bdf_integrator.hpp>
#include <yabi/csc_matrix.hpp>
#include <yabi/detail/klu_linear_solver.ipp>
#include <yabi/detail/umfpack_linear_solver.ipp>
#include <yabi/sparsity_pattern.hpp>

#include <cppad/cppad.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
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

// Pattern-stable numeric refactorization must not redo the symbolic analysis.
template <typename Solver> void checkSymbolicReuse(const char *name) {
    const std::size_t n = 4;
    yabi::SparsityPattern pattern;
    pattern.rows = n;
    pattern.cols = n;
    pattern.column_rows = {{0, 1}, {0, 1, 2}, {1, 2, 3}, {2, 3}};

    yabi::CscMatrix<double> P;
    check(P.assignPattern(pattern, true), std::string(name) + ": assign pattern");
    for (std::size_t j = 0; j < n; ++j)
        for (int k = P.col_ptr[j]; k < P.col_ptr[j + 1]; ++k)
            P.val[(std::size_t)k] = (P.row_idx[(std::size_t)k] == (int)j) ? 4.0 : 1.0;

    Solver solver;
    check(solver.supportsCsc(), std::string(name) + ": supports CSC");
    check(solver.prefersSparseFactorization(), std::string(name) + ": prefers sparse");
    solver.initialize((unsigned int)n);

    int ipiv[4] = {0, 0, 0, 0};
    long d1[4] = {0, 0, 0, 0};
    long d2[4] = {0, 0, 0, 0};

    check(solver.factorizeCsc(P, ipiv, d1, d2, (unsigned int)n),
          std::string(name) + ": first factorize");
    check(solver.symbolicFactorizations() == 1, std::string(name) + ": first symbolic");

    // Same pattern, new values: reuse symbolic, only numeric changes.
    for (std::size_t k = 0; k < P.val.size(); ++k)
        P.val[k] *= 2.0;
    check(solver.factorizeCsc(P, ipiv, d1, d2, (unsigned int)n),
          std::string(name) + ": second factorize");
    check(solver.symbolicFactorizations() == 1,
          std::string(name) + ": symbolic reused across pattern-unchanged steps");

    double b[4] = {1.0, 1.0, 1.0, 1.0};
    check(solver.solveCsc(P, ipiv, d1, d2, b, (unsigned int)n, 1) == 0,
          std::string(name) + ": solve");
    for (double value : b)
        check(std::isfinite(value), std::string(name) + ": finite solution");

    // Different pattern: a new symbolic analysis is required.
    yabi::SparsityPattern dense = pattern;
    dense.column_rows.assign(n, {0, 1, 2, 3});
    yabi::CscMatrix<double> P2;
    check(P2.assignPattern(dense, false), std::string(name) + ": dense pattern");
    for (std::size_t j = 0; j < n; ++j)
        for (int k = P2.col_ptr[j]; k < P2.col_ptr[j + 1]; ++k)
            P2.val[(std::size_t)k] = (P2.row_idx[(std::size_t)k] == (int)j) ? 4.0 : 1.0;
    int ipiv2[4] = {0, 0, 0, 0};
    check(solver.factorizeCsc(P2, ipiv2, d1, d2, (unsigned int)n),
          std::string(name) + ": dense factorize");
    check(solver.symbolicFactorizations() == 2,
          std::string(name) + ": new pattern triggers new symbolic");
}

template <typename Solver> std::vector<double> integrateSparse(std::size_t n, const char *name) {
    CppAD::ADFun<double> fun = makeTape(n);
    yabi::BDFIntegrator<double> integrator(fun);
    Solver *solver = new Solver();
    integrator.setLinearSolver(solver);
    check(integrator.getSparseAssembly(),
          std::string(name) + ": sparse assembly auto-enabled for native CSC solver");

    std::vector<double> y0(n);
    for (std::size_t i = 0; i < n; ++i)
        y0[i] = 1.0 + 0.01 * (double)i;
    std::vector<double> abs_err(n, 1e-8);

    if (!integrator.resetIntegration(y0.data(), 0.01, 1e-6, abs_err.data(), false, false, nullptr,
                                     nullptr, nullptr, nullptr, 0)) {
        check(false, std::string(name) + ": reset");
        return {};
    }
    if (!integrator.setAlgorithmSettings(5, 1e-16, false, 1)) {
        check(false, std::string(name) + ": algorithm settings");
        return {};
    }
    // Refresh P every step so a pattern-stable run performs many numeric
    // factorizations and the symbolic analysis is reused across all of them.
    if (!integrator.setNewtonSettings(-1.0, 0.01, 0.30, 10, 3, true)) {
        check(false, std::string(name) + ": newton settings");
        return {};
    }
    if (!integrator.performIntegration(0.1, 100000)) {
        check(false, std::string(name) + ": integration");
        return {};
    }

    std::vector<double> y(n);
    integrator.getSolutionVector(y.data());

    check(integrator.getFactorizations() > 1, std::string(name) + ": more than one factorization");
    check(solver->symbolicFactorizations() == 1,
          std::string(name) + ": exactly one symbolic factorization over the run");
    std::cout << "  " << name << ": factorizations=" << integrator.getFactorizations()
              << " symbolic=" << solver->symbolicFactorizations() << "\n";
    return y;
}

std::vector<double> integrateDense(std::size_t n) {
    CppAD::ADFun<double> fun = makeTape(n);
    yabi::BDFIntegrator<double> integrator(fun);
    std::vector<double> y0(n);
    for (std::size_t i = 0; i < n; ++i)
        y0[i] = 1.0 + 0.01 * (double)i;
    std::vector<double> abs_err(n, 1e-8);
    if (!integrator.resetIntegration(y0.data(), 0.01, 1e-6, abs_err.data(), false, false, nullptr,
                                     nullptr, nullptr, nullptr, 0))
        return {};
    if (!integrator.setAlgorithmSettings(5, 1e-16, false, 1)) {
        return {};
    }
    if (!integrator.setNewtonSettings(-1.0, 0.01, 0.30, 10, 3, true)) {
        return {};
    }
    if (!integrator.performIntegration(0.1, 100000))
        return {};
    std::vector<double> y(n);
    integrator.getSolutionVector(y.data());
    return y;
}

void checkAgainstDense(const std::vector<double> &sparse, const std::vector<double> &dense,
                       const char *name) {
    check(!sparse.empty() && sparse.size() == dense.size(),
          std::string(name) + ": sparse result size");
    if (sparse.empty() || sparse.size() != dense.size())
        return;
    double max_diff = 0.0;
    for (std::size_t i = 0; i < sparse.size(); ++i)
        max_diff = std::max(max_diff, std::fabs(sparse[i] - dense[i]));
    check(max_diff < 1e-10, std::string(name) + ": sparse matches dense LAPACK");
    std::cout << "  " << name << ": max|sparse-dense|=" << max_diff << "\n";
}

} // namespace

int main() {
    std::cout << "T31 SuiteSparse native sparse path\n";

    checkSymbolicReuse<yabi::UmfpackLinearSolver<double>>("umfpack");
    checkSymbolicReuse<yabi::KluLinearSolver<double>>("klu");

    const std::size_t n = 50;
    std::vector<double> dense = integrateDense(n);
    checkAgainstDense(integrateSparse<yabi::UmfpackLinearSolver<double>>(n, "umfpack heat-1D"),
                      dense, "umfpack heat-1D");
    checkAgainstDense(integrateSparse<yabi::KluLinearSolver<double>>(n, "klu heat-1D"), dense,
                      "klu heat-1D");

    if (failures != 0) {
        std::cerr << failures << " SuiteSparse sparse check(s) failed" << std::endl;
        return 1;
    }
    std::cout << "PASS: native CSC + symbolic reuse (UMFPACK, KLU)\n";
    return 0;
}
