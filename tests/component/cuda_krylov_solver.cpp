// T36e — GPU sparse iterative Krylov solver (own implementation), cross-checked
// against the CPU dense-LU path.  Built only when YABI_USE_CUDA=ON.
//
// Proves the CudaKrylovSolver (cuBLAS BiCGStab + cuSPARSE SpMV) reproduces the
// CPU dense-LU trajectory on the stiff HIRES benchmark (Hairer & Wanner §IV.4).
#include <yabi/bdf_integrator.hpp>
#include <yabi/cppad_rhs.hpp>
#include <yabi/cuda_krylov_solver.hpp>
#include <yabi/detail/dense_linear_solver.ipp>

#include <cppad/cppad.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

// Unit check: a known tridiagonal matrix (1D Laplacian: 2 on the diagonal,
// -1 off) so the solver's factorizeCsc/solveCsc can be tested in isolation.
yabi::CscMatrix<double> buildTridiagonal(unsigned n) {
    yabi::SparsityPattern pattern;
    pattern.rows = n;
    pattern.cols = n;
    pattern.column_rows.resize(n);
    for (unsigned j = 0; j < n; ++j) {
        if (j > 0)
            pattern.column_rows[j].push_back(j - 1);
        pattern.column_rows[j].push_back(j);
        if (j + 1 < n)
            pattern.column_rows[j].push_back(j + 1);
    }
    yabi::CscMatrix<double> A;
    A.assignPattern(pattern, /*add_diagonal=*/false);
    for (std::size_t j = 0; j < n; ++j)
        for (int k = A.col_ptr[j]; k < A.col_ptr[j + 1]; ++k)
            A.val[k] = ((std::size_t)A.row_idx[k] == j) ? 2.0 : -1.0;
    return A;
}

bool solveTridiagonalDirectly() {
    const unsigned n = 64;
    yabi::CscMatrix<double> A = buildTridiagonal(n);
    std::vector<double> b(n), x(n);
    for (unsigned i = 0; i < n; ++i)
        b[i] = (double)(i + 1);

    yabi::CudaKrylovSolver<double> solver;
    if (!solver.initialize(n))
        return false;
    solver.setTolerance(1e-12, 0.0, 500, 30);
    if (!solver.factorizeCsc(A, nullptr, nullptr, nullptr, n))
        return false;
    if (solver.solveCsc(A, nullptr, nullptr, nullptr, b.data(), n, 1) != 0)
        return false;
    std::copy(b.begin(), b.end(), x.begin());

    // Residual check: A*x should equal the original b.
    double maxres = 0.0;
    for (unsigned i = 0; i < n; ++i) {
        double Ax = 2.0 * x[i];
        if (i > 0)
            Ax -= x[i - 1];
        if (i + 1 < n)
            Ax -= x[i + 1];
        maxres = std::max(maxres, std::fabs(Ax - (double)(i + 1)));
    }
    std::cout << "  unit: tridiagonal n=" << n << " max|Ax-b| = " << maxres << "\n";
    return maxres < 1e-8;
}

CppAD::ADFun<double> makeHiresTape() {
    CppAD::vector<CppAD::AD<double>> x(8);
    x[0] = 1.0;
    x[1] = 0.0;
    x[2] = 0.0;
    x[3] = 0.0;
    x[4] = 0.0;
    x[5] = 0.0;
    x[6] = 0.0;
    x[7] = 0.0057;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(8);
    f[0] = -1.71 * x[0] + 0.43 * x[1] + 8.32 * x[2] + 0.0007;
    f[1] = 1.71 * x[0] - 8.75 * x[1];
    f[2] = -10.03 * x[2] + 0.43 * x[3] + 0.035 * x[4];
    f[3] = 8.32 * x[1] + 1.71 * x[2] - 1.12 * x[3];
    f[4] = -1.745 * x[4] + 0.43 * x[5] + 0.43 * x[6];
    f[5] = -280.0 * x[5] * x[7] + 0.69 * x[3] + 1.71 * x[4] - 0.43 * x[5] + 0.69 * x[6];
    f[6] = 280.0 * x[5] * x[7] - 1.81 * x[6];
    f[7] = -f[6];
    CppAD::ADFun<double> fun;
    fun.Dependent(x, f);
    fun.optimize();
    return fun;
}

bool integrate(bool gpu, std::vector<double> &y) {
    CppAD::ADFun<double> fun = makeHiresTape();
    yabi::BDFIntegrator<double> integrator(fun);
    if (gpu) {
        auto *solver = new yabi::CudaKrylovSolver<double>();
        solver->setTolerance(1e-10, 0.0, 200, 30);
        integrator.setLinearSolver(solver);
    } else {
        integrator.setLinearSolver(new yabi::DenseLUSolver<double>());
    }

    double y0[8] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057};
    double abs_err[8] = {1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10};
    if (!integrator.resetIntegration(y0, 0.01, 1e-8, abs_err, false, false, NULL, NULL, NULL,
                                     NULL, 0))
        return false;
    if (!integrator.setAlgorithmSettings(5, 1e-16, true, 1))
        return false;
    if (!integrator.performIntegration(0.1, 1000000))
        return false;
    y.resize(8);
    return integrator.getSolutionVector(y.data());
}

} // namespace

int main() {
    std::cout << "T36e CUDA (cuSPARSE Krylov) vs CPU dense (HIRES t=0.1)\n";

    if (!solveTridiagonalDirectly()) {
        std::cerr << "FAIL: direct tridiagonal solve\n";
        return 1;
    }

    std::vector<double> cpu, gpu;
    if (!integrate(false, cpu)) {
        std::cerr << "FAIL: CPU dense integration\n";
        return 1;
    }
    if (!integrate(true, gpu)) {
        std::cerr << "FAIL: GPU (cuSPARSE Krylov) integration\n";
        return 1;
    }

    double maxdiff = 0.0;
    for (std::size_t i = 0; i < 8; ++i)
        maxdiff = std::max(maxdiff, std::fabs(cpu[i] - gpu[i]));
    std::cout << "  max|CPU-GPU| = " << maxdiff << "\n";

    if (maxdiff < 1e-6) {
        std::cout << "PASS: cuSPARSE Krylov state solve matches the dense LAPACK path\n";
        return 0;
    }
    std::cerr << "FAIL: GPU Krylov state solve diverged from the CPU reference\n";
    return 1;
}
