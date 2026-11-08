/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// T42 — solver agreement matrix: every available linear solver reproduces the
// same HIRES trajectory as the dense LAPACK reference.  Built-in solvers (dense,
// matrix-free Krylov) are always tested; the external solvers (CUDA, AMGCL,
// Ginkgo, cuDSS) are included only when their YABI_USE_* flag is on (via a
// YABI_HAVE_* define).

#include <yabi/bdf_integrator.hpp>
#include <yabi/cppad_rhs.hpp>
#include <yabi/detail/dense_linear_solver.ipp>
#include <yabi/matrix_free_solver.hpp>

#ifdef YABI_HAVE_CUDA
#include <yabi/cuda_linear_solver.hpp>
#endif
#ifdef YABI_HAVE_AMGCL
#include <yabi/amgcl_linear_solver.hpp>
#endif
#ifdef YABI_HAVE_GINKGO
#include <yabi/ginkgo_linear_solver.hpp>
#endif
#ifdef YABI_HAVE_CUDSS
#include <yabi/cudss_linear_solver.hpp>
#endif

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

double y0[8] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057};
double abs_err[8] = {1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10};

// Integrate HIRES with a freshly-built solver (the integrator takes ownership).
bool integrate(const std::function<std::unique_ptr<yabi::LinearSolver<double>>()> &make_solver,
               std::vector<double> &y) {
    CppAD::ADFun<double> fun = makeHiresTape();
    yabi::BDFIntegrator<double> it(fun);
    it.setLinearSolver(make_solver().release()); // integrator owns it
    if (!it.resetIntegration(y0, 0.01, 1e-8, abs_err, false, false, NULL, NULL, NULL, NULL, 0) ||
        !it.setAlgorithmSettings(5, 1e-16, true, 1) || !it.performIntegration(0.1, 1000000))
        return false;
    y.resize(8);
    return it.getSolutionVector(y.data());
}

void checkAgainst(const std::string &name, const std::vector<double> &ref, bool ok,
                  const std::vector<double> &y) {
    if (!ok) {
        check(false, name + ": integration failed");
        return;
    }
    double d = 0.0;
    for (std::size_t i = 0; i < 8; ++i)
        d = std::max(d, std::fabs(ref[i] - y[i]));
    std::cout << "  " << name << ": max|dense - solver| = " << d << "\n";
    check(d < 1e-9, name + ": agrees with dense");
}

} // namespace

int main() {
    std::cout << "T42 solver agreement matrix (HIRES t=0.1)\n";

    // Dense reference.
    std::vector<double> ref;
    check(integrate([]() { return std::make_unique<yabi::DenseLUSolver<double>>(); }, ref),
          "dense reference integration");

    // Matrix-free Krylov (always available in the CppAD build).
    {
        std::vector<double> y;
        bool ok = integrate(
            []() {
                auto s = std::make_unique<yabi::GmresLinearSolver<double>>();
                s->setPreconditioner(yabi::KrylovPreconditioner::Direct);
                s->setTolerance(1e-12, 0.0, 200, 30);
                return s;
            },
            y);
        checkAgainst("GMRES (matrix-free)", ref, ok, y);
    }

#ifdef YABI_HAVE_CUDA
    {
        std::vector<double> y;
        bool ok = integrate([]() { return std::make_unique<yabi::CudaLinearSolver<double>>(); }, y);
        checkAgainst("cuSOLVER", ref, ok, y);
    }
#endif
#ifdef YABI_HAVE_AMGCL
    {
        std::vector<double> y;
        bool ok = integrate([]() { return std::make_unique<yabi::AmgclLinearSolver<double>>(); }, y);
        checkAgainst("AMGCL", ref, ok, y);
    }
#endif
#ifdef YABI_HAVE_GINKGO
    {
        std::vector<double> y;
        bool ok = integrate([]() { return std::make_unique<yabi::GinkgoLinearSolver<double>>(); }, y);
        checkAgainst("Ginkgo", ref, ok, y);
    }
#endif
#ifdef YABI_HAVE_CUDSS
    {
        std::vector<double> y;
        bool ok = integrate([]() { return std::make_unique<yabi::CudssLinearSolver<double>>(); }, y);
        checkAgainst("cuDSS", ref, ok, y);
    }
#endif

    if (failures) {
        std::cerr << failures << " solver-matrix check(s) failed\n";
        return 1;
    }
    std::cout << "PASS: all available solvers agree with dense\n";
    return 0;
}
