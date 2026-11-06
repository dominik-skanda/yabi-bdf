/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// T32 — matrix-free Newton-Krylov component test.
//
// Proves end-to-end that:
//   * GMRES / FGMRES / BiCGStab drive the state Newton solve through J*v only,
//   * a preconditioned Krylov solve reproduces the direct-solver trajectory,
//   * the state solve never falls back to the direct CSC factorization, and
//   * forward sensitivities still use the explicit J/H path and stay correct
//     when a matrix-free solver is installed.

#include <yabi/bdf_integrator.hpp>
#include <yabi/cppad_rhs.hpp>
#include <yabi/detail/dense_linear_solver.ipp>
#include <yabi/matrix_free_solver.hpp>

#include <cppad/cppad.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
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

// HIRES (Hairer & Wanner IV.4), the stiff benchmark named in the T32 gate.
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

enum class Mode { Dense, Gmres, Fgmres, Bicgstab };

struct RunResult {
    bool ok = false;
    std::vector<double> y;
    std::vector<double> sensitivity;
    unsigned int jacobian_evaluations = 0;
    unsigned int fallbacks = 0;
    bool last_converged = true;
};

RunResult integrate(Mode mode, yabi::KrylovPreconditioner preconditioner, bool sensitivity) {
    CppAD::ADFun<double> fun = makeHiresTape();
    yabi::BDFIntegrator<double> integrator(fun);

    yabi::KrylovSolverBase<double> *krylov = nullptr;
    switch (mode) {
    case Mode::Dense:
        integrator.setLinearSolver(new yabi::DenseLUSolver<double>());
        break;
    case Mode::Gmres:
        krylov = new yabi::GmresLinearSolver<double>();
        break;
    case Mode::Fgmres:
        krylov = new yabi::FgmresLinearSolver<double>();
        break;
    case Mode::Bicgstab:
        krylov = new yabi::BicgstabLinearSolver<double>();
        break;
    }
    if (krylov) {
        krylov->setPreconditioner(preconditioner);
        krylov->setTolerance(1e-12, 0.0, 200, 30);
        integrator.setLinearSolver(krylov);
        check(integrator.getMatrixFreeAssembly(), "installing a Krylov solver enables matrix-free");
    } else {
        check(!integrator.getMatrixFreeAssembly(), "dense solver keeps matrix-free off");
    }

    double y0[8] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057};
    double abs_err[8] = {1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10};

    // Forward sensitivities are generated from the replay tape: the state
    // integration is taped, then forward() solves the variational equations
    // through the explicit J/H path.
    yabi::IntegrationOptions<double> options;
    options.taping = sensitivity;

    RunResult result;
    if (!integrator.resetIntegration(y0, 0.01, 1e-8, abs_err, options))
        return result;
    if (!integrator.setAlgorithmSettings(5, 1e-16, true, 1))
        return result;
    if (!integrator.performIntegration(0.1, 1000000))
        return result;

    result.y.resize(8);
    if (!integrator.getSolutionVector(result.y.data()))
        return result;

    if (sensitivity) {
        double seed[8] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        if (!integrator.forward(1, seed, nullptr, false))
            return result;
        result.sensitivity.resize(8);
        if (!integrator.getForwardSolutionVector(result.sensitivity.data()))
            return result;
    }

    result.jacobian_evaluations = integrator.getJacobianEvaluations();
    if (krylov) {
        result.fallbacks = krylov->fallbackCount();
        result.last_converged = krylov->lastConverged();
    }
    result.ok = true;
    return result;
}

double maxDifference(const std::vector<double> &a, const std::vector<double> &b) {
    if (a.size() != b.size())
        return std::numeric_limits<double>::infinity();
    double diff = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        diff = std::max(diff, std::fabs(a[i] - b[i]));
    return diff;
}

void checkStateSolve(Mode mode, const char *name, yabi::KrylovPreconditioner preconditioner,
                     const RunResult &dense) {
    const RunResult result = integrate(mode, preconditioner, false);
    check(result.ok, std::string(name) + ": integration succeeds");
    if (!result.ok || !dense.ok)
        return;

    const double diff = maxDifference(result.y, dense.y);
    check(diff < 1e-9, std::string(name) + ": state matches the direct solver (diff " +
                           std::to_string(diff) + ")");
    check(result.fallbacks == 0, std::string(name) + ": no direct CSC fallback");
    check(result.last_converged, std::string(name) + ": last Krylov solve converged");
    check(result.jacobian_evaluations == dense.jacobian_evaluations,
          std::string(name) + ": identical Jacobian evaluation count");
    std::cout << "  " << name << ": max|direct-krylov|=" << diff
              << ", fallbacks=" << result.fallbacks << "\n";
}

void checkSensitivity(Mode mode, const char *name, yabi::KrylovPreconditioner preconditioner,
                      const RunResult &dense) {
    const RunResult result = integrate(mode, preconditioner, true);
    check(result.ok, std::string(name) + ": sensitivity integration succeeds");
    if (!result.ok || !dense.ok)
        return;

    const double state_diff = maxDifference(result.y, dense.y);
    const double sensitivity_diff = maxDifference(result.sensitivity, dense.sensitivity);
    check(state_diff < 1e-9, std::string(name) + ": sensitivity run state matches direct");
    check(sensitivity_diff < 1e-7, std::string(name) +
                                       ": forward sensitivity matches the dense J/H path (diff " +
                                       std::to_string(sensitivity_diff) + ")");
    std::cout << "  " << name << ": max|direct-krylov| sensitivity=" << sensitivity_diff << "\n";
}

} // namespace

int main() {
    std::cout << "T32 matrix-free Newton-Krylov (HIRES)\n";

    const RunResult dense = integrate(Mode::Dense, yabi::KrylovPreconditioner::None, false);
    check(dense.ok, "direct reference integration succeeds");

    const RunResult dense_sensitivity =
        integrate(Mode::Dense, yabi::KrylovPreconditioner::None, true);
    check(dense_sensitivity.ok, "direct reference sensitivity integration succeeds");

    // HIRES is very stiff; the exact CSC preconditioner keeps the Krylov
    // iteration short while P*v still goes through the matrix-free operator.
    checkStateSolve(Mode::Gmres, "GMRES/direct-P", yabi::KrylovPreconditioner::Direct, dense);
    checkStateSolve(Mode::Fgmres, "FGMRES/direct-P", yabi::KrylovPreconditioner::Direct, dense);
    checkStateSolve(Mode::Bicgstab, "BiCGSTAB/direct-P", yabi::KrylovPreconditioner::Direct, dense);

    // Jacobi is a genuine approximate preconditioner (the P factorization is
    // not reused as a solve); the Krylov iteration must still converge.
    checkStateSolve(Mode::Gmres, "GMRES/Jacobi", yabi::KrylovPreconditioner::Jacobi, dense);
    checkStateSolve(Mode::Fgmres, "FGMRES/Jacobi", yabi::KrylovPreconditioner::Jacobi, dense);
    checkStateSolve(Mode::Bicgstab, "BiCGSTAB/Jacobi", yabi::KrylovPreconditioner::Jacobi, dense);

    checkSensitivity(Mode::Gmres, "GMRES sensitivity", yabi::KrylovPreconditioner::Jacobi,
                     dense_sensitivity);

    if (failures != 0) {
        std::cerr << failures << " matrix-free check(s) failed" << std::endl;
        return 1;
    }
    std::cout << "PASS: matrix-free state Newton solve and unchanged sensitivity path\n";
    return 0;
}
