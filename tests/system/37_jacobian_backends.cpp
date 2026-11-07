// T33 — selectable Jacobian backends (CppAD / HIRES).
//
// Proves that the explicit JacobianBackend seam reproduces the same HIRES
// trajectory for Dense / Sparse / MatrixFree / Auto, and that an explicit
// backend the RHS + solver cannot satisfy fails loudly instead of silently
// downgrading to dense.

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

// HIRES (Hairer & Wanner IV.4), the stiff benchmark shared with T31/T32.
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

struct RunResult {
    bool ok = false;
    std::vector<double> y;
};

// `krylov` installs a GMRES solver before selecting the backend, so that the
// MatrixFree backend has a matrix-free solver to use.
RunResult integrate(yabi::JacobianBackend backend, bool krylov) {
    CppAD::ADFun<double> fun = makeHiresTape();
    yabi::BDFIntegrator<double> integrator(fun);

    if (krylov) {
        auto *solver = new yabi::GmresLinearSolver<double>();
        solver->setPreconditioner(yabi::KrylovPreconditioner::Direct);
        solver->setTolerance(1e-12, 0.0, 200, 30);
        integrator.setLinearSolver(solver);
    }
    integrator.setJacobianBackend(backend);

    double y0[8] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057};
    double abs_err[8] = {1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10};

    RunResult result;
    if (!integrator.resetIntegration(y0, 0.01, 1e-8, abs_err, false, false, NULL, NULL, NULL,
                                     NULL, 0))
        return result;
    if (!integrator.setAlgorithmSettings(5, 1e-16, true, 1))
        return result;
    if (!integrator.performIntegration(0.1, 1000000))
        return result;

    result.y.resize(8);
    if (!integrator.getSolutionVector(result.y.data()))
        return result;
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

} // namespace

int main() {
    std::cout << "T33 selectable Jacobian backends (CppAD/HIRES)\n";

    const RunResult reference = integrate(yabi::JacobianBackend::Dense, false);
    check(reference.ok, "dense reference integration succeeds");

    const RunResult dense = integrate(yabi::JacobianBackend::Dense, false);
    const RunResult sparse = integrate(yabi::JacobianBackend::Sparse, false);
    const RunResult mfree = integrate(yabi::JacobianBackend::MatrixFree, true);
    const RunResult auto_ = integrate(yabi::JacobianBackend::Auto, false);

    check(dense.ok, "Dense integration succeeds");
    check(sparse.ok, "Sparse integration succeeds");
    check(mfree.ok, "MatrixFree integration succeeds");
    check(auto_.ok, "Auto integration succeeds");

    check(maxDifference(dense.y, reference.y) < 1e-9, "Dense == reference");
    check(maxDifference(sparse.y, reference.y) < 1e-9, "Sparse == reference");
    check(maxDifference(mfree.y, reference.y) < 1e-9, "MatrixFree == reference");
    check(maxDifference(auto_.y, reference.y) < 1e-9, "Auto == reference");

    // An explicit backend the solver cannot satisfy fails loudly.
    const RunResult no_krylov = integrate(yabi::JacobianBackend::MatrixFree, false);
    check(!no_krylov.ok, "MatrixFree without a Krylov solver fails loudly");

    if (failures != 0) {
        std::cerr << failures << " Jacobian-backend check(s) failed" << std::endl;
        return 1;
    }
    std::cout << "PASS: Dense/Sparse/MatrixFree/Auto agree and fail loudly when unsatisfiable\n";
    return 0;
}
