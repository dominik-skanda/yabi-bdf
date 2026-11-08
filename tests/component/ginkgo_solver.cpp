// T36d — external open-source sparse solver adapter (Ginkgo), cross-checked
// against the dense LAPACK path.  Built only when YABI_USE_GINKGO=ON.
//
// Proves the open solver architecture: GinkgoLinearSolver delegates the CSC
// sparse Newton solve to Ginkgo (BiCGStab, OpenMP backend) through the shared
// ExternalSparseSolver base and reproduces the CPU dense trajectory on HIRES.
#include <yabi/bdf_integrator.hpp>
#include <yabi/cppad_rhs.hpp>
#include <yabi/detail/dense_linear_solver.ipp>
#include <yabi/ginkgo_linear_solver.hpp>

#include <cppad/cppad.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

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

bool integrate(bool ginkgo, std::vector<double> &y) {
    CppAD::ADFun<double> fun = makeHiresTape();
    yabi::BDFIntegrator<double> integrator(fun);
    if (ginkgo)
        integrator.setLinearSolver(new yabi::GinkgoLinearSolver<double>());
    else
        integrator.setLinearSolver(new yabi::DenseLUSolver<double>());

    double y0[8] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057};
    double abs_err[8] = {1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10};
    if (!integrator.resetIntegration(y0, 0.01, 1e-8, abs_err, false, false, NULL, NULL, NULL, NULL,
                                     0) ||
        !integrator.setAlgorithmSettings(5, 1e-16, true, 1) ||
        !integrator.performIntegration(0.01, 1000000))
        return false;
    y.resize(8);
    return integrator.getSolutionVector(y.data());
}

} // namespace

int main() {
    std::cout << "T36d Ginkgo sparse solver vs CPU dense (HIRES t=0.01)\n";

    std::vector<double> cpu, ginkgo;
    if (!integrate(false, cpu)) {
        std::cerr << "FAIL: CPU dense integration\n";
        return 1;
    }
    if (!integrate(true, ginkgo)) {
        std::cerr << "FAIL: Ginkgo integration\n";
        return 1;
    }

    double maxdiff = 0.0;
    for (std::size_t i = 0; i < 8; ++i)
        maxdiff = std::max(maxdiff, std::fabs(cpu[i] - ginkgo[i]));
    std::cout << "  max|CPU-Ginkgo| = " << maxdiff << "\n";

    if (maxdiff < 1e-9) {
        std::cout << "PASS: Ginkgo sparse solve matches the dense LAPACK path\n";
        return 0;
    }
    std::cerr << "FAIL: Ginkgo state solve diverged from the CPU reference\n";
    return 1;
}
