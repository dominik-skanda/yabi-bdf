// T36c — parallel multi-instance GPU solves.
//
// Runs K BDFIntegrator + CudaLinearSolver instances concurrently on K threads,
// each with its own CUDA stream + cuSOLVER handle, and verifies every one
// reproduces the CPU dense-LAPACK reference.  Built only when YABI_USE_CUDA=ON.
//
// NOTE: CppAD tape *recording* (Independent/Dependent) is not thread-safe, so
// every integrator is constructed on the main thread before the workers start;
// each worker only runs resetIntegration/performIntegration on its own, already
// independent integrator.
#include <yabi/bdf_integrator.hpp>
#include <yabi/cppad_rhs.hpp>
#include <yabi/cuda_linear_solver.hpp>
#include <yabi/detail/dense_linear_solver.ipp>

#include <cppad/cppad.hpp>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <thread>
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

constexpr double hires_y0[8] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0057};
constexpr double hires_abs_err[8] = {1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10};

// One concurrent worker: runs its own (pre-built) integrator + GPU solver.
void worker(int id, yabi::BDFIntegrator<double> &integrator, const std::vector<double> &reference,
            std::atomic<int> *fails) {
    double yy[8];
    if (!integrator.resetIntegration(const_cast<double *>(hires_y0), 0.01, 1e-8,
                                     const_cast<double *>(hires_abs_err), false, false, NULL, NULL,
                                     NULL, NULL, 0) ||
        !integrator.setAlgorithmSettings(5, 1e-16, true, 1) ||
        !integrator.performIntegration(0.1, 1000000) ||
        !integrator.getSolutionVector(yy)) {
        std::cerr << "worker " << id << ": integration FAILED\n";
        ++*fails;
        return;
    }
    double maxdiff = 0.0;
    for (std::size_t i = 0; i < 8; ++i)
        maxdiff = std::max(maxdiff, std::fabs(yy[i] - reference[i]));
    if (maxdiff > 1e-9) {
        std::cerr << "worker " << id << ": diverged from reference (max " << maxdiff << ")\n";
        ++*fails;
    }
}

} // namespace

int main() {
    std::cout << "T36c parallel multi-instance GPU solves (HIRES t=0.1)\n";

    // CPU reference (single-threaded, dense LAPACK).
    CppAD::ADFun<double> fun = makeHiresTape();
    yabi::BDFIntegrator<double> ref(fun);
    ref.setLinearSolver(new yabi::DenseLUSolver<double>());
    std::vector<double> reference(8);
    if (!ref.resetIntegration(const_cast<double *>(hires_y0), 0.01, 1e-8,
                              const_cast<double *>(hires_abs_err), false, false, NULL, NULL, NULL,
                              NULL, 0) ||
        !ref.setAlgorithmSettings(5, 1e-16, true, 1) ||
        !ref.performIntegration(0.1, 1000000) ||
        !ref.getSolutionVector(reference.data())) {
        std::cerr << "FAIL: CPU reference integration\n";
        return 1;
    }

    // Build all GPU integrators on the main thread (CppAD recording is not
    // thread-safe); the BDFIntegrator(AFun&) convenience constructor deep-copies
    // the tape.  Each integrator is then MOVED into its worker thread so it is
    // used AND destroyed on that same thread (CppAD's thread_alloc is per-thread).
    constexpr int K = 4;
    std::atomic<int> fails{0};
    std::vector<std::thread> threads;
    for (int k = 0; k < K; ++k) {
        auto integ = std::make_unique<yabi::BDFIntegrator<double>>(fun);
        integ->setLinearSolver(new yabi::CudaLinearSolver<double>());
        threads.emplace_back(
            [k, integ = std::move(integ), &reference, &fails]() mutable {
                worker(k, *integ, reference, &fails);
            });
    }
    for (auto &t : threads)
        t.join();

    if (fails == 0) {
        std::cout << "PASS: " << K << " concurrent GPU instances all match the CPU reference\n";
        return 0;
    }
    std::cerr << "FAIL: " << fails << " concurrent instance(s) diverged\n";
    return 1;
}
