/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// T35 — parallel sensitivity on a larger nonlinear parametrized system: a chain
// of coupled nonlinear decay reactions
//     y_0' = -p_0 y_0^2,   y_i' = p_{i-1} y_{i-1} - p_i y_i^2
// (a literature-style reaction-chain model), with every rate p_i a parameter.
// The simultaneous sensitivity over all P parameters is integrated serially and
// with 4 threads; the two must agree to tolerance.

#include <yabi/bdf_integrator.hpp>

#include <cppad/cppad.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {
constexpr int N = 16; // states
constexpr int P = 16; // rate-constant parameters

void makeTape(CppAD::ADFun<double> &fun, const double *y0, const double *p) {
    CppAD::vector<CppAD::AD<double>> x(N + 1 + P);
    for (int i = 0; i < N; ++i)
        x[i] = y0[i];
    x[N] = 0.0;
    for (int i = 0; i < P; ++i)
        x[N + 1 + i] = p[i];
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(N);
    for (int i = 0; i < N; ++i) {
        f[i] = -x[N + 1 + i] * x[i] * x[i];
        if (i > 0)
            f[i] += x[N + i] * x[i - 1];
    }
    fun.Dependent(x, f);
    fun.optimize();
}

bool run(unsigned int threads, std::vector<std::array<double, N>> &dirs) {
    double y0[N], p[P];
    for (int i = 0; i < N; ++i)
        y0[i] = (i == 0) ? 1.0 : 0.0;
    for (int i = 0; i < P; ++i)
        p[i] = 0.1 * (double)(i + 1);

    CppAD::ADFun<double> fun;
    makeTape(fun, y0, p);

    std::vector<double> yseed(P * N, 0.0), pseed(P * P, 0.0);
    std::vector<double *> yseed_ptr(P), pseed_ptr(P), abs_ptr(P);
    std::vector<double> rel_err(P, 1e-10);
    std::vector<std::array<double, N>> abs_arr(P);
    for (int d = 0; d < P; ++d) {
        pseed[d * P + d] = 1.0;
        abs_arr[d].fill(1e-10);
        yseed_ptr[d] = &yseed[d * N];
        pseed_ptr[d] = &pseed[d * P];
        abs_ptr[d] = abs_arr[d].data();
    }

    yabi::BDFIntegrator<double> integrator(fun);
    integrator.setSensitivityThreads(threads);
    yabi::IntegrationOptions<double> opts;
    opts.taping = false;
    opts.direct = false;
    opts.y_seed_list = yseed_ptr.data();
    opts.p_seed_list = pseed_ptr.data();
    opts.relative_error_derivatives = rel_err.data();
    opts.absolut_error_derivatives = abs_ptr.data();
    opts.numOfderivatives = P;

    double abs_err[N];
    for (int i = 0; i < N; ++i)
        abs_err[i] = 1e-10;
    if (!integrator.resetIntegration(y0, 0.0, p, 0.05, 1e-10, abs_err, opts) ||
        !integrator.performIntegration(1.0, 1000000)) {
        std::cerr << "FAIL: integration (threads=" << threads << ")\n";
        return false;
    }

    dirs.resize(P);
    std::vector<double *> dir_ptr(P);
    for (int d = 0; d < P; ++d)
        dir_ptr[d] = dirs[d].data();
    if (!integrator.getSimultaneousDerivativesVector(dir_ptr.data())) {
        std::cerr << "FAIL: getSimultaneousDerivativesVector (threads=" << threads << ")\n";
        return false;
    }
    return true;
}

} // namespace

int main() {
    std::cout << "T35 parallel sensitivity (nonlinear chain, N=" << N << ", P=" << P << ")\n";

    std::vector<std::array<double, N>> serial, parallel;
    if (!run(1, serial))
        return 1;
    if (!run(4, parallel))
        return 1;

    double maxdiff = 0.0;
    for (int d = 0; d < P; ++d)
        for (int i = 0; i < N; ++i)
            maxdiff = std::max(maxdiff, std::fabs(serial[d][i] - parallel[d][i]));
    std::cout << "  max|serial - parallel| = " << maxdiff << "\n";

    if (maxdiff < 1e-12) {
        std::cout << "PASS: parallel sensitivity matches serial\n";
        return 0;
    }
    std::cerr << "FAIL: parallel sensitivity diverged\n";
    return 1;
}
