/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// T43 — Pollution problem (CWI "Test Set for IVP Solvers", pollu): a
// 20-species atmospheric chemistry system (chemical ozone model), autonomous
// and stiff (rate constants span 1e-8 .. 4e11).  Reference solution at t = 60
// transcribed from pollu.f (true-double-precision RADAU5 on a Cray C90,
// rtol = atol = 1.1e-18).  Rate constants k1..k25, reaction rates r(1..25),
// and the 20 right-hand sides are reproduced exactly; the Jacobian is obtained
// by CppAD (no hand-written Jacobian, avoiding transcription risk).

#include <yabi/bdf_integrator.hpp>

#include <cppad/cppad.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

constexpr std::size_t n = 20;

// Rate constants (pollu.f, Fortran D notation decoded).
const double k1 = 0.35, k2 = 26.6, k3 = 12300.0, k4 = 0.86e-3, k5 = 0.82e-3;
const double k6 = 15000.0, k7 = 0.13e-3, k8 = 24000.0, k9 = 16500.0, k10 = 9000.0;
const double k11 = 0.022, k12 = 12000.0, k13 = 1.88, k14 = 16300.0, k15 = 4.8e7;
const double k16 = 0.35e-3, k17 = 0.0175, k18 = 1.0e8, k19 = 4.44e11, k20 = 1240.0;
const double k21 = 2.1, k22 = 5.78, k23 = 0.0474, k24 = 1780.0, k25 = 3.12;

// Reference solution at t = 60 (pollu.f solut).
const double ref[n] = {
    5.646255480022769e-2, 1.342484130422339e-1, 4.139734331099427e-9, 5.523140207484359e-3,
    2.018977262302196e-7, 1.464541863493966e-7, 7.784249118997964e-2, 3.245075353396018e-1,
    7.494013383880406e-3, 1.622293157301561e-8, 1.135863833257075e-8, 2.230505975721359e-3,
    2.087162882798630e-4, 1.396921016840158e-5, 8.964884856898295e-3, 4.352846369330103e-18,
    6.899219696263405e-3, 1.007803037365946e-4, 1.772146513969984e-6, 5.682943292316392e-5};

} // namespace

int main() {
    CppAD::vector<CppAD::AD<double>> x(n);
    x[1] = 0.2;   // y2
    x[3] = 0.04;  // y4
    x[6] = 0.1;   // y7
    x[7] = 0.3;   // y8
    x[8] = 0.01;  // y9
    x[16] = 0.007; // y17
    CppAD::Independent(x);

    // Reaction rates r(1..25) -> r[0..24].
    CppAD::AD<double> r[25];
    r[0] = k1 * x[0];
    r[1] = k2 * x[1] * x[3];
    r[2] = k3 * x[4] * x[1];
    r[3] = k4 * x[6];
    r[4] = k5 * x[6];
    r[5] = k6 * x[6] * x[5];
    r[6] = k7 * x[8];
    r[7] = k8 * x[8] * x[5];
    r[8] = k9 * x[10] * x[1];
    r[9] = k10 * x[10] * x[0];
    r[10] = k11 * x[12];
    r[11] = k12 * x[9] * x[1];
    r[12] = k13 * x[13];
    r[13] = k14 * x[0] * x[5];
    r[14] = k15 * x[2];
    r[15] = k16 * x[3];
    r[16] = k17 * x[3];
    r[17] = k18 * x[15];
    r[18] = k19 * x[15];
    r[19] = k20 * x[16] * x[5];
    r[20] = k21 * x[18];
    r[21] = k22 * x[18];
    r[22] = k23 * x[0] * x[3];
    r[23] = k24 * x[18] * x[0];
    r[24] = k25 * x[19];

    CppAD::vector<CppAD::AD<double>> f(n);
    f[0] = -r[0] - r[9] - r[13] - r[22] - r[23] + r[1] + r[2] + r[8] + r[10] + r[11] + r[21] +
           r[24];
    f[1] = -r[1] - r[2] - r[8] - r[11] + r[0] + r[20];
    f[2] = -r[14] + r[0] + r[16] + r[18] + r[21];
    f[3] = -r[1] - r[15] - r[16] - r[22] + r[14];
    f[4] = -r[2] + 2.0 * r[3] + r[5] + r[6] + r[12] + r[19];
    f[5] = -r[5] - r[7] - r[13] - r[19] + r[2] + 2.0 * r[17];
    f[6] = -r[3] - r[4] - r[5] + r[12];
    f[7] = r[3] + r[4] + r[5] + r[6];
    f[8] = -r[6] - r[7];
    f[9] = -r[11] + r[6] + r[8];
    f[10] = -r[8] - r[9] + r[7] + r[10];
    f[11] = r[8];
    f[12] = -r[10] + r[9];
    f[13] = -r[12] + r[11];
    f[14] = r[13];
    f[15] = -r[17] - r[18] + r[15];
    f[16] = -r[19];
    f[17] = r[19];
    f[18] = -r[20] - r[21] - r[23] + r[22] + r[24];
    f[19] = -r[24] + r[23];

    CppAD::ADFun<double> fun;
    fun.Dependent(x, f);
    fun.optimize();

    yabi::BDFIntegrator<double> it(fun);
    double y0[n] = {0, 0.2, 0, 0.04, 0, 0, 0.1, 0.3, 0.01, 0, 0, 0, 0, 0, 0, 0, 0.007, 0, 0, 0};
    double atol[n];
    std::fill(atol, atol + n, 1e-12);

    if (!it.resetIntegration(y0, 10.0, 1e-10, atol, false, false, NULL, NULL, NULL, NULL, 0) ||
        !it.setAlgorithmSettings(5, 1e-16, true, 1) || !it.performIntegration(60.0, 10000000)) {
        std::cerr << "FAIL: POLLU integration did not converge\n";
        return 1;
    }
    double y[n];
    if (!it.getSolutionVector(y)) {
        std::cerr << "FAIL: POLLU getSolutionVector\n";
        return 1;
    }

    double max_rel = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        max_rel = std::max(max_rel, std::fabs(y[i] - ref[i]) / (1.0 + std::fabs(ref[i])));
    std::cout << "POLLU max relative error vs CWI reference = " << max_rel << "\n";

    if (max_rel < 1e-6) {
        std::cout << "PASS: POLLU matches the CWI reference solution\n";
        return 0;
    }
    std::cerr << "FAIL: POLLU diverged from the CWI reference\n";
    return 1;
}
