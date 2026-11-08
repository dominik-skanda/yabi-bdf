/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// T43 — Beam problem (CWI "Test Set for IVP Solvers", beam): a semi-discretized
// 1D beam of dimension 80 (N=40 angles + 40 angular velocities), non-autonomous
// through a point load applied for t <= pi and removed for t > pi.  The RHS
// requires solving a tridiagonal system at every evaluation (done symbolically
// inside the CppAD tape, so the Jacobian is automatic).  Reference solution at
// t = 5 transcribed from beam.f (RADAU on an AlphaServer DS20E, rtol=atol=1.1e-18).

#include <yabi/bdf_integrator.hpp>

#include <cppad/cppad.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

constexpr int N = 40;
constexpr std::size_t dim = 2 * N; // 80
constexpr double kPi = 3.14159265358979323846;
constexpr double NSQ = (double)(N * N);      // 1600
constexpr double NQUATR = NSQ * NSQ;         // 2.56e6

// Reference at t = 5 (beam.f solut), all 80 components: angles y[0..39],
// velocities y[40..79].
const double ref[dim] = {
    -5.792366591285007e-3, -1.695298550721735e-2, -2.769103312973094e-2, -3.800815655879158e-2,
    -4.790616859743763e-2, -5.738710435274594e-2, -6.645327313454617e-2, -7.510730581979037e-2,
    -8.335219765414992e-2, -9.119134654647947e-2, -9.862858700132091e-2, -0.1056682200378002,
    -0.1123150395409595, -0.1185743552727245, -0.1244520128755561, -0.1299544113264161,
    -0.1350885180610398, -0.1398618819194422, -0.1442826441015242, -0.1483595472463012,
    -0.1521019429001447, -0.1555197978061129, -0.1586236993420229, -0.1614248603702127,
    -0.1639351238193223, -0.1661669673440852, -0.1681335081778558, -0.1698485080602243,
    -0.1713263782440888, -0.1725821847462537, -0.1736316537975654, -0.1744911773840049,
    -0.1751778187863392, -0.1757093178712902, -0.1761040960228576, -0.1763812607175549,
    -0.1765606097564671, -0.1766626352260565, -0.1767085270807460, -0.1767201761075488,
    0.03747362681329794,   0.1099117880217593,    0.1798360474312799,    0.2472427305442391,
    0.3121293820596567,    0.3744947377019500,    0.4343386073492798,    0.4916620354601748,
    0.5464677854586807,    0.5987609702624270,    0.6485493611110740,    0.6958435169132503,
    0.7406572668520808,    0.7830081747813241,    0.8229176659201515,    0.8604110305190560,
    0.8955175502377805,    0.9282708263127953,    0.9587089334522034,    0.9868747821728363,
    1.012816579961883,     1.036587736679858,     1.058246826481355,     1.077857811433353,
    1.095490222005369,     1.111219164294898,     1.125125269286501,     1.137294526609229,
    1.147818025153607,     1.156792132004482,     1.164318845130183,     1.170505992596124,
    1.175467424299550,     1.179323003228859,     1.182198586299667,     1.184226111223146,
    1.185543909805575,     1.186297084203716,     1.186637618908127,     1.186724615113034};

} // namespace

int main() {
    // Tape domain = state (80) + time (1).  The last slot is the time t.
    CppAD::vector<CppAD::AD<double>> x(dim + 1);
    for (std::size_t i = 0; i < dim; ++i)
        x[i] = 0.0; // initial state (all zero) — beam starts from rest
    x[dim] = 0.0;   // time
    CppAD::Independent(x);

    CppAD::AD<double> t = x[dim];

    // sin/cos of angle differences: sth[i]=sin(th[i]-th[i-1]), cth[i]=cos(...)
    // for i=1..N-1 (Fortran I=2..N).  Index 0 is unused.
    CppAD::AD<double> sth[N], cth[N];
    sth[0] = 0.0;
    cth[0] = 0.0;
    for (int i = 1; i < N; ++i) {
        CppAD::AD<double> diff = x[i] - x[i - 1];
        sth[i] = CppAD::sin(diff);
        cth[i] = CppAD::cos(diff);
    }

    // Forcing (point load), active only for t <= pi.
    CppAD::AD<double> fabs = 1.5 * CppAD::sin(t) * CppAD::sin(t);
    CppAD::AD<double> fx = -fabs;
    CppAD::AD<double> fy = fabs;
    CppAD::AD<double> active =
        CppAD::CondExpLe(t, CppAD::AD<double>(kPi), CppAD::AD<double>(1.0), CppAD::AD<double>(0.0));

    // Right-hand side of the linear system V.
    CppAD::AD<double> V[N];
    for (int i = 0; i < N; ++i) {
        CppAD::AD<double> term1;
        if (i == 0)
            term1 = (-3.0 * x[0] + x[1]) * NQUATR;
        else if (i == N - 1)
            term1 = (x[N - 2] - x[N - 1]) * NQUATR;
        else
            term1 = (x[i - 1] - 2.0 * x[i] + x[i + 1]) * NQUATR;
        CppAD::AD<double> term2 =
            active * NSQ * (fy * CppAD::cos(x[i]) - fx * CppAD::sin(x[i]));
        V[i] = term1 + term2;
    }

    // W = D*V, then add velocity^2.
    CppAD::AD<double> W[N];
    W[0] = sth[1] * V[1];
    for (int i = 1; i < N - 1; ++i)
        W[i] = -sth[i] * V[i - 1] + sth[i + 1] * V[i + 1];
    W[N - 1] = -sth[N - 1] * V[N - 2];
    for (int i = 0; i < N; ++i)
        W[i] = W[i] + x[N + i] * x[N + i]; // x[N+i] = velocity

    // Solve the tridiagonal system C*W = W (Thomas elimination).
    CppAD::AD<double> alpha[N], beta[N - 1];
    alpha[0] = 1.0;
    for (int i = 1; i < N; ++i) {
        alpha[i] = 2.0;
        beta[i - 1] = -cth[i];
    }
    alpha[N - 1] = 3.0;
    for (int i = N - 2; i >= 0; --i) {
        CppAD::AD<double> q = beta[i] / alpha[i + 1];
        W[i] = W[i] - W[i + 1] * q;
        alpha[i] = alpha[i] - beta[i] * q;
    }
    W[0] = W[0] / alpha[0];
    for (int i = 1; i < N; ++i)
        W[i] = (W[i] - beta[i - 1] * W[i - 1]) / alpha[i];

    // U = C*V + D*W (the acceleration).
    CppAD::AD<double> U[N];
    U[0] = V[0] - cth[1] * V[1] + sth[1] * W[1];
    for (int i = 1; i < N - 1; ++i)
        U[i] = 2.0 * V[i] - cth[i] * V[i - 1] - cth[i + 1] * V[i + 1] - sth[i] * W[i - 1] +
               sth[i + 1] * W[i + 1];
    U[N - 1] = 3.0 * V[N - 1] - cth[N - 1] * V[N - 2] - sth[N - 1] * W[N - 2];

    CppAD::vector<CppAD::AD<double>> f(dim);
    for (int i = 0; i < N; ++i) {
        f[i] = x[N + i]; // d(angle)/dt = velocity
        f[N + i] = U[i]; // d(velocity)/dt = acceleration
    }

    CppAD::ADFun<double> fun;
    fun.Dependent(x, f);
    fun.optimize();

    yabi::BDFIntegrator<double> it(fun);
    std::vector<double> y0(dim, 0.0), atol(dim, 1e-13);

    if (!it.resetIntegration(y0.data(), 0.0, 0.5, 1e-11, atol.data(), false, false, NULL, NULL,
                             NULL, NULL, 0) ||
        !it.setAlgorithmSettings(5, 1e-16, true, 1) || !it.performIntegration(5.0, 10000000)) {
        std::cerr << "FAIL: BEAM integration did not converge\n";
        return 1;
    }
    std::vector<double> y(dim);
    if (!it.getSolutionVector(y.data())) {
        std::cerr << "FAIL: BEAM getSolutionVector\n";
        return 1;
    }

    double max_rel = 0.0;
    for (std::size_t i = 0; i < dim; ++i)
        max_rel = std::max(max_rel, std::fabs(y[i] - ref[i]) / (1.0 + std::fabs(ref[i])));
    std::cout << "BEAM max relative error vs CWI reference = " << max_rel << "\n";

    if (max_rel < 1e-6) {
        std::cout << "PASS: BEAM matches the CWI reference solution\n";
        return 0;
    }
    std::cerr << "FAIL: BEAM diverged from the CWI reference\n";
    return 1;
}
