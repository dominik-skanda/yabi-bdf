// dump_trajectories.cpp — integrate every benchmark and dump (t, y0, y1, ...) to
// trajectories/<name>.csv so a plotting script can render the full suite.
//
// One file per benchmark; columns are "t,y0,y1,...". Sample times are chosen per
// problem (linear for normal ranges, log-spaced for Robertson/E5 which span decades).
#include <yabi/bdf_integrator.hpp>
#include <cppad/cppad.hpp>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::vector<double> lin(double t0, double t1, int n) {
    std::vector<double> ts;
    for (int i = 1; i <= n; i++) ts.push_back(t0 + (t1 - t0) * i / (double)n);
    return ts;
}
static std::vector<double> logspace(double t0, double t1, int n) {
    std::vector<double> ts;
    double a = std::log10(t0), b = std::log10(t1);
    for (int i = 1; i <= n; i++) ts.push_back(std::pow(10.0, a + (b - a) * i / (double)n));
    return ts;
}

// Integrate rhs from y0 to each sample time and write "t,y0,y1,..." lines.
// param is empty for autonomous RHS, else the parameter vector (time+parameter form).
static void dump(const std::string &name, CppAD::ADFun<double> &rhs, const std::vector<double> &y0,
                 const std::vector<double> &abs_err, double t_end, unsigned max_order,
                 const std::vector<double> &ts, const std::vector<double> &param = {}) {
    yabi::BDFIntegrator<double> it(rhs);
    unsigned n = (unsigned)y0.size();
    bool ok;
    if (param.empty()) {
        ok = it.resetIntegration(const_cast<double *>(y0.data()), t_end, 1e-10,
                                 const_cast<double *>(abs_err.data()), false, false, NULL, NULL, NULL, NULL, 0);
    } else {
        ok = it.resetIntegration(const_cast<double *>(y0.data()), 0.0, const_cast<double *>(param.data()), 1.0, 1e-10,
                                 const_cast<double *>(abs_err.data()), false, false, NULL, NULL, NULL, NULL, 0);
    }
    if (!ok) { std::cerr << name << ": reset failed\n"; return; }
    it.setAlgorithmSettings(max_order, std::numeric_limits<double>::epsilon(), true, 1);
    std::ofstream f("trajectories/" + name + ".csv");
    f << "t";
    for (unsigned i = 0; i < n; i++) f << ",y" << i;
    f << "\n";
    f << "0";
    for (unsigned i = 0; i < n; i++) f << "," << y0[i];
    f << "\n";
    std::vector<double> y(n);
    for (double t : ts) {
        if (!it.performIntegration(t, 10000000)) { std::cerr << name << ": integrate to " << t << " failed\n"; break; }
        if (!it.getSolutionVector(y.data())) break;
        f << t;
        for (unsigned i = 0; i < n; i++) f << "," << y[i];
        f << "\n";
    }
    std::cout << "  " << name << " -> trajectories/" << name << ".csv\n";
}

// --- tape builders -------------------------------------------------------------
static CppAD::ADFun<double> tape_brusselator() {
    CppAD::vector<CppAD::AD<double>> x(2); x[0] = 1; x[1] = 2; CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(2); f[0] = 1.0 + x[0] * x[0] * x[1] - 4.0 * x[0]; f[1] = 3.0 * x[0] - x[0] * x[0] * x[1];
    CppAD::ADFun<double> r; r.Dependent(x, f); r.optimize(); return r;
}
static CppAD::ADFun<double> tape_robertson() {
    CppAD::vector<CppAD::AD<double>> x(3); x[0] = 1; x[1] = 0; x[2] = 0; CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(3);
    f[0] = -0.04 * x[0] + 1e4 * x[1] * x[2]; f[1] = 0.04 * x[0] - 1e4 * x[1] * x[2] - 3e7 * x[1] * x[1]; f[2] = 3e7 * x[1] * x[1];
    CppAD::ADFun<double> r; r.Dependent(x, f); r.optimize(); return r;
}
static CppAD::ADFun<double> tape_vanderpol() {
    const double mu = 1000.0;
    CppAD::vector<CppAD::AD<double>> x(2); x[0] = 2; x[1] = 0; CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(2); f[0] = x[1]; f[1] = mu * (1 - x[0] * x[0]) * x[1] - x[0];
    CppAD::ADFun<double> r; r.Dependent(x, f); r.optimize(); return r;
}
static CppAD::ADFun<double> tape_hires() {
    CppAD::vector<CppAD::AD<double>> x(8); x[0] = 1; x[7] = 0.0057; CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(8);
    f[0] = -1.71 * x[0] + 0.43 * x[1] + 8.32 * x[2] + 0.0007; f[1] = 1.71 * x[0] - 8.75 * x[1];
    f[2] = -10.03 * x[2] + 0.43 * x[3] + 0.035 * x[4]; f[3] = 8.32 * x[1] + 1.71 * x[2] - 1.12 * x[3];
    f[4] = -1.745 * x[4] + 0.43 * x[5] + 0.43 * x[6]; f[5] = -280 * x[5] * x[7] + 0.69 * x[3] + 1.71 * x[4] - 0.43 * x[5] + 0.69 * x[6];
    f[6] = 280 * x[5] * x[7] - 1.81 * x[6]; f[7] = -f[6];
    CppAD::ADFun<double> r; r.Dependent(x, f); r.optimize(); return r;
}
static CppAD::ADFun<double> tape_prothero() {
    CppAD::vector<CppAD::AD<double>> x(2); x[0] = 0.5; x[1] = 0; CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(2); f[0] = 2.0 * (x[0] - x[0] * x[0]) - x[1]; f[1] = x[0] * x[0] - x[1];
    CppAD::ADFun<double> r; r.Dependent(x, f); r.optimize(); return r;
}
static CppAD::ADFun<double> tape_oregonator() {
    CppAD::vector<CppAD::AD<double>> x(3); x[0] = 1; x[1] = 1; x[2] = 1; CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(3);
    f[0] = 77.27 * (x[1] + x[0] * (1.0 - 8.375e-6 * x[0] - x[1])); f[1] = (x[2] - (1.0 + x[0]) * x[1]) / 77.27; f[2] = 0.161 * (x[0] - x[2]);
    CppAD::ADFun<double> r; r.Dependent(x, f); r.optimize(); return r;
}
static CppAD::ADFun<double> tape_pendulum() {
    const double mu = 10.0;
    CppAD::vector<CppAD::AD<double>> x(2); x[0] = M_PI / 2; x[1] = 0; CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(2); f[0] = x[1]; f[1] = -mu * x[1] - 9.81 * sin(x[0]);
    CppAD::ADFun<double> r; r.Dependent(x, f); r.optimize(); return r;
}
static CppAD::ADFun<double> tape_polanyi() {
    const int n = 20;
    CppAD::vector<CppAD::AD<double>> x(n); for (int i = 0; i < n; i++) x[i] = 1.0 + 0.01 * i; CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(n);
    for (int i = 0; i < n; i++) {
        CppAD::AD<double> left = (i > 0) ? x[i - 1] : CppAD::AD<double>(0.0);
        CppAD::AD<double> right = (i < n - 1) ? x[i + 1] : CppAD::AD<double>(0.0);
        f[i] = 0.01 * (left - 2.0 * x[i] + right) - 0.1 * x[i];
    }
    CppAD::ADFun<double> r; r.Dependent(x, f); r.optimize(); return r;
}
static CppAD::ADFun<double> tape_network() {
    CppAD::vector<CppAD::AD<double>> x(10);
    x[0] = 1; x[7] = 0.04; x[8] = 1e4; x[9] = 3e7; CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(6);
    f[0] = -x[7] * x[0] + x[8] * x[1] * x[2]; f[1] = x[7] * x[0] - x[8] * x[1] * x[2] - x[9] * x[1] * x[1]; f[2] = x[9] * x[1] * x[1];
    f[3] = x[9] * x[1] * x[1] - 0.1 * x[3]; f[4] = 0.1 * x[3] - 5 * x[4]; f[5] = 5 * x[4] - 0.01 * x[5];
    CppAD::ADFun<double> r; r.Dependent(x, f); r.optimize(); return r;
}
static CppAD::ADFun<double> tape_e5() {
    CppAD::vector<CppAD::AD<double>> x(3); x[0] = 1.76e-3; CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(3);
    f[0] = -7.89e-10 * x[0] - 1.1e7 * x[0] * x[2]; f[1] = 7.89e-10 * x[0] - 1.13e9 * x[1] * x[2];
    f[2] = 7.89e-10 * x[0] - 1.1e7 * x[0] * x[2] - 1.13e9 * x[1] * x[2];
    CppAD::ADFun<double> r; r.Dependent(x, f); r.optimize(); return r;
}
static CppAD::ADFun<double> tape_oscillator() {
    CppAD::vector<CppAD::AD<double>> x(2); x[0] = 1; x[1] = 0; CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(2); f[0] = x[1]; f[1] = -x[0];
    CppAD::ADFun<double> r; r.Dependent(x, f); r.optimize(); return r;
}

int main() {
    fs::create_directories("trajectories");
    std::cout << "Dumping trajectories:\n";

    { auto r = tape_brusselator(); dump("brusselator", r, {1, 2}, {1e-10, 1e-10}, 5, 6, lin(0, 5, 200)); }
    { auto r = tape_robertson(); dump("robertson", r, {1, 0, 0}, {1e-10, 1e-14, 1e-10}, 1e4, 6, logspace(1e-2, 1e4, 200)); }
    { auto r = tape_vanderpol(); dump("vanderpol", r, {2, 0}, {1e-8, 1e-8}, 2000, 6, lin(0, 2000, 400)); }
    { auto r = tape_hires(); dump("hires", r, {1, 0, 0, 0, 0, 0, 0, 0.0057}, std::vector<double>(8, 1e-10), 321.812, 6, lin(0, 321.812, 400)); }
    { auto r = tape_prothero(); dump("prothero_robinson", r, {0.5, 0}, {1e-10, 1e-10}, 10, 6, lin(0, 10, 200)); }
    { auto r = tape_oregonator(); dump("oregonator", r, {1, 1, 1}, {1e-10, 1e-10, 1e-10}, 10, 6, lin(0, 10, 400)); }
    { auto r = tape_pendulum(); dump("pendulum", r, {M_PI / 2, 0}, {1e-10, 1e-10}, 5, 6, lin(0, 5, 200)); }
    { auto r = tape_polanyi(); std::vector<double> pw_y0; for (int i = 0; i < 20; i++) pw_y0.push_back(1.0 + 0.01 * i); dump("polanyi_wigner", r, pw_y0, std::vector<double>(20, 1e-6), 10, 6, lin(0, 10, 100)); }
    { auto r = tape_network(); dump("network", r, {1, 0, 0, 0, 0, 0}, {1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10}, 100, 6, lin(0, 100, 200), {0.04, 1e4, 3e7}); }
    { auto r = tape_e5(); dump("e5", r, {1.76e-3, 0, 0}, {1e-18, 1e-24, 1e-18}, 1e6, 6, logspace(1e-1, 1e6, 200)); }
    { auto r = tape_oscillator(); dump("oscillator", r, {1, 0}, {1e-10, 1e-10}, 10, 6, lin(0, 10, 200)); }

    std::cout << "Done.\n";
    return 0;
}
