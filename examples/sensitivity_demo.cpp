/*
 * YABI sensitivity demo.
 *
 * Model: two linear compartments with two parameters
 *
 *   y0' = -k1 * y0
 *   y1' =  k1 * y0 - k2 * y1
 *
 * The right-hand side is taped with CppAD and handed to yabi::BDFIntegrator.
 * The program demonstrates and checks
 *   1. plain forward integration,
 *   2. forward-mode sensitivities (w.r.t. initial values and parameters),
 *   3. reverse/adjoint sensitivities,
 *   4. simultaneous (variational) sensitivities during the integration.
 *
 * Results are compared against central finite differences of the plain
 * integrator.  The program returns non-zero if a check fails.
 */

#include <yabi/bdf_integrator.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr unsigned int N = 2;
constexpr unsigned int P = 2;
constexpr double T_END = 2.0;
constexpr double REL_TOL = 1e-9;
constexpr double ABS_TOL = 1e-12;
constexpr double MAX_STEP = 0.1;
constexpr unsigned int MAX_STEPS = 1000000;
constexpr double CHECK_TOL = 1e-5;

typedef std::array<double, N> State;
typedef std::array<double, P> Parameter;

CppAD::ADFun<double> make_model() {
    CppAD::vector<CppAD::AD<double> > x(N + 1 + P);
    x[0] = 1.0;  // y0
    x[1] = 0.0;  // y1
    x[2] = 0.0;  // t
    x[3] = 0.7;  // k1
    x[4] = 0.3;  // k2

    CppAD::Independent(x);

    CppAD::vector<CppAD::AD<double> > f(N);
    f[0] = -x[3] * x[0];
    f[1] = x[3] * x[0] - x[4] * x[1];

    CppAD::ADFun<double> rhs;
    rhs.Dependent(x, f);
    rhs.optimize();
    return rhs;
}

State integrate(CppAD::ADFun<double>& model, State y0, Parameter p) {
    yabi::BDFIntegrator<double> integrator(model);
    double absolute_error[N] = {ABS_TOL, ABS_TOL};
    State y;

    if (!integrator.resetIntegration(y0.data(), 0.0, p.data(), MAX_STEP,
                                     REL_TOL, absolute_error, false, false,
                                     NULL, NULL, NULL, NULL, 0)) {
        std::cerr << "resetIntegration failed\n";
        std::exit(2);
    }
    if (!integrator.performIntegration(T_END, MAX_STEPS)) {
        std::cerr << "performIntegration failed\n";
        std::exit(2);
    }
    if (!integrator.getSolutionVector(y.data())) {
        std::cerr << "getSolutionVector failed\n";
        std::exit(2);
    }
    return y;
}

struct Sensitivity {
    State y;
    std::array<State, N> dy_dy0;  // [input][output]
    std::array<State, P> dy_dp;   // [parameter][output]
};

void fill_forward(CppAD::ADFun<double>& model, const State& y0,
                  const Parameter& p, Sensitivity& out) {
    State y0_copy = y0;
    Parameter p_copy = p;
    for (unsigned int i = 0; i < N; ++i) {
        State seed = {{0.0, 0.0}};
        Parameter pseed = {{0.0, 0.0}};
        seed[i] = 1.0;
        yabi::BDFIntegrator<double> integrator(model);
        double absolute_error[N] = {ABS_TOL, ABS_TOL};
        if (!integrator.resetIntegration(y0_copy.data(), 0.0, p_copy.data(),
                                         MAX_STEP, REL_TOL, absolute_error,
                                         true, false, NULL, NULL, NULL, NULL,
                                         0) ||
            !integrator.performIntegration(T_END, MAX_STEPS) ||
            !integrator.forward(1, seed.data(), pseed.data(), false)) {
            std::cerr << "forward sensitivity failed\n";
            std::exit(2);
        }
        State dir;
        integrator.getForwardSolutionVector(dir.data());
        for (unsigned int j = 0; j < N; ++j) out.dy_dy0[i][j] = dir[j];
    }

    for (unsigned int k = 0; k < P; ++k) {
        State seed = {{0.0, 0.0}};
        Parameter pseed = {{0.0, 0.0}};
        pseed[k] = 1.0;
        yabi::BDFIntegrator<double> integrator(model);
        double absolute_error[N] = {ABS_TOL, ABS_TOL};
        if (!integrator.resetIntegration(y0_copy.data(), 0.0, p_copy.data(),
                                         MAX_STEP, REL_TOL, absolute_error,
                                         true, false, NULL, NULL, NULL, NULL,
                                         0) ||
            !integrator.performIntegration(T_END, MAX_STEPS) ||
            !integrator.forward(1, seed.data(), pseed.data(), false)) {
            std::cerr << "forward parameter sensitivity failed\n";
            std::exit(2);
        }
        State dir;
        integrator.getForwardSolutionVector(dir.data());
        for (unsigned int j = 0; j < N; ++j) out.dy_dp[k][j] = dir[j];
    }
}

void fill_reverse(CppAD::ADFun<double>& model, const State& y0,
                  const Parameter& p, Sensitivity& out) {
    State y0_copy = y0;
    Parameter p_copy = p;
    for (unsigned int j = 0; j < N; ++j) {
        yabi::BDFIntegrator<double> integrator(model);
        double absolute_error[N] = {ABS_TOL, ABS_TOL};
        double weight[N] = {0.0, 0.0};
        weight[j] = 1.0;
        if (!integrator.resetIntegration(y0_copy.data(), 0.0, p_copy.data(),
                                         MAX_STEP, REL_TOL, absolute_error,
                                         true, false, NULL, NULL, NULL, NULL,
                                         0) ||
            !integrator.performIntegration(T_END, MAX_STEPS) ||
            !integrator.reverse(1, weight)) {
            std::cerr << "reverse sensitivity failed\n";
            std::exit(2);
        }

        double* adj_species[N];
        double* adj_parameter[P];
        for (unsigned int i = 0; i < N; ++i) adj_species[i] = new double[1];
        for (unsigned int i = 0; i < P; ++i) adj_parameter[i] = new double[1];

        integrator.getReverseSolutionVector(adj_species, adj_parameter);
        for (unsigned int i = 0; i < N; ++i) out.dy_dy0[i][j] = adj_species[i][0];
        for (unsigned int i = 0; i < P; ++i) out.dy_dp[i][j] = adj_parameter[i][0];

        for (unsigned int i = 0; i < N; ++i) delete[] adj_species[i];
        for (unsigned int i = 0; i < P; ++i) delete[] adj_parameter[i];
    }
}

void fill_simultaneous(CppAD::ADFun<double>& model, const State& y0,
                       const Parameter& p, Sensitivity& out) {
    State y0_copy = y0;
    Parameter p_copy = p;
    const unsigned int ndir = N + P;
    std::vector<std::array<double, N> > y_seed(ndir);
    std::vector<std::array<double, P> > p_seed(ndir);
    std::vector<double*> y_seed_ptr(ndir), p_seed_ptr(ndir), abs_ptr(ndir);
    std::vector<std::array<double, N> > abs_error(ndir);
    std::vector<double> rel_error(ndir, REL_TOL);

    for (unsigned int d = 0; d < ndir; ++d) {
        y_seed[d] = {{0.0, 0.0}};
        p_seed[d] = {{0.0, 0.0}};
        if (d < N)
            y_seed[d][d] = 1.0;
        else
            p_seed[d][d - N] = 1.0;
        abs_error[d] = {{ABS_TOL, ABS_TOL}};
        y_seed_ptr[d] = y_seed[d].data();
        p_seed_ptr[d] = p_seed[d].data();
        abs_ptr[d] = abs_error[d].data();
    }

    yabi::BDFIntegrator<double> integrator(model);
    double absolute_error[N] = {ABS_TOL, ABS_TOL};
    if (!integrator.resetIntegration(y0_copy.data(), 0.0, p_copy.data(),
                                     MAX_STEP, REL_TOL, absolute_error, false,
                                     false, y_seed_ptr.data(),
                                     p_seed_ptr.data(), rel_error.data(),
                                     abs_ptr.data(), ndir) ||
        !integrator.performIntegration(T_END, MAX_STEPS)) {
        std::cerr << "simultaneous sensitivity integration failed\n";
        std::exit(2);
    }

    std::vector<std::array<double, N> > dir(ndir);
    std::vector<double*> dir_ptr(ndir);
    for (unsigned int d = 0; d < ndir; ++d) dir_ptr[d] = dir[d].data();
    if (!integrator.getSimultaniousDerivativesVector(dir_ptr.data())) {
        std::cerr << "getSimultaniousDerivativesVector failed\n";
        std::exit(2);
    }

    for (unsigned int d = 0; d < ndir; ++d) {
        if (d < N)
            for (unsigned int j = 0; j < N; ++j) out.dy_dy0[d][j] = dir[d][j];
        else
            for (unsigned int j = 0; j < N; ++j)
                out.dy_dp[d - N][j] = dir[d][j];
    }
}

Sensitivity finite_differences(CppAD::ADFun<double>& model, const State& y0,
                               const Parameter& p) {
    Sensitivity out;
    out.y = integrate(model, y0, p);
    const double eps = 1e-6;

    for (unsigned int i = 0; i < N; ++i) {
        State plus = y0, minus = y0;
        plus[i] += eps;
        minus[i] -= eps;
        State yp = integrate(model, plus, p);
        State ym = integrate(model, minus, p);
        for (unsigned int j = 0; j < N; ++j)
            out.dy_dy0[i][j] = (yp[j] - ym[j]) / (2.0 * eps);
    }

    for (unsigned int k = 0; k < P; ++k) {
        Parameter plus = p, minus = p;
        plus[k] += eps;
        minus[k] -= eps;
        State yp = integrate(model, y0, plus);
        State ym = integrate(model, y0, minus);
        for (unsigned int j = 0; j < N; ++j)
            out.dy_dp[k][j] = (yp[j] - ym[j]) / (2.0 * eps);
    }
    return out;
}

bool close(double a, double b) {
    return std::fabs(a - b) <=
           CHECK_TOL * std::max(1.0, std::max(std::fabs(a), std::fabs(b)));
}

void report(const std::string& name, const Sensitivity& got,
            const Sensitivity& ref, bool& ok) {
    bool local_ok = true;
    for (unsigned int i = 0; i < N; ++i)
        for (unsigned int j = 0; j < N; ++j)
            local_ok = local_ok && close(got.dy_dy0[i][j], ref.dy_dy0[i][j]);
    for (unsigned int k = 0; k < P; ++k)
        for (unsigned int j = 0; j < N; ++j)
            local_ok = local_ok && close(got.dy_dp[k][j], ref.dy_dp[k][j]);
    std::cout << std::left << std::setw(16) << name
              << (local_ok ? "PASS" : "FAIL") << '\n';
    ok = ok && local_ok;
}

}  // namespace

int main() {
    CppAD::ADFun<double> model = make_model();
    State y0 = {{1.0, 0.0}};
    Parameter p = {{0.7, 0.3}};

    Sensitivity reference = finite_differences(model, y0, p);
    bool ok = true;

    std::cout << "y(" << T_END << ") = " << reference.y[0] << ", "
              << reference.y[1] << "\n\n";

    Sensitivity forward;
    forward.y = reference.y;
    fill_forward(model, y0, p, forward);
    report("forward", forward, reference, ok);

    Sensitivity reverse;
    reverse.y = reference.y;
    fill_reverse(model, y0, p, reverse);
    report("reverse", reverse, reference, ok);

    Sensitivity simultaneous = {};
    simultaneous.y = reference.y;
    fill_simultaneous(model, y0, p, simultaneous);
    report("simultaneous", simultaneous, reference, ok);

    return ok ? 0 : 1;
}
