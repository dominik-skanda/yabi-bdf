/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// T44 — machine-readable performance/accuracy telemetry.  Runs the canonical
// benchmarks across a tolerance sweep and prints, per (benchmark, rtol), the
// global error vs an independent reference, the wall-clock time, and the
// diagnostic counters (Jacobian evaluations, factorizations, accepted steps,
// final order).  Consumed by benchmarks/report.py to produce the evidence table
// and work-precision plots.

#include <yabi/bdf_integrator.hpp>

#include <cppad/cppad.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace {

struct Run {
    std::string name;
    double rtol;
    double error;
    double runtime_ms;
    unsigned int jacobian_evals;
    unsigned int factorizations;
    unsigned int accepted_steps;
    unsigned int final_order;
};

// Robertson (autonomous), reference at t=1e4 from a Radau solve.
struct Robertson {
    static CppAD::ADFun<double> tape() {
        CppAD::vector<CppAD::AD<double>> x(3);
        x[0] = 1.0;
        x[1] = 0.0;
        x[2] = 0.0;
        CppAD::Independent(x);
        CppAD::vector<CppAD::AD<double>> f(3);
        f[0] = -0.04 * x[0] + 1.0e4 * x[1] * x[2];
        f[1] = 0.04 * x[0] - 1.0e4 * x[1] * x[2] - 3.0e7 * x[1] * x[1];
        f[2] = 3.0e7 * x[1] * x[1];
        CppAD::ADFun<double> fun;
        fun.Dependent(x, f);
        fun.optimize();
        return fun;
    }
    static double *y0() {
        static double v[3] = {1.0, 0.0, 0.0};
        return v;
    }
    static double *abserr() {
        static double v[3] = {1e-12, 1e-18, 1e-12};
        return v;
    }
    static const double *ref() {
        static const double v[3] = {0.107300428538, 4.80016697257e-07, 0.892699091445};
        return v;
    }
    static constexpr double t_end = 1.0e4;
    static constexpr double max_step = 100.0;
    static constexpr std::size_t n = 3;
};

// Brusselator (autonomous), reference y1(5)=0.938299, y2(5)=4.52728.
struct Brusselator {
    static CppAD::ADFun<double> tape() {
        CppAD::vector<CppAD::AD<double>> x(2);
        x[0] = 1.0;
        x[1] = 2.0;
        CppAD::Independent(x);
        CppAD::vector<CppAD::AD<double>> f(2);
        f[0] = 1.0 + x[0] * x[0] * x[1] - 4.0 * x[0];
        f[1] = 3.0 * x[0] - x[0] * x[0] * x[1];
        CppAD::ADFun<double> fun;
        fun.Dependent(x, f);
        fun.optimize();
        return fun;
    }
    static double *y0() {
        static double v[2] = {1.0, 2.0};
        return v;
    }
    static double *abserr() {
        static double v[2] = {1e-10, 1e-10};
        return v;
    }
    static const double *ref() {
        static const double v[2] = {0.93829903281, 4.52727610726};
        return v;
    }
    static constexpr double t_end = 5.0;
    static constexpr double max_step = 5.0;
    static constexpr std::size_t n = 2;
};

template <typename B> Run runAt(const std::string &name, double rtol) {
    CppAD::ADFun<double> fun = B::tape();
    yabi::BDFIntegrator<double> it(fun);

    auto t0 = std::chrono::high_resolution_clock::now();
    bool ok = it.resetIntegration(B::y0(), B::max_step, rtol, B::abserr(), false, false, NULL,
                                  NULL, NULL, NULL, 0);
    if (ok)
        ok = it.setAlgorithmSettings(5, std::numeric_limits<double>::epsilon(), true, 1);
    if (ok)
        ok = it.performIntegration(B::t_end, 10000000);
    auto t1 = std::chrono::high_resolution_clock::now();

    Run r;
    r.name = name;
    r.rtol = rtol;
    r.runtime_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::vector<double> y(B::n);
    if (!ok || !it.getSolutionVector(y.data())) {
        r.error = std::numeric_limits<double>::infinity();
        r.jacobian_evals = r.factorizations = r.accepted_steps = r.final_order = 0;
        return r;
    }
    double e = 0.0;
    for (std::size_t i = 0; i < B::n; ++i)
        e = std::max(e, std::fabs(y[i] - B::ref()[i]) / (1.0 + std::fabs(B::ref()[i])));
    r.error = e;
    r.jacobian_evals = it.getJacobianEvaluations();
    r.factorizations = it.getFactorizations();
    r.accepted_steps = it.getAcceptedSteps();
    r.final_order = it.getCurrentOrder();
    return r;
}

} // namespace

int main() {
    const double rtols[] = {1e-4, 1e-6, 1e-8, 1e-10};
    std::printf("benchmark,rtol,error,runtime_ms,jacobian_evals,factorizations,steps,order\n");
    for (double rtol : rtols) {
        for (const Run &r : {runAt<Robertson>("robertson", rtol), runAt<Brusselator>("brusselator", rtol)}) {
            std::printf("%s,%.1e,%.6e,%.3f,%u,%u,%u,%u\n", r.name.c_str(), r.rtol, r.error,
                        r.runtime_ms, r.jacobian_evals, r.factorizations, r.accepted_steps,
                        r.final_order);
        }
    }
    return 0;
}
