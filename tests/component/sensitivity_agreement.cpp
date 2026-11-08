/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// T42 — cross-component sensitivity agreement: all four first-order modes
// (forward, reverse, simultaneous/variational, continuous adjoint) must produce
// the same state-Jacobian dy(T)/dy0, verified against the exact solution of the
// coupled system y1'=y1^2, y2'=y1*y2.  This is the order-1 agreement the T28
// method guide flagged as not-yet-asserted.

#include <yabi/bdf_integrator.hpp>
#include <yabi/cppad_rhs.hpp>

#include <cppad/cppad.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const char *what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << std::endl;
        ++failures;
    }
}

CppAD::ADFun<double> makeCoupled() {
    CppAD::vector<CppAD::AD<double>> x(2);
    x[0] = 2.0;
    x[1] = 3.0;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(2);
    f[0] = x[0] * x[0];
    f[1] = x[0] * x[1];
    CppAD::ADFun<double> fun;
    fun.Dependent(x, f);
    fun.optimize();
    return fun;
}

const double a = 2.0, b = 3.0, T = 0.2;

// Exact state Jacobian dy/dy0 at t=T (2x2, row-major [dy_i/dy0_j]):
//   [[1/(1-aT)^2, 0], [bT/(1-aT)^2, 1/(1-aT)]]
void exactJacobian(double J[2][2]) {
    const double e = 1.0 - a * T;
    J[0][0] = 1.0 / (e * e);
    J[0][1] = 0.0;
    J[1][0] = b * T / (e * e);
    J[1][1] = 1.0 / e;
}

bool near(double x, double y) { return std::fabs(x - y) < 1e-8; }

// Fill J[row][col] = d y_row / d y0_col, cross-checked against the exact value.
void verify(const char *name, double J[2][2]) {
    double E[2][2];
    exactJacobian(E);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            check(near(J[i][j], E[i][j]), name);
}

} // namespace

int main() {
    std::cout << "T42 sensitivity 4-mode agreement (coupled y'=y^2, y*y)\n";
    double y0[2] = {a, b};
    double abs_err[2] = {1e-12, 1e-12};

    // --- forward(1): columns of the Jacobian --------------------------------
    double J_fwd[2][2];
    {
        CppAD::ADFun<double> fun = makeCoupled();
        yabi::BDFIntegrator<double> it(fun);
        it.resetIntegration(y0, 0.0, NULL, 0.01, 1e-12, abs_err, true, false, NULL, NULL, NULL,
                            NULL, 0);
        it.performIntegration(T, 100000);
        for (int j = 0; j < 2; ++j) {
            double seed[2] = {(double)(j == 0), (double)(j == 1)};
            it.forward(1, seed, NULL, true); // reset for each new direction
            double dir[2];
            it.getForwardSolutionVector(dir);
            J_fwd[0][j] = dir[0];
            J_fwd[1][j] = dir[1];
        }
        verify("forward(1)", J_fwd);
    }

    // --- reverse(1): rows of the Jacobian -----------------------------------
    double J_rev[2][2];
    {
        CppAD::ADFun<double> fun = makeCoupled();
        yabi::BDFIntegrator<double> it(fun);
        it.resetIntegration(y0, 0.0, NULL, 0.01, 1e-12, abs_err, true, false, NULL, NULL, NULL,
                            NULL, 0);
        it.performIntegration(T, 100000);
        for (int i = 0; i < 2; ++i) {
            double weight[2] = {(double)(i == 0), (double)(i == 1)};
            it.reverse(1, weight);
            double *adj_sp[2] = {new double[1], new double[1]};
            double *adj_p[2] = {nullptr, nullptr};
            it.getReverseSolutionVector(adj_sp, adj_p);
            J_rev[i][0] = adj_sp[0][0];
            J_rev[i][1] = adj_sp[1][0];
            delete[] adj_sp[0];
            delete[] adj_sp[1];
        }
        verify("reverse(1)", J_rev);
    }

    // --- simultaneous (variational): columns --------------------------------
    double J_sim[2][2];
    {
        CppAD::ADFun<double> fun = makeCoupled();
        yabi::BDFIntegrator<double> it(fun);
        double yseed_arr[4] = {1.0, 0.0, 0.0, 1.0}; // e0, e1
        double *yseed_ptr[2] = {&yseed_arr[0], &yseed_arr[2]};
        double rel_seed[2] = {1e-12, 1e-12};
        double abs_seed[2][2] = {{1e-12, 1e-12}, {1e-12, 1e-12}};
        double *abs_seed_ptr[2] = {abs_seed[0], abs_seed[1]};
        yabi::IntegrationOptions<double> opts;
        opts.taping = false;
        opts.direct = false;
        opts.y_seed_list = yseed_ptr;
        opts.relative_error_derivatives = rel_seed;
        opts.absolut_error_derivatives = abs_seed_ptr;
        opts.numOfderivatives = 2;
        it.resetIntegration(y0, 0.0, NULL, 0.01, 1e-12, abs_err, opts);
        it.performIntegration(T, 100000);
        double dirs[4];
        double *dir_ptr[2] = {&dirs[0], &dirs[2]};
        it.getSimultaneousDerivativesVector(dir_ptr);
        J_sim[0][0] = dirs[0];
        J_sim[1][0] = dirs[1];
        J_sim[0][1] = dirs[2];
        J_sim[1][1] = dirs[3];
        verify("simultaneous", J_sim);
    }

    // --- continuous adjoint: rows -------------------------------------------
    double J_cadj[2][2];
    {
        CppAD::ADFun<double> fun = makeCoupled();
        yabi::BDFIntegrator<double> it(fun);
        it.resetIntegration(y0, 0.0, NULL, 0.01, 1e-12, abs_err, false, false, NULL, NULL, NULL,
                            NULL, 0);
        it.performIntegration(T, 100000);
        for (int i = 0; i < 2; ++i) {
            double weight[2] = {(double)(i == 0), (double)(i == 1)};
            double adj_y0[2];
            it.continuousAdjoint(weight, adj_y0, nullptr);
            J_cadj[i][0] = adj_y0[0];
            J_cadj[i][1] = adj_y0[1];
        }
        verify("continuousAdjoint", J_cadj);
    }

    // --- all four agree with each other -------------------------------------
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) {
            check(near(J_fwd[i][j], J_rev[i][j]) && near(J_fwd[i][j], J_sim[i][j]) &&
                      near(J_fwd[i][j], J_cadj[i][j]),
                  "all four modes agree");
        }

    if (failures) {
        std::cerr << failures << " sensitivity-agreement check(s) failed\n";
        return 1;
    }
    std::cout << "PASS: forward == reverse == simultaneous == continuous adjoint\n";
    return 0;
}
