/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// Unit tests for the UserODERHS higher-order hooks (pure/analytic): the adjoint
// product (from the analytic Jacobian), the Hessian-in-direction, the
// second-order directional, and the order-k directional source — verified against
// the same hand-derived values as the CppAD path (f = (y1^2, y1*y2)).

#include "test_support.hpp"

#include <yabi/user_ode_rhs.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace {

constexpr int n = 2;

void ode(const double *y, double *f) {
    f[0] = y[0] * y[0];
    f[1] = y[0] * y[1];
}

// column-major J[i + n*j] = df_i/dy_j
void jacobian(const double *y, double *J) {
    J[0 + n * 0] = 2.0 * y[0]; // df0/dy0
    J[1 + n * 0] = y[1];       // df1/dy0
    J[0 + n * 1] = 0.0;        // df0/dy1
    J[1 + n * 1] = y[0];       // df1/dy1
}

// Hess(lambda^T f) * v = [[2 l0, l1],[l1, 0]] * v
void hv(const double *y, const double *lambda, const double *v, double *r) {
    (void)y;
    r[0] = 2.0 * lambda[0] * v[0] + lambda[1] * v[1];
    r[1] = lambda[1] * v[0];
}

// H * v * v = (2 v0^2, 2 v0 v1)
void sod(const double *y, const double *v, double *r) {
    (void)y;
    r[0] = 2.0 * v[0] * v[0];
    r[1] = 2.0 * v[0] * v[1];
}

// order-k nonlinear source (S flattened: S[m*n+i] = S_{m+1}[i]).
void okd(unsigned k, const double *y, const double *S, double *r) {
    (void)y;
    if (k == 2) {
        r[0] = 2.0 * S[0] * S[0];       // H1(S1,S1)
        r[1] = 2.0 * S[0] * S[1];       // H2(S1,S1)
    } else if (k == 3) {
        r[0] = 6.0 * S[0] * S[2];       // 3 H1(S1,S2) = 3*2*S1_0*S2_0
        r[1] = 3.0 * (S[0] * S[3] + S[1] * S[2]); // 3 H2(S1,S2)
    } else {
        r[0] = r[1] = 0.0;
    }
}

yabi::UserODERHS<double> makeRhs() {
    yabi::UserODERHS<double> rhs(n, ode, jacobian);
    rhs.setHessianVectorProduct(hv);
    rhs.setSecondOrderDirectional(sod);
    rhs.setOrderKDirectional(okd);
    return rhs;
}

} // namespace

int main() {
    yabi::UserODERHS<double> rhs = makeRhs();
    const std::vector<double> x = {2.0, 3.0};

    // --- adjointProduct = J^T lambda (from the analytic Jacobian) -----------
    {
        std::vector<double> lambda = {1.0, 2.0}, result;
        yabi_test::check(rhs.adjointProduct(x, lambda, result), "adjointProduct succeeds");
        yabi_test::check(result.size() == 2 && yabi_test::near(result[0], 10.0) &&
                             yabi_test::near(result[1], 4.0),
                         "adjointProduct = J^T lambda");
    }

    // --- hessianVectorProduct ------------------------------------------------
    {
        std::vector<double> lambda = {1.0, 2.0}, v = {1.0, 0.0}, result;
        yabi_test::check(rhs.hessianVectorProduct(x, lambda, v, result),
                         "hessianVectorProduct succeeds");
        yabi_test::check(result.size() == 2 && yabi_test::near(result[0], 2.0) &&
                             yabi_test::near(result[1], 2.0),
                         "hessianVectorProduct = Hess * v");
    }

    // --- secondOrderDirectional ----------------------------------------------
    {
        std::vector<double> v = {1.0, 2.0}, result;
        yabi_test::check(rhs.secondOrderDirectional(x, v, result),
                         "secondOrderDirectional succeeds");
        yabi_test::check(result.size() == 2 && yabi_test::near(result[0], 2.0) &&
                             yabi_test::near(result[1], 4.0),
                         "secondOrderDirectional = H v v");
    }

    // --- orderKDirectional(2) and (3) ---------------------------------------
    {
        std::vector<std::vector<double>> S2 = {{1.0, 0.0}, {0.0, 1.0}};
        std::vector<double> result;
        yabi_test::check(rhs.orderKDirectional(2, x, S2, result), "orderKDirectional(2) succeeds");
        yabi_test::check(result.size() == 2 && yabi_test::near(result[0], 2.0) &&
                             yabi_test::near(result[1], 0.0),
                         "orderKDirectional(2) = H(S1,S1)");

        std::vector<std::vector<double>> S3 = {{1.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}};
        yabi_test::check(rhs.orderKDirectional(3, x, S3, result), "orderKDirectional(3) succeeds");
        yabi_test::check(result.size() == 2 && yabi_test::near(result[0], 0.0) &&
                             yabi_test::near(result[1], 3.0),
                         "orderKDirectional(3) = 3 H(S1,S2) + T");
    }

    // --- a backend without the hooks fails loudly ---------------------------
    {
        yabi::UserODERHS<double> bare(n, ode, jacobian); // no hooks
        std::vector<double> result, lambda = {1.0, 0.0}, v = {1.0, 0.0};
        yabi_test::check(!bare.hessianVectorProduct(x, lambda, v, result),
                         "no hessianVectorProduct without the hook");
        yabi_test::check(!bare.secondOrderDirectional(x, v, result),
                         "no secondOrderDirectional without the hook");
        std::vector<std::vector<double>> S = {{1.0, 0.0}, {0.0, 1.0}};
        yabi_test::check(!bare.orderKDirectional(2, x, S, result),
                         "no orderKDirectional without the hook");
    }

    return yabi_test::finish("UserODERHS higher-order hooks");
}
