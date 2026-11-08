/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// Unit tests for the CppADRHS higher-order hooks (T25/T26/T27/T40): the adjoint
// product, the Hessian-in-direction, the second-order directional, and the
// order-k directional source — verified against hand-derived values on a coupled
// nonlinear system f = (y1^2, y1*y2), where every derivative is known.

#include "test_support.hpp"

#include <yabi/cppad_rhs.hpp>

#include <cppad/cppad.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace {

// f1 = y1^2, f2 = y1*y2.
//   J = [[2 y1, 0], [y2, y1]]
//   H1 = [[2,0],[0,0]]   H2 = [[0,1],[1,0]]   T = 0
CppAD::ADFun<double> makeCoupledTape() {
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

} // namespace

int main() {
    CppAD::ADFun<double> fun = makeCoupledTape();
    yabi::CppADRHS<double> rhs(fun);

    const std::vector<double> x = {2.0, 3.0}; // y = (2, 3)

    // --- adjointProduct = J^T lambda ----------------------------------------
    {
        std::vector<double> lambda = {1.0, 2.0}, result;
        yabi_test::check(rhs.adjointProduct(x, lambda, result), "adjointProduct succeeds");
        // J^T lambda = (2 y1 l1 + y2 l2, y1 l2) = (4*1 + 3*2, 2*2) = (10, 4)
        yabi_test::check(result.size() == 2 && yabi_test::near(result[0], 10.0) &&
                             yabi_test::near(result[1], 4.0),
                         "adjointProduct = J^T lambda");
    }

    // --- hessianVectorProduct = Hess(lambda^T f) * v ------------------------
    {
        std::vector<double> lambda = {1.0, 2.0}, v = {1.0, 0.0}, result;
        yabi_test::check(rhs.hessianVectorProduct(x, lambda, v, result),
                         "hessianVectorProduct succeeds");
        // Hess(lambda^T f) = [[2 l1, l2],[l2, 0]] = [[2,2],[2,0]]; *v = (2, 2)
        yabi_test::check(result.size() == 2 && yabi_test::near(result[0], 2.0) &&
                             yabi_test::near(result[1], 2.0),
                         "hessianVectorProduct = Hess * v");
    }

    // --- secondOrderDirectional = H * v * v ---------------------------------
    {
        std::vector<double> v = {1.0, 2.0}, result;
        yabi_test::check(rhs.secondOrderDirectional(x, v, result),
                         "secondOrderDirectional succeeds");
        // H1(v,v) = 2 v1^2 = 2;  H2(v,v) = 2 v1 v2 = 4
        yabi_test::check(result.size() == 2 && yabi_test::near(result[0], 2.0) &&
                             yabi_test::near(result[1], 4.0),
                         "secondOrderDirectional = H v v");
    }

    // --- orderKDirectional(2) = H(S1,S1) ------------------------------------
    {
        std::vector<std::vector<double>> S = {{1.0, 0.0}, {0.0, 1.0}};
        std::vector<double> result;
        yabi_test::check(rhs.orderKDirectional(2, x, S, result), "orderKDirectional(2) succeeds");
        // H1(S1,S1) = 2, H2(S1,S1) = 0
        yabi_test::check(result.size() == 2 && yabi_test::near(result[0], 2.0) &&
                             yabi_test::near(result[1], 0.0),
                         "orderKDirectional(2) = H(S1,S1)");
    }

    // --- orderKDirectional(3) = 3 H(S1,S2) + T(S1,S1,S1) --------------------
    {
        std::vector<std::vector<double>> S = {{1.0, 0.0}, {0.0, 1.0}, {0.0, 0.0}};
        std::vector<double> result;
        yabi_test::check(rhs.orderKDirectional(3, x, S, result), "orderKDirectional(3) succeeds");
        // H1(S1,S2) = 2*1*0 = 0 -> 3*0 = 0;  H2(S1,S2) = 1*1 + 0*0 = 1 -> 3
        yabi_test::check(result.size() == 2 && yabi_test::near(result[0], 0.0) &&
                             yabi_test::near(result[1], 3.0),
                         "orderKDirectional(3) = 3 H(S1,S2) + T");
    }

    // --- orderKDirectional rejects k < 2 and wrong-size inputs ---------------
    {
        std::vector<double> result;
        yabi_test::check(!rhs.orderKDirectional(1, x, {{1.0, 0.0}}, result),
                         "orderKDirectional rejects k < 2");
        std::vector<std::vector<double>> badS = {{1.0}}; // wrong direction size
        yabi_test::check(!rhs.orderKDirectional(2, x, badS, result),
                         "orderKDirectional rejects wrong direction size");
    }

    // --- secondOrderDirectional rejects wrong-size v ------------------------
    {
        std::vector<double> result;
        yabi_test::check(!rhs.secondOrderDirectional(x, {1.0}, result),
                         "secondOrderDirectional rejects wrong-size v");
    }

    return yabi_test::finish("CppADRHS higher-order hooks");
}
