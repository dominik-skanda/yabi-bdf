// T12h — Second-order sensitivity (forward + reverse), verified against the exact
// analytical solution of y' = y^2.
//
// YABI's forward(order) is incremental: CppAD Forward(q) requires the lower Taylor orders
// to be computed first, so to get order q you must run orders 1..q-1 first. The canonical
// sequence (matching the 2011 original INTEGRATOR_66_HOME/test.cpp) is:
//   forward(1, seed, NULL, true);  forward(1, seed, NULL, false);
//   forward(2, seed, NULL, true);  forward(2, seed, NULL, false);
// then getForwardSolutionVector returns the 2nd-order directional derivative. A direct
// forward(2) without a prior forward(1) aborts (documented limitation, identical in 2011).
//
// Reference: y' = y^2, y(0) = y0  =>  y(t) = y0/(1 - y0 t)
//   dy/dy0    = 1/(1 - y0 t)^2
//   d2y/dy0^2 = 2 t/(1 - y0 t)^3
// At y0 = 1, t = 0.5:  y = 2, dy/dy0 = 4, d2y/dy0^2 = 8.
#include <yabi/bdf_integrator.hpp>
#include <cppad/cppad.hpp>
#include <cmath>
#include <iostream>

int main() {
    std::cout << "=== T12h Second-order sensitivity (forward + reverse) ===" << std::endl;
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(1);
    x[0] = 1.0;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(1);
    f[0] = x[0] * x[0];
    rhs.Dependent(x, f);
    rhs.optimize();

    double y0[1] = {1.0}, abs_err[1] = {1e-12}, seed[1] = {1.0}, w[1] = {1.0}, d[1];
    const double dy_exact = 4.0, d2y_exact = 8.0;
    bool ok = true;

    // Forward: first then second order (incremental).
    {
        yabi::BDFIntegrator<double> it(rhs);
        if (!it.resetIntegration(y0, 0.0, NULL, 0.1, 1e-12, abs_err, true, false, NULL, NULL, NULL, NULL, 0)) return 1;
        if (!it.performIntegration(0.5, 100000)) return 1;

        it.forward(1, seed, NULL, true);
        it.forward(1, seed, NULL, false);
        it.getForwardSolutionVector(d);
        double r1 = std::fabs(d[0] - dy_exact) / (1.0 + dy_exact);
        std::cout << "forward(1) = " << d[0] << " (exact " << dy_exact << ") rel=" << r1 << std::endl;
        if (r1 > 1e-6) ok = false;

        it.forward(2, seed, NULL, true);
        it.forward(2, seed, NULL, false);
        it.getForwardSolutionVector(d);
        double r2 = std::fabs(d[0] - d2y_exact) / (1.0 + d2y_exact);
        std::cout << "forward(2) = " << d[0] << " (exact " << d2y_exact << ") rel=" << r2 << std::endl;
        if (r2 > 1e-6) ok = false;
    }

    // Reverse: second-order adjoint.
    {
        yabi::BDFIntegrator<double> it(rhs);
        if (!it.resetIntegration(y0, 0.0, NULL, 0.1, 1e-12, abs_err, true, false, NULL, NULL, NULL, NULL, 0)) return 1;
        if (!it.performIntegration(0.5, 100000)) return 1;
        it.forward(1, seed, NULL, true);
        it.forward(1, seed, NULL, false);
        it.forward(2, seed, NULL, true);
        it.forward(2, seed, NULL, false);
        double *adj[1];
        adj[0] = new double[3];
        if (!it.reverse(2, w)) { std::cerr << "FAIL reverse(2)" << std::endl; return 1; }
        it.getReverseSolutionVector(adj, NULL);
        double ra1 = std::fabs(adj[0][0] - dy_exact) / (1.0 + dy_exact);
        double ra2 = std::fabs(adj[0][1] - d2y_exact) / (1.0 + d2y_exact);
        std::cout << "reverse(2) = [order1 " << adj[0][0] << ", order2 " << adj[0][1] << "]"
                  << " rel=(" << ra1 << "," << ra2 << ")" << std::endl;
        if (ra1 > 1e-6 || ra2 > 1e-6) ok = false;
        delete[] adj[0];
    }

    std::cout << (ok ? "PASS" : "FAIL") << std::endl;
    return ok ? 0 : 1;
}
