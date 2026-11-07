// T12h — Higher-order sensitivity (forward + reverse), verified against the exact
// analytical solution of y' = y^2.
//
// Reference: y' = y^2, y(0) = y0  =>  y(t) = y0/(1 - y0 t)
//   d^n y / dy0^n = n! t^(n-1) / (1 - y0 t)^(n+1)
// At y0 = 1, t = 0.5:  dy/dy0 = 4, d2y/dy0^2 = 8, d3y/dy0^3 = 24, d4y/dy0^4 = 96.
//
// forward(order) returns the order-th directional derivative d^order y / d seed^order
// in the seed direction.  reverse(order) returns, per state, the adjoint Taylor
// coefficients of orders 1..order, aligned with the same pure-derivative
// convention.  Order >= 2 is computed with a full Taylor-polynomial replay of the
// frozen BDF algorithm (no incremental forward(1), forward(2), ... sequence is
// required); see README "Higher-order (forward / reverse)".
#include <yabi/bdf_integrator.hpp>
#include <cppad/cppad.hpp>
#include <cmath>
#include <iostream>

static double exact_derivative(unsigned n, double y0, double t) {
    double f = 1.0;
    for (unsigned k = 2; k <= n; k++) f *= (double)k;
    return f * std::pow(t, (double)(n - 1)) / std::pow(1.0 - y0 * t, (double)(n + 1));
}

int main() {
    std::cout << "=== T12h Higher-order sensitivity (forward + reverse) ===" << std::endl;
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double>> x(1);
    x[0] = 1.0;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double>> f(1);
    f[0] = x[0] * x[0];
    rhs.Dependent(x, f);
    rhs.optimize();

    const double y0 = 1.0, t = 0.5;
    double y_start[1] = {y0}, abs_err[1] = {1e-12}, seed[1] = {1.0}, w[1] = {1.0}, d[1];
    const unsigned int order = 4;
    const double ex[4] = {exact_derivative(1, y0, t), exact_derivative(2, y0, t),
                          exact_derivative(3, y0, t), exact_derivative(4, y0, t)};
    bool ok = true;

    // Forward: one call per order (order >= 2 replays the full Taylor polynomial).
    {
        yabi::BDFIntegrator<double> it(rhs);
        if (!it.resetIntegration(y_start, 0.0, NULL, 0.1, 1e-12, abs_err, true, false, NULL,
                                 NULL, NULL, NULL, 0))
            return 1;
        if (!it.performIntegration(t, 100000)) return 1;

        for (unsigned q = 1; q <= order; q++) {
            it.forward(q, seed, NULL, true);
            it.getForwardSolutionVector(d);
            double rel = std::fabs(d[0] - ex[q - 1]) / (1.0 + ex[q - 1]);
            std::cout << "forward(" << q << ") = " << d[0] << " (exact " << ex[q - 1]
                      << ") rel=" << rel << std::endl;
            if (rel > 1e-6) ok = false;
        }
    }

    // Reverse: adjoint of orders 1..order.
    {
        yabi::BDFIntegrator<double> it(rhs);
        if (!it.resetIntegration(y_start, 0.0, NULL, 0.1, 1e-12, abs_err, true, false, NULL,
                                 NULL, NULL, NULL, 0))
            return 1;
        if (!it.performIntegration(t, 100000)) return 1;

        it.forward(order, seed, NULL, true);
        double *adj[1];
        adj[0] = new double[order];
        if (!it.reverse(order, w)) { std::cerr << "FAIL reverse(" << order << ")" << std::endl; return 1; }
        it.getReverseSolutionVector(adj, NULL);
        std::cout << "reverse(" << order << ") = [";
        for (unsigned k = 0; k < order; k++) {
            double rel = std::fabs(adj[0][k] - ex[k]) / (1.0 + ex[k]);
            std::cout << (k ? " " : "") << "order" << (k + 1) << " " << adj[0][k];
            if (rel > 1e-6) ok = false;
        }
        std::cout << "]" << std::endl;
        delete[] adj[0];
    }

    std::cout << (ok ? "PASS" : "FAIL") << std::endl;
    return ok ? 0 : 1;
}
