// Pure (CppAD-free) Brusselator — the user_ode.hpp / UserODERHS path.
// Same problem as system test 12, but the RHS and Jacobian are hand-written
// (no AD, no finite differences): f = A + y0^2*y1 - (B+1)*y0;  g = B*y0 - y0^2*y1.
// Reference (CppAD system test 12): y1(5) = 0.938299, y2(5) = 4.52728.
#include <yabi/bdf_integrator.hpp>
#include <yabi/user_ode_rhs.hpp>

#include <cmath>
#include <iostream>
#include <limits>

int main() {
    const int n = 2;
    const double A = 1.0, B = 3.0;

    // Autonomous ODE f(y).
    auto f = [&](const double *y, double *F) {
        F[0] = A + y[0] * y[0] * y[1] - (B + 1.0) * y[0];
        F[1] = B * y[0] - y[0] * y[0] * y[1];
    };
    // Analytic Jacobian J(y) = df/dy, column-major: J[i + n*j] = df_i/dy_j.
    auto J = [&](const double *y, double *Jm) {
        Jm[0 + n * 0] = 2.0 * y[0] * y[1] - (B + 1.0); // df0/dy0
        Jm[1 + n * 0] = B - 2.0 * y[0] * y[1];         // df1/dy0
        Jm[0 + n * 1] = y[0] * y[0];                   // df0/dy1
        Jm[1 + n * 1] = -y[0] * y[0];                  // df1/dy1
    };

    yabi::UserODERHS<double> rhs(n, f, J);
    yabi::BDFIntegrator<double> integrator(rhs);

    double y0[2] = {1.0, 2.0};
    double abs_err[2] = {1e-10, 1e-10};
    double y[2];

    if (!integrator.resetIntegration(y0, 5.0, 1e-10, abs_err, false, false, NULL, NULL, NULL, NULL,
                                     0) ||
        !integrator.setAlgorithmSettings(5, std::numeric_limits<double>::epsilon(), true, 1) ||
        !integrator.performIntegration(5.0, 10000000) ||
        !integrator.getSolutionVector(y)) {
        std::cerr << "FAIL: integration" << std::endl;
        return 1;
    }

    const double ref[2] = {0.938299, 4.52728};
    const double tol[2] = {1e-4, 1e-4};
    std::cout << "y1(5) = " << y[0] << ", y2(5) = " << y[1] << std::endl;
    for (int i = 0; i < 2; ++i)
        if (std::fabs(y[i] - ref[i]) > tol[i]) {
            std::cerr << "FAIL: y[" << i << "] = " << y[i] << " vs ref " << ref[i] << std::endl;
            return 1;
        }
    std::cout << "PASS (pure UserODERHS, no CppAD)" << std::endl;
    return 0;
}
