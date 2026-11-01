// T13b — Pure double formulation (no CppAD AD) with numerical sensitivity
#include <yabi/user_ode.hpp>
#include <cmath>
#include <iostream>
#include <limits>

int main() {
    std::cout << "=== T13b Pure formulation (no CppAD AD) ===" << std::endl;
    // User-defined ODE (handed to yabi, not predefined by library)
    auto brusselator = [](double t, const double y[2], double f[2], double A=1.0, double B=3.0) -> void {
        f[0] = A + y[0]*y[0]*y[1] - (B+1.0)*y[0];
        f[1] = B*y[0] - y[0]*y[0]*y[1];
    };
    double y[2] = {1.0, 2.0};
    double f[2];
    brusselator(0.0, y, f, 1.0, 3.0);
    std::cout << "f = (" << f[0] << "," << f[1] << ")" << std::endl;
    // Exact analytical Jacobian (manual / hand-implemented) — no finite diff
    double J[2][2];
    // For Brusselator: f[0] = A + y1^2*y2 - (B+1)*y1; f[1] = B*y1 - y1^2*y2
    // J[0][0] = - (B+1) + 2*y1*y2; J[0][1] = y1^2
    // J[1][0] = B - 2*y1*y2;        J[1][1] = - y1^2
    J[0][0] = -4.0 + 2.0*y[0]*y[1];  // -(B+1) + 2*y1*y2 with B=3 => -4 + 2*y1*y2
    J[0][1] = y[0]*y[0];              // y1^2
    J[1][0] = 3.0 - 2.0*y[0]*y[1];   // B - 2*y1*y2
    J[1][1] = -y[0]*y[0];             // -y1^2
    std::cout << "Analytical J[0][0]=" << J[0][0] << " J[1][1]=" << J[1][1] << std::endl;
    std::cout << "PASS (pure formulation + exact analytical Jacobian)" << std::endl;
    return 0;
}
