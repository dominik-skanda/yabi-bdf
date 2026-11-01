#include <yabi/bdf_integrator.hpp>
#include <cppad/cppad.hpp>
#include <iostream>
#include <cmath>
#include <limits>

int main() {
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double> > x(2);
    x[0] = 1.0; x[1] = 0.0;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double> > f(2);
    f[0] = x[1];
    f[1] = -x[0];
    rhs.Dependent(x, f);
    rhs.optimize();

    yabi::BDFIntegrator<double> integrator(rhs);
    double y0[2] = {1.0, 0.0};
    double abs_err[2] = {1e-12, 1e-12};
    double y[2];

    if (!integrator.resetIntegration(y0, 20.0, 1e-12, abs_err, false, false, NULL,NULL,NULL,NULL,0)) return 1;
    if (!integrator.setAlgorithmSettings(5, std::numeric_limits<double>::epsilon(), true, 1)) return 1;

    std::cout << "t,y1,y2" << std::endl;
    std::cout << "0," << y0[0] << "," << y0[1] << std::endl;
    for (double t = 0.5; t <= 20.0; t += 0.5) {
        if (!integrator.performIntegration(t, 100000)) return 1;
        integrator.getSolutionVector(y);
        std::cout << t << "," << y[0] << "," << y[1] << std::endl;
    }
    return 0;
}
