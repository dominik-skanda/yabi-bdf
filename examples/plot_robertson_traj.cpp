#include <yabi/bdf_integrator.hpp>
#include <cppad/cppad.hpp>
#include <iostream>
#include <limits>
#include <cmath>

int main() {
    CppAD::ADFun<double> rhs;
    CppAD::vector<CppAD::AD<double> > x(3);
    x[0] = 1.0; x[1] = 0.0; x[2] = 0.0;
    CppAD::Independent(x);
    CppAD::vector<CppAD::AD<double> > f(3);
    double k1 = 0.04, k2 = 1.0e4, k3 = 3.0e7;
    f[0] = -k1*x[0] + k2*x[1]*x[2];
    f[1] =  k1*x[0] - k2*x[1]*x[2] - k3*x[1]*x[1];
    f[2] =                    k3*x[1]*x[1];
    rhs.Dependent(x, f);
    rhs.optimize();

    yabi::BDFIntegrator<double> integrator(rhs);
    double y0[3] = {1.0, 0.0, 0.0};
    double abs_err[3] = {1e-12, 1e-18, 1e-12};
    double y[3];

    if (!integrator.resetIntegration(y0, 10000.0, 1e-10, abs_err, false, false, NULL,NULL,NULL,NULL,0)) {
        return 1;
    }
    if (!integrator.setAlgorithmSettings(5, std::numeric_limits<double>::epsilon(), true, 1)) {
        return 1;
    }

    std::cout << "t,y1,y2,y3" << std::endl;
    std::cout << "0," << y0[0] << "," << y0[1] << "," << y0[2] << std::endl;

    // Sample to match Hairer-Wanner behavior
    const double targets[] = {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 10000.0};
    for (int k = 0; k < 11; ++k) {
        if (!integrator.performIntegration(targets[k], 10000000)) {
            return 1;
        }
        if (!integrator.getSolutionVector(y)) {
            return 1;
        }
        std::cout << targets[k] << "," << y[0] << "," << y[1] << "," << y[2] << std::endl;
    }
    return 0;
}
