/*
 * Regression test: an invalid error weight must fail loudly, not return junk.
 *
 * With atol = 0 and a state that is exactly 0 the error weight of that
 * species is 0.  logb(0) is -inf and scaling the Newton matrix with it used
 * to produce a singular system whose dgetrf failure was ignored; the
 * integrator then returned a wrong answer as success.  The T09 guards must
 * turn that into performIntegration() == false.
 *
 * The second case shows the same model with a positive atol still integrates
 * correctly, so the guard only rejects the invalid configuration.
 */

#include <yabi/bdf_integrator.hpp>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

namespace {

struct Decay
{
    CppAD::ADFun<double> rhs;

    Decay()
    {
        CppAD::vector<CppAD::AD<double> > x(2);
        x[0] = 1.0;
        x[1] = 0.0;
        CppAD::Independent(x);

        CppAD::vector<CppAD::AD<double> > f(2);
        f[0] = -x[0];
        f[1] = -x[1];

        rhs.Dependent(x, f);
        rhs.optimize();
    }
};

bool run(double atol, bool &performed, double &y0_out)
{
    Decay model;
    yabi::BDFIntegrator<double> integrator(model.rhs);

    double y_start[2] = {1.0, 0.0};
    double absolute_error[2] = {atol, atol};
    double y[2] = {0.0, 0.0};

    if (!integrator.resetIntegration(y_start, 1.0, 1e-6, absolute_error, false,
                                     false, NULL, NULL, NULL, NULL, 0))
        return false;

    performed = integrator.performIntegration(1.0, 1000000);
    if (performed && !integrator.getSolutionVector(y))
        return false;

    y0_out = y[0];
    return true;
}

}  // namespace

int main()
{
    bool performed = false;
    double y0 = 0.0;

    // Invalid: atol = 0 plusy[1] == 0 must be rejected, not silently solved.
    if (!run(0.0, performed, y0))
    {
        std::cerr << "FAIL: setup failed for the invalid case\n";
        return 1;
    }
    if (performed)
    {
        std::cerr << "FAIL: atol=0 with a zero state returned success, y[0]="
                  << std::setprecision(17) << y0 << "\n";
        return 1;
    }

    // Valid: a positive atol keeps the model solvable and accurate.
    if (!run(1e-14, performed, y0))
    {
        std::cerr << "FAIL: setup failed for the valid case\n";
        return 1;
    }
    if (!performed)
    {
        std::cerr << "FAIL: atol=1e-14 was rejected unexpectedly\n";
        return 1;
    }

    const double exact = std::exp(-1.0);
    const double relerr = std::fabs(y0 - exact) / exact;
    std::cout << std::setprecision(15)
              << "invalid case: rejected as expected\n"
              << "valid case:   y(1)=" << y0 << " exact=" << exact
              << " relerr=" << relerr << "\n";

    if (relerr > 1e-4)
    {
        std::cerr << "FAIL: valid case inaccurate\n";
        return 1;
    }

    std::cout << "PASS\n";
    return 0;
}
