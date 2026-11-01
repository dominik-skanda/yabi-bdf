#pragma once
#include <yabi/bdf_integrator.hpp>
#include <vector>
namespace yabi {
// Manual sensitivity interface — accepts pre-computed seed/adjoint matrices.
// Does NOT replace CppAD AD path; supplementary interface only.
bool manualIntegrateSensitivity(
    BDFIntegrator<double>& integ,
    double* y_start, double T_start, double max_step, double rel_err, double* abs_err,
    bool taping, bool direct,
    double** y_seed_forward, double** p_seed_forward,
    double** y_seed_reverse, double** p_seed_reverse,
    unsigned int num_derivatives);
}
