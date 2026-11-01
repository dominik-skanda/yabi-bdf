#pragma once
#include <cmath>
#include <functional>

namespace yabi {
// Generic pure-C++ ODE interface — no predefined equations.
// User hands over: f(t, y) callable (e.g. std::function, lambda, or raw pointer)
// and sensitivity matrices (seed / adjoint) by pointer arrays.
// Example usage (not predefined):
//   std::function<void(double t, const double* y, int n, double* f)> rhs_user;
// This header owns NO benchmark dynamics; it only provides the interface contract.
// See include/yabi/sensitivity_manual.hpp for manual sensitivity overload.
}
