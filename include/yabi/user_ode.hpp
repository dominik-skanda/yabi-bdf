#pragma once
#include <cmath>
#include <functional>
#include <vector>

// Clear user interface: you provide f(t, y) callable.
// Example usage (no predefined benchmark inside yabi):
//
//   std::function<void(double t, const double* y, int n, double* f)> my_ode =
//       [](double t, const double* y, int n, double* f) -> void {
//           f[0] = ...; f[1] = ...; // your dynamics
//       };
//
// Then pass my_ode to yabi::BDFIntegrator through the manual sensitivity
// overloads (see include/yabi/sensitivity_manual.hpp).
//
// No benchmark equation lives in this file or in the library core.
// Only the interface contract is defined here.

namespace yabi {
// User-provided dynamics signature (no library equation hidden).
using UserODE = std::function<void(double t, const double* y, int n, double* f)>;
}
