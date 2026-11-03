/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */
// Pure (CppAD-free) instantiation of the integrator.  YABI_NO_CPPAD drops the
// CppAD adapter include and the CppAD::ADFun convenience constructor, so this
// translation unit (and the yabi_pure library) has no dependency on CppAD.
// (CMake passes -DYABI_NO_CPPAD to this target; the guard keeps it standalone-safe.)
#ifndef YABI_NO_CPPAD
#define YABI_NO_CPPAD
#endif
#include <yabi/bdf_integrator.hpp>

namespace yabi {
template class BDFIntegrator<double>;
} // namespace yabi
