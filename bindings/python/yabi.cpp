/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

// T37 — Python bindings (pybind11, zero-copy NumPy) for the analytic path.
//
// Exposes yabi::BDFIntegrator<double> driven by a yabi::UserODERHS whose RHS and
// analytic Jacobian are Python callables.  The state buffers (resetIntegration's
// y_start and getSolutionVector's y) are passed to YABI as raw pointers into the
// caller's NumPy arrays (py::array_t<double>::mutable_data()), so there is no
// copy on the state data.  Built against `yabi_pure` (YABI_NO_CPPAD): the CppAD
// auto-diff path is intentionally out of scope for this first milestone.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <yabi/bdf_integrator.hpp>
#include <yabi/user_ode_rhs.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

using Integrator = yabi::BDFIntegrator<double>;

// Wrap a Python callable (f or jac) as a C++ callback.  `out_size` is the number
// of doubles the callable must return (n for f, n*n for the column-major J).
struct PyCallable {
    py::object fn;
    std::size_t in_size;
    std::size_t out_size;

    void operator()(const double *y, double *out) const {
        // The caller holds the GIL (pybind11 does not release it here), so we
        // can call straight back into Python.
        py::array_t<double> y_view(in_size, const_cast<double *>(y));
        py::array_t<double> r = py::array_t<double>::ensure(fn(y_view));
        if (r.ndim() != 1 || r.size() != static_cast<py::ssize_t>(out_size)) {
            throw std::runtime_error("RHS/Jacobian callable must return " +
                                     std::to_string(out_size) + " doubles");
        }
        std::copy(r.data(), r.data() + out_size, out);
    }
};

} // namespace

PYBIND11_MODULE(yabi, m) {
    m.doc() = "YABI BDF integrator — Python bindings (analytic / UserODERHS path)";

    py::class_<Integrator>(m, "BDFIntegrator")
        .def(py::init([](py::object f, py::object jac, std::size_t n) {
                 // Build a temporary UserODERHS around the Python callables;
                 // BDFIntegrator's constructor clones it (owning the copies), so
                 // the temporary may die here.
                 yabi::UserODERHS<double> rhs(n, PyCallable{f, n, n}, PyCallable{jac, n, n * n});
                 return new Integrator(rhs);
             }),
             py::arg("f"), py::arg("jac"), py::arg("n"),
             "Construct from Python callables f(y)->f and jac(y)->J (column-major).")
        .def("reset_integration",
             [](Integrator &it, py::array_t<double> y0, double max_step, double rtol,
                py::array_t<double> atol, bool taping) {
                 // Zero-copy: hand YABI the raw buffer pointers (no copy).
                 // `taping` records the trajectory so forward()/reverse() replay.
                 return it.resetIntegration(y0.mutable_data(), max_step, rtol, atol.mutable_data(),
                                            taping, false, nullptr, nullptr, nullptr, nullptr, 0);
             },
             py::arg("y0"), py::arg("max_step"), py::arg("rtol"), py::arg("atol"),
             py::arg("taping") = false)
        .def("integrate",
             [](Integrator &it, double t_end) { return it.performIntegration(t_end, 10000000); },
             py::arg("t_end"))
        .def("get_solution",
             [](Integrator &it, py::array_t<double> y) {
                 // Zero-copy: YABI writes directly into y's buffer.
                 return it.getSolutionVector(y.mutable_data());
             },
             py::arg("y"))
        // First-order continuous adjoint (T25): adj = d(seed_weights . y(T)) / d y0.
        // This is the stable, analytic-path reverse (the discrete reverse() is
        // CppAD-only), so it is the backward hook for autograd wrappers.
        .def("continuous_adjoint",
             [](Integrator &it, py::array_t<double> seed_weights, py::array_t<double> adj_y0) {
                 return it.continuousAdjoint(seed_weights.mutable_data(), adj_y0.mutable_data(),
                                             nullptr);
             },
             py::arg("seed_weights"), py::arg("adj_y0"))
        .def("set_algorithm_settings",
             [](Integrator &it, unsigned max_order, double mach_prec, bool hit_end,
                unsigned ew_index) {
                 return it.setAlgorithmSettings(max_order, mach_prec, hit_end, ew_index);
             },
             py::arg("max_order"), py::arg("mach_prec"), py::arg("hit_end") = true,
             py::arg("ew_index") = 1);
}
