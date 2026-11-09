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
#include <yabi/matrix_free_solver.hpp>
#include <yabi/user_ode_rhs.hpp>

// T54 — optional CUDA (cuSOLVER) Newton solve.  Only compiled when the module
// is built with YABI_USE_CUDA (the CMake side links cuSOLVER/cudart and defines
// YABI_HAS_CUDA); the default build stays CUDA-free.
#ifdef YABI_HAS_CUDA
#include <yabi/cuda_linear_solver.hpp>
#endif

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
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

    PyCallable(py::object f, std::size_t in, std::size_t out)
        : fn(std::move(f)), in_size(in), out_size(out) {}

    ~PyCallable() {
        // The RHS may be destroyed with the GIL released (the adjoint's cloned
        // RHS is destroyed inside continuousAdjoint); drop the reference under
        // the GIL so the member destructor's dec_ref is a no-op.
        py::gil_scoped_acquire acquire;
        fn = py::object();
    }

    // The RHS is cloned by the integrator core (e.g. inside continuousAdjoint),
    // which may run with the GIL released; copying the py::object must therefore
    // (re-)acquire it for the refcount.  Reentrant no-op when already held.
    PyCallable(const PyCallable &other) : in_size(other.in_size), out_size(other.out_size) {
        py::gil_scoped_acquire acquire;
        fn = other.fn;
    }
    PyCallable &operator=(const PyCallable &other) {
        if (this == &other)
            return *this;
        py::gil_scoped_acquire acquire;
        fn = other.fn;
        in_size = other.in_size;
        out_size = other.out_size;
        return *this;
    }

    void operator()(const double *y, double *out) const {
        // The caller may have released the GIL (integrate/continuous_adjoint do,
        // so independent trajectories can solve in parallel); re-acquire it here
        // to call back into Python.  No-op when the current thread already holds
        // the GIL (e.g. during reset_integration).
        py::gil_scoped_acquire acquire;
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
             [](Integrator &it, double t_end) {
                 // Release the GIL so independent trajectories (each their own
                 // Integrator) solve concurrently; the RHS/Jacobian callbacks
                 // re-acquire it when they call back into Python.
                 py::gil_scoped_release release;
                 return it.performIntegration(t_end, 10000000);
             },
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
                 // Snapshot the raw pointers first (pybind11 NumPy access needs
                 // the GIL), then release it so the backward adjoint loop over
                 // trajectories parallelizes.  The release is scoped so the GIL
                 // is re-acquired before the array arguments are destroyed.
                 double *seed = seed_weights.mutable_data();
                 double *adj = adj_y0.mutable_data();
                 bool ok;
                 {
                     py::gil_scoped_release release;
                     ok = it.continuousAdjoint(seed, adj, nullptr);
                 }
                 return ok;
             },
             py::arg("seed_weights"), py::arg("adj_y0"))
        .def("set_algorithm_settings",
             [](Integrator &it, unsigned max_order, double mach_prec, bool hit_end,
                unsigned ew_index) {
                 return it.setAlgorithmSettings(max_order, mach_prec, hit_end, ew_index);
             },
             py::arg("max_order"), py::arg("mach_prec"), py::arg("hit_end") = true,
             py::arg("ew_index") = 1);

    // Matrix-free Newton (T32): f is the RHS, jv is the J·v (Jacobian-vector
    // product).  The Newton solve uses a GMRES Krylov solver with a J-free
    // identity preconditioner, so the full Jacobian is never formed — one JVP per
    // Krylov iteration.  Used by solve_ivp's solver="matrix_free".
    m.def("matrix_free_integrator",
          [](py::object f, py::object jv, std::size_t n) {
              auto jv_fn = [jv, n](const double *y, const double *v, double *Jv) {
                  // integrate() releases the GIL; re-acquire it to call the
                  // Python JVP back.
                  py::gil_scoped_acquire acquire;
                  py::array_t<double> y_arr(n, const_cast<double *>(y));
                  py::array_t<double> v_arr(n, const_cast<double *>(v));
                  py::array_t<double> r = py::array_t<double>::ensure(jv(y_arr, v_arr));
                  if (r.size() != static_cast<py::ssize_t>(n))
                      throw std::runtime_error("jv must return n doubles");
                  std::copy(r.data(), r.data() + n, Jv);
              };
              auto dummy_jac = [n](const double *, double *J) { std::fill(J, J + n * n, 0.0); };
              yabi::UserODERHS<double> rhs(n, PyCallable{f, n, n}, dummy_jac);
              rhs.setJacobianVectorProduct(jv_fn);

              auto *it = new Integrator(rhs);
              it->setJacobianBackend(yabi::JacobianBackend::MatrixFree);
              auto *solver = new yabi::GmresLinearSolver<double>();
              solver->setTolerance(1e-10, 0.0, 200, 30);
              // J-free identity preconditioner, borrowed by the solver (setUser
              // Preconditioner stores a raw pointer) and kept alive for the
              // module's lifetime.  Each integrator gets its own instance (the
              // preconditioner stores its n across update/apply, so it is NOT
              // stateless and cannot be shared).  The vector is append-only and
              // guarded so concurrent construction (parallel batch) is safe.
              static std::mutex keep_alive_mutex;
              static std::vector<std::shared_ptr<yabi::FunctionPreconditioner<double>>> keep_alive;
              auto prec = std::make_shared<yabi::FunctionPreconditioner<double>>(
                  false, [](const double *r, double *z, unsigned int sz) {
                      std::copy(r, r + sz, z);
                      return true;
                  });
              {
                  std::lock_guard<std::mutex> lock(keep_alive_mutex);
                  keep_alive.push_back(prec);
              }
              solver->setUserPreconditioner(prec.get());
              it->setLinearSolver(solver);
              return it;
          },
          py::arg("f"), py::arg("jv"), py::arg("n"));

#ifdef YABI_HAS_CUDA
    // T54 — dense Newton solve offloaded to the GPU via cuSOLVER (CudaLinearSolver).
    // f/jac still run on the host (numpy); only the O(n³) factorize + solve are
    // on-device.  Used by solve_ivp's device="cuda" (dense).
    m.def("cuda_integrator",
          [](py::object f, py::object jac, std::size_t n) {
              yabi::UserODERHS<double> rhs(n, PyCallable{f, n, n}, PyCallable{jac, n, n * n});
              auto *it = new Integrator(rhs);
              it->setLinearSolver(new yabi::CudaLinearSolver<double>());
              return it;
          },
          py::arg("f"), py::arg("jac"), py::arg("n"));
#endif
}
