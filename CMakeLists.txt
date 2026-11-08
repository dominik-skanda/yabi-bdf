cmake_minimum_required(VERSION 3.16)
project(YABI VERSION 0.2.0 LANGUAGES CXX)

option(YABI_BUILD_EXAMPLES "Build YABI example programs" ON)
option(YABI_BUILD_TESTS "Build the unit/component/system test suite" ON)
# Backwards-compatible alias for the old flat testbench (now tests/system).
# Only consulted when supplied, so fresh configurations see a single toggle.
if(DEFINED YABI_BUILD_TESTBENCH)
    message(DEPRECATION "YABI_BUILD_TESTBENCH is deprecated; use YABI_BUILD_TESTS")
    set(YABI_BUILD_TESTS "${YABI_BUILD_TESTBENCH}")
endif()

# The examples self-check (they return non-zero on failure), so they are
# registered as ctest cases just like the test suite.  enable_testing() is
# therefore unconditional: disabling tests must not silently disable the
# example checks.
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# ---------------------------------------------------------------------------
# CppAD is an external dependency and is deliberately not bundled with YABI.
# It is located through pkg-config, through CPPAD_ROOT, or through the usual
# compiler search paths.
# ---------------------------------------------------------------------------
set(CPPAD_ROOT "" CACHE PATH
    "Root of a CppAD installation (the directory containing include/ and lib/)")

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(CPPAD_PC QUIET cppad)
endif()

if(CPPAD_PC_FOUND)
    set(CPPAD_INCLUDE_DIR ${CPPAD_PC_INCLUDE_DIRS})
    set(CPPAD_LINK_LIBRARIES ${CPPAD_PC_LIBRARIES})
    link_directories(${CPPAD_PC_LIBRARY_DIRS})
else()
    find_path(CPPAD_INCLUDE_DIR cppad/cppad.hpp
        HINTS ${CPPAD_ROOT} $ENV{CPPAD_ROOT}
        PATH_SUFFIXES include)
    find_library(CPPAD_LIBRARY NAMES cppad_lib
        HINTS ${CPPAD_ROOT} $ENV{CPPAD_ROOT}
        PATH_SUFFIXES lib lib64 lib/${CMAKE_LIBRARY_ARCHITECTURE})
    if(CPPAD_LIBRARY)
        set(CPPAD_LINK_LIBRARIES ${CPPAD_LIBRARY})
    endif()
endif()

option(YABI_USE_CPPAD "Enable CppAD-based integrator (default ON)" ON)
if(YABI_USE_CPPAD)
if(NOT CPPAD_INCLUDE_DIR)
    message(FATAL_ERROR
        "CppAD was not found.\n"
        "Install CppAD (https://github.com/coin-or/CppAD) and make sure "
        "cppad/cppad.hpp is on the include path, or configure with "
        "-DCPPAD_ROOT=/path/to/cppad.")
endif()
message(STATUS "CppAD headers: ${CPPAD_INCLUDE_DIR}")
if(CPPAD_LINK_LIBRARIES)
    message(STATUS "CppAD library: ${CPPAD_LINK_LIBRARIES}")
endif()
else()
    message(STATUS "CppAD disabled (-DYABI_USE_CPPAD=OFF). Pure/manual formulation available.")
endif()

# ---------------------------------------------------------------------------
# Experimental CUDA (cuSOLVER) linear solver (T36).  OFF by default; the default
# build stays CUDA-free.  Only the host-side cuSOLVER/cuBLAS C API is used (no
# nvcc, no CUDA language), so we just need the headers + libcusolver + libcudart.
# ---------------------------------------------------------------------------
option(YABI_USE_CUDA "Build the experimental cuSOLVER linear solver + GPU test (T36)" OFF)
if(YABI_USE_CUDA)
    find_path(CUDA_INCLUDE_DIR cusolverDn.h
        PATHS /usr/local/cuda-12.6/include /usr/local/cuda/include $ENV{CUDA_HOME}
        PATH_SUFFIXES include)
    find_library(CUSOLVER_LIBRARY cusolver
        PATHS /usr/local/cuda-12.6/lib64 /usr/local/cuda/lib64 $ENV{CUDA_HOME}
        PATH_SUFFIXES lib64 lib)
    find_library(CUDART_LIBRARY cudart
        PATHS /usr/local/cuda-12.6/lib64 /usr/local/cuda/lib64 $ENV{CUDA_HOME}
        PATH_SUFFIXES lib64 lib)
    if(CUDA_INCLUDE_DIR AND CUSOLVER_LIBRARY AND CUDART_LIBRARY)
        message(STATUS "CUDA (cuSOLVER) found: ${CUDA_INCLUDE_DIR}")
    else()
        message(FATAL_ERROR
            "YABI_USE_CUDA=ON but CUDA (cusolverDn.h, libcusolver, libcudart) was not found.")
    endif()
endif()

# ---------------------------------------------------------------------------
# External open-source sparse solver adapter (T36d): AMGCL (header-only, MIT) +
# Boost headers.  OFF by default; the default build stays dependency-free.  The
# adapter is an EXAMPLE of the open solver architecture (swap the solver by
# swapping the adapter object).
# ---------------------------------------------------------------------------
option(YABI_USE_AMGCL "Build the external AMGCL sparse-solver adapter + test (T36d)" OFF)
if(YABI_USE_AMGCL)
    find_path(AMGCL_INCLUDE_DIR amgcl/backend/builtin.hpp
        PATHS $ENV{HOME}/.local/opt/amgcl /usr/local/include /usr/include
        PATH_SUFFIXES include)
    find_path(BOOST_INCLUDE_DIR boost/version.hpp
        PATHS $ENV{HOME}/.local/opt/boost /usr/local/include /usr/include
        PATH_SUFFIXES include)
    find_package(OpenMP QUIET)
    if(AMGCL_INCLUDE_DIR AND BOOST_INCLUDE_DIR)
        message(STATUS "AMGCL found: ${AMGCL_INCLUDE_DIR} (Boost: ${BOOST_INCLUDE_DIR})")
    else()
        message(FATAL_ERROR
            "YABI_USE_AMGCL=ON but AMGCL or Boost headers were not found (see README for provisioning).")
    endif()
endif()

option(YABI_USE_GINKGO "Build the external Ginkgo sparse-solver adapter + test (T36d)" OFF)
if(YABI_USE_GINKGO)
    find_path(GINKGO_INCLUDE_DIR ginkgo/ginkgo.hpp
        PATHS $ENV{HOME}/.local/opt/ginkgo /usr/local/include /usr/include
        PATH_SUFFIXES include)
    find_library(GINKGO_LIBRARY ginkgo PATHS $ENV{HOME}/.local/opt/ginkgo PATH_SUFFIXES lib)
    find_library(GINKGO_DEVICE_LIBRARY ginkgo_device
        PATHS $ENV{HOME}/.local/opt/ginkgo PATH_SUFFIXES lib)
    find_library(GINKGO_OMP_LIBRARY ginkgo_omp
        PATHS $ENV{HOME}/.local/opt/ginkgo PATH_SUFFIXES lib)
    find_package(OpenMP REQUIRED)
    if(GINKGO_INCLUDE_DIR AND GINKGO_LIBRARY AND GINKGO_OMP_LIBRARY)
        get_filename_component(GINKGO_LIBRARY_DIR "${GINKGO_LIBRARY}" DIRECTORY)
        message(STATUS "Ginkgo found: ${GINKGO_INCLUDE_DIR} (lib: ${GINKGO_LIBRARY_DIR})")
    else()
        message(FATAL_ERROR
            "YABI_USE_GINKGO=ON but Ginkgo was not found (see README for provisioning).")
    endif()
endif()

option(YABI_USE_CUDSS "Build the cuDSS sparse direct solver adapter + test (T36d)" OFF)
if(YABI_USE_CUDSS)
    find_path(CUDSS_INCLUDE_DIR cudss.h
        PATHS $ENV{HOME}/.local/opt/cudss /usr/local/include /usr/include PATH_SUFFIXES include)
    find_library(CUDSS_LIBRARY cudss
        PATHS $ENV{HOME}/.local/opt/cudss PATH_SUFFIXES lib lib64)
    # cuDSS needs CUDA (cudart) for the host-side cudaMalloc/cudaMemcpy.
    find_path(CUDA_INCLUDE_DIR cuda_runtime.h
        PATHS /usr/local/cuda-12.6/include /usr/local/cuda/include $ENV{CUDA_HOME} PATH_SUFFIXES include)
    find_library(CUDART_LIBRARY cudart
        PATHS /usr/local/cuda-12.6/lib64 /usr/local/cuda/lib64 $ENV{CUDA_HOME} PATH_SUFFIXES lib64 lib)
    if(CUDSS_INCLUDE_DIR AND CUDSS_LIBRARY AND CUDA_INCLUDE_DIR AND CUDART_LIBRARY)
        get_filename_component(CUDSS_LIBRARY_DIR "${CUDSS_LIBRARY}" DIRECTORY)
        message(STATUS "cuDSS found: ${CUDSS_INCLUDE_DIR} (lib: ${CUDSS_LIBRARY_DIR})")
    else()
        message(FATAL_ERROR
            "YABI_USE_CUDSS=ON but cuDSS or CUDA was not found (see README for provisioning).")
    endif()
endif()

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
add_library(yabi src/bdf_integrator.cpp)
add_library(YABI::yabi ALIAS yabi)

target_include_directories(yabi
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(yabi SYSTEM PUBLIC ${CPPAD_INCLUDE_DIR})
target_link_libraries(yabi
    PUBLIC
        ${LAPACK_LIBRARIES}
        ${BLAS_LIBRARIES}
        ${CPPAD_LINK_LIBRARIES}
)
set_target_properties(yabi PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)

# Pure library: the same integrator core compiled without CppAD (YABI_NO_CPPAD
# drops the cppad_rhs.hpp include and the CppAD::ADFun convenience constructor),
# so a user_ode.hpp / UserODERHS consumer can build and run without <cppad/cppad.hpp>.
add_library(yabi_pure src/bdf_integrator_pure.cpp)
target_include_directories(yabi_pure
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_definitions(yabi_pure PUBLIC YABI_NO_CPPAD)
target_link_libraries(yabi_pure
    PUBLIC
        ${LAPACK_LIBRARIES}
        ${BLAS_LIBRARIES}
)
set_target_properties(yabi_pure PROPERTIES POSITION_INDEPENDENT_CODE ON)

# T44 — telemetry harness: emits the machine-readable evidence table consumed by
# benchmarks/report.py, and self-checks tolerance adherence (error decreases as
# rtol is tightened; the tightest rtol reaches < 1e-6), so the evidence layer is
# CI-verified as a ctest under the "benchmark" label.
if(YABI_USE_CPPAD)
    add_executable(yabi_telemetry benchmarks/telemetry.cpp)
    target_link_libraries(yabi_telemetry PRIVATE yabi)
    add_test(NAME telemetry COMMAND yabi_telemetry)
    set_tests_properties(telemetry PROPERTIES LABELS "benchmark")
endif()

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
if(YABI_BUILD_EXAMPLES)
    foreach(example basic nonautonomous sensitivity_demo)
        add_executable(yabi_${example} examples/${example}.cpp)
        target_link_libraries(yabi_${example} PRIVATE yabi)
        list(APPEND YABI_EXAMPLE_TARGETS yabi_${example})
        # Each example checks its own result and returns non-zero on failure.
        add_test(NAME example_${example} COMMAND yabi_${example})
        set_tests_properties(example_${example} PROPERTIES LABELS "example")
    endforeach()
    # Trajectory dumper for the benchmark gallery (utility, not a ctest).
    add_executable(dump_trajectories examples/dump_trajectories.cpp)
    target_link_libraries(dump_trajectories PRIVATE yabi)
endif()

# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------
# tests/unit      isolated component contracts (CppAD-free where possible)
# tests/component cross-component integration (native sparse, SuiteSparse)
# tests/system    mathematical benchmarks (canonical ODE problems)
# add_subdirectory(tests) is at the end so the optional SuiteSparse probes have
# run before the component tests are configured.

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
if(YABI_USE_CPPAD)
install(TARGETS yabi
    EXPORT YABITargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
else()
install(TARGETS yabi_pure
    EXPORT YABITargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(DIRECTORY include/yabi DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT YABITargets
    NAMESPACE YABI::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/YABI
)

# Optional: UMFPACK sparse solver example (if SuiteSparse available)
find_path(UMFPACK_INCLUDE_DIR umfpack.h PATHS /usr/include/suitesparse)
find_library(UMFPACK_LIBRARY umfpack)
if(UMFPACK_INCLUDE_DIR AND UMFPACK_LIBRARY)
  add_executable(yabi_heat1d_umfpack examples/heat_1d_sparse_umfpack.cpp)
  target_include_directories(yabi_heat1d_umfpack PRIVATE ${UMFPACK_INCLUDE_DIR})
  target_link_libraries(yabi_heat1d_umfpack PRIVATE yabi ${UMFPACK_LIBRARY} amd cholmod colamd ccolamd camd suitesparseconfig lapack blas cppad_lib m)
endif()
if(UMFPACK_INCLUDE_DIR AND UMFPACK_LIBRARY)
  add_executable(yabi_heat1d_umfpack_sens examples/heat_1d_umfpack_forward_sens.cpp)
  target_include_directories(yabi_heat1d_umfpack_sens PRIVATE ${UMFPACK_INCLUDE_DIR})
  target_link_libraries(yabi_heat1d_umfpack_sens PRIVATE yabi ${UMFPACK_LIBRARY} amd cholmod colamd ccolamd camd suitesparseconfig lapack blas cppad_lib m)
endif()
if(UMFPACK_INCLUDE_DIR AND UMFPACK_LIBRARY)
  add_executable(yabi_heat1d_umfpack_sens2 examples/heat1d_umfpack_sens_simple.cpp)
  target_include_directories(yabi_heat1d_umfpack_sens2 PRIVATE ${UMFPACK_INCLUDE_DIR})
  target_link_libraries(yabi_heat1d_umfpack_sens2 PRIVATE yabi ${UMFPACK_LIBRARY} amd cholmod colamd ccolamd camd suitesparseconfig lapack blas cppad_lib m)
endif()
if(UMFPACK_INCLUDE_DIR AND UMFPACK_LIBRARY)
  add_executable(yabi_heat1d_umfpack_sens3 examples/heat1d_umfpack_sens2.cpp)
  target_include_directories(yabi_heat1d_umfpack_sens3 PRIVATE ${UMFPACK_INCLUDE_DIR})
  target_link_libraries(yabi_heat1d_umfpack_sens3 PRIVATE yabi ${UMFPACK_LIBRARY} amd cholmod colamd ccolamd camd suitesparseconfig lapack blas cppad_lib m)
endif()
if(UMFPACK_INCLUDE_DIR AND UMFPACK_LIBRARY)
  add_executable(yabi_heat1d_umfpack_sens4 examples/heat1d_umfpack_sens3.cpp)
  target_include_directories(yabi_heat1d_umfpack_sens4 PRIVATE ${UMFPACK_INCLUDE_DIR})
  target_link_libraries(yabi_heat1d_umfpack_sens4 PRIVATE yabi ${UMFPACK_LIBRARY} amd cholmod colamd ccolamd camd suitesparseconfig lapack blas cppad_lib m)
endif()
add_executable(yabi_heat1d_dense_lit examples/heat1d_dense_literature.cpp)
target_link_libraries(yabi_heat1d_dense_lit PRIVATE yabi cppad_lib m)
add_executable(yabi_heat1d_dense_analytic examples/heat1d_dense_analytic_jac.cpp)
target_link_libraries(yabi_heat1d_dense_analytic PRIVATE yabi cppad_lib m)
if(UMFPACK_INCLUDE_DIR AND UMFPACK_LIBRARY)
  add_executable(yabi_heat1d_umfpack_lit examples/heat1d_umfpack_literature.cpp)
  target_include_directories(yabi_heat1d_umfpack_lit PRIVATE ${UMFPACK_INCLUDE_DIR})
  target_link_libraries(yabi_heat1d_umfpack_lit PRIVATE yabi ${UMFPACK_LIBRARY} amd cholmod colamd ccolamd camd suitesparseconfig lapack blas cppad_lib m)
endif()
if(UMFPACK_INCLUDE_DIR AND UMFPACK_LIBRARY)
  add_executable(yabi_heat1d_umfpack_sens_dense_api examples/heat1d_umfpack_sens_dense_equiv.cpp)
  target_include_directories(yabi_heat1d_umfpack_sens_dense_api PRIVATE ${UMFPACK_INCLUDE_DIR})
  target_link_libraries(yabi_heat1d_umfpack_sens_dense_api PRIVATE yabi ${UMFPACK_LIBRARY} amd cholmod colamd ccolamd camd suitesparseconfig lapack blas cppad_lib m)
endif()

# Optional SuiteSparse solvers for the component test tier.
find_path(KLU_INCLUDE_DIR klu.h PATHS /usr/include/suitesparse)
find_library(KLU_LIBRARY klu)

# -----------------------------------------------------------------
# Plot runner (matplotlib / numpy / pandas) — external, not required
# -----------------------------------------------------------------
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  find_path(PYTHON_MATPLOTLIB matplotlib __init__.py PATHS /usr/lib/python3/dist-packages /usr/local/lib/python3.14/site-packages)
  if(PYTHON_MATPLOTLIB)
    add_custom_target(plot_all
      COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/pyplot/plot_all.py
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      COMMENT "Plotting yabi-bdf examples to plots/ (requires matplotlib/numpy/pandas)")
  else()
    message(STATUS "matplotlib not found in python3 — plot_all target skipped; install with pip install matplotlib numpy pandas")
  endif()
endif()

# CSV dump for plotting
# CSV dump for plotting

# Plot examples (additional targets, not part of tests)
add_executable(plot_brusselator_traj examples/plot_brusselator_traj.cpp)
target_link_libraries(plot_brusselator_traj PRIVATE yabi cppad_lib m)
add_executable(plot_robertson_traj examples/plot_robertson_traj.cpp)
target_link_libraries(plot_robertson_traj PRIVATE yabi cppad_lib m)

# ---------------------------------------------------------------------------
# Tests (unit / component / system)
# ---------------------------------------------------------------------------
if(YABI_BUILD_TESTS)
    add_subdirectory(tests)
endif()
