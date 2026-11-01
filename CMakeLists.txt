cmake_minimum_required(VERSION 3.16)
project(YABI VERSION 0.2.0 LANGUAGES CXX)

option(YABI_BUILD_EXAMPLES "Build YABI example programs" ON)
option(YABI_BUILD_TESTBENCH "Build YABI testbench" ON)

# The examples self-check (they return non-zero on failure), so they are
# registered as ctest cases just like the testbench programs.  enable_testing()
# is therefore unconditional: YABI_BUILD_TESTBENCH=OFF must not silently
# disable the example checks.
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
    endforeach()
endif()

# ---------------------------------------------------------------------------
# Testbench
# ---------------------------------------------------------------------------
if(YABI_BUILD_TESTBENCH)
    # Numbered tests
    file(GLOB YABI_TEST_SOURCES CONFIGURE_DEPENDS testbench/[0-9]*.cpp)
    # Manual/pure sensitivity tests always included
    file(GLOB YABI_MANUAL_SOURCES CONFIGURE_DEPENDS testbench/manual_*.cpp testbench/pure_*.cpp)
    list(APPEND YABI_TEST_SOURCES ${YABI_MANUAL_SOURCES})
    foreach(source ${YABI_TEST_SOURCES})
        get_filename_component(name ${source} NAME_WE)
        add_executable(yabi_test_${name} ${source})
        if(YABI_USE_CPPAD)
            target_link_libraries(yabi_test_${name} PRIVATE yabi)
        else()
            target_link_libraries(yabi_test_${name} PRIVATE yabi_pure)
        endif()
        add_test(NAME ${name} COMMAND yabi_test_${name})
    endforeach()
endif()

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
