# YABI test suite.
#
# Tiers (ctest labels):
#   unit       isolated component contracts; no integrator, no CppAD where possible
#   component  cross-component integration (native sparse, SuiteSparse)
#   system     mathematical benchmarks and regressions (canonical ODE problems)
#   example    self-checking examples from the top-level CMakeLists
#
# Run one tier with, for example, `ctest --test-dir build -L unit`.

function(yabi_register_test source label)
    get_filename_component(name "${source}" NAME_WE)
    set(target "yabi_test_${name}")
    add_executable("${target}" "${source}")
    if(YABI_USE_CPPAD)
        target_link_libraries("${target}" PRIVATE yabi)
    else()
        target_link_libraries("${target}" PRIVATE yabi_pure)
    endif()
    add_test(NAME "${name}" COMMAND "${target}")
    set_tests_properties("${name}" PROPERTIES LABELS "${label}")
endfunction()

# ---------------------------------------------------------------------------
# Unit tier — one executable per component, each independently runnable.
# ---------------------------------------------------------------------------
file(GLOB YABI_UNIT_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/unit/test_*.cpp")
if(NOT YABI_USE_CPPAD)
    # The RHS adapter unit test builds a CppAD tape.
    list(FILTER YABI_UNIT_SOURCES EXCLUDE REGEX "/test_cppad_rhs_sparsity\\.cpp$")
endif()
foreach(source ${YABI_UNIT_SOURCES})
    yabi_register_test("${source}" "unit")
endforeach()

# ---------------------------------------------------------------------------
# Component tier — multi-component behaviour that is still smaller than a full
# mathematical benchmark.
# ---------------------------------------------------------------------------
if(YABI_USE_CPPAD)
    yabi_register_test("${CMAKE_CURRENT_SOURCE_DIR}/component/native_sparse.cpp" "component;sparse")
    yabi_register_test("${CMAKE_CURRENT_SOURCE_DIR}/component/matrix_free_newton.cpp"
        "component;matrixfree")
endif()

# Native CSC + symbolic reuse requires both SuiteSparse solvers.  Registered
# only when they are present, so the default build stays dependency-light.
if(YABI_USE_CPPAD AND UMFPACK_INCLUDE_DIR AND UMFPACK_LIBRARY AND KLU_INCLUDE_DIR AND KLU_LIBRARY)
    add_executable(yabi_test_suitesparse_sparse
        "${CMAKE_CURRENT_SOURCE_DIR}/component/suitesparse_sparse.cpp")
    target_include_directories(yabi_test_suitesparse_sparse PRIVATE
        ${UMFPACK_INCLUDE_DIR} ${KLU_INCLUDE_DIR})
    target_link_libraries(yabi_test_suitesparse_sparse PRIVATE yabi
        ${UMFPACK_LIBRARY} ${KLU_LIBRARY} amd cholmod colamd ccolamd camd suitesparseconfig
        lapack blas cppad_lib m)
    add_test(NAME suitesparse_sparse COMMAND yabi_test_suitesparse_sparse)
    set_tests_properties(suitesparse_sparse PROPERTIES LABELS "component;sparse;suitesparse")
endif()

# ---------------------------------------------------------------------------
# System tier — the mathematical testbench.
# ---------------------------------------------------------------------------
if(YABI_USE_CPPAD)
    file(GLOB YABI_SYSTEM_CPPAD CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/system/[0-9]*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/system/manual_*.cpp")
    foreach(source ${YABI_SYSTEM_CPPAD})
        yabi_register_test("${source}" "system")
    endforeach()
endif()

# Pure/user_ode benchmarks run in both configurations.
file(GLOB YABI_SYSTEM_PURE CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/system/pure_*.cpp")
foreach(source ${YABI_SYSTEM_PURE})
    yabi_register_test("${source}" "system;pure")
endforeach()
