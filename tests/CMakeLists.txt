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
    # The CppAD RHS adapter tests build a CppAD tape.
    list(FILTER YABI_UNIT_SOURCES EXCLUDE REGEX "/test_cppad_.*\\.cpp$")
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
    # T42 — the four first-order sensitivity modes agree (forward/reverse/
    # simultaneous/continuous adjoint).
    yabi_register_test("${CMAKE_CURRENT_SOURCE_DIR}/component/sensitivity_agreement.cpp"
        "component;sensitivity")
    # T42 — CppAD (AD) and UserODERHS (analytic) backends agree.
    yabi_register_test("${CMAKE_CURRENT_SOURCE_DIR}/component/rhs_backend_agreement.cpp"
        "component;backend")
endif()

# T42 — solver agreement matrix: one loop test over every available solver.
# The external solvers are compiled in only when their flag is on (YABI_HAVE_*).
if(YABI_USE_CPPAD)
    add_executable(yabi_test_solver_matrix
        "${CMAKE_CURRENT_SOURCE_DIR}/component/solver_matrix.cpp")
    target_link_libraries(yabi_test_solver_matrix PRIVATE yabi)
    if(YABI_USE_CUDA)
        target_compile_definitions(yabi_test_solver_matrix PRIVATE YABI_HAVE_CUDA)
        target_include_directories(yabi_test_solver_matrix PRIVATE ${CUDA_INCLUDE_DIR})
        target_link_libraries(yabi_test_solver_matrix PRIVATE ${CUSOLVER_LIBRARY} ${CUDART_LIBRARY})
    endif()
    if(YABI_USE_AMGCL)
        target_compile_definitions(yabi_test_solver_matrix PRIVATE YABI_HAVE_AMGCL)
        target_include_directories(yabi_test_solver_matrix PRIVATE ${AMGCL_INCLUDE_DIR} ${BOOST_INCLUDE_DIR})
        if(OpenMP_CXX_FOUND)
            target_link_libraries(yabi_test_solver_matrix PRIVATE OpenMP::OpenMP_CXX)
        endif()
    endif()
    if(YABI_USE_GINKGO)
        target_compile_definitions(yabi_test_solver_matrix PRIVATE YABI_HAVE_GINKGO)
        target_include_directories(yabi_test_solver_matrix PRIVATE ${GINKGO_INCLUDE_DIR})
        target_link_libraries(yabi_test_solver_matrix PRIVATE
            ${GINKGO_LIBRARY} ${GINKGO_DEVICE_LIBRARY} ${GINKGO_OMP_LIBRARY} OpenMP::OpenMP_CXX)
        target_link_options(yabi_test_solver_matrix PRIVATE "-Wl,-rpath,${GINKGO_LIBRARY_DIR}")
    endif()
    if(YABI_USE_CUDSS)
        target_compile_definitions(yabi_test_solver_matrix PRIVATE YABI_HAVE_CUDSS)
        target_include_directories(yabi_test_solver_matrix PRIVATE ${CUDSS_INCLUDE_DIR} ${CUDA_INCLUDE_DIR})
        target_link_libraries(yabi_test_solver_matrix PRIVATE ${CUDSS_LIBRARY} ${CUDART_LIBRARY})
        target_link_options(yabi_test_solver_matrix PRIVATE "-Wl,-rpath,${CUDSS_LIBRARY_DIR}")
    endif()
    add_test(NAME solver_matrix COMMAND yabi_test_solver_matrix)
    set_tests_properties(solver_matrix PROPERTIES LABELS "component;solvers")
endif()

# Pure (CppAD-free) path: UserODERHS supplies the analytic sparsity pattern and
# J*v, so the same matrix-free Newton solve must run in both configurations.
yabi_register_test("${CMAKE_CURRENT_SOURCE_DIR}/component/matrix_free_pure.cpp"
    "component;matrixfree;pure")
yabi_register_test("${CMAKE_CURRENT_SOURCE_DIR}/component/matrix_free_jfree.cpp"
    "component;matrixfree;pure")

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

# Experimental CUDA (cuSOLVER) solver test (T36): registered only when
# YABI_USE_CUDA=ON, so the default build stays CUDA-free.
if(YABI_USE_CUDA)
    add_executable(yabi_test_cuda_solver
        "${CMAKE_CURRENT_SOURCE_DIR}/component/cuda_solver.cpp")
    target_include_directories(yabi_test_cuda_solver PRIVATE ${CUDA_INCLUDE_DIR})
    target_link_libraries(yabi_test_cuda_solver PRIVATE yabi ${CUSOLVER_LIBRARY} ${CUDART_LIBRARY})
    add_test(NAME cuda_solver COMMAND yabi_test_cuda_solver)
    set_tests_properties(cuda_solver PROPERTIES LABELS "component;cuda")

    # T36c — parallel multi-instance GPU solves (thread-safe per-instance streams).
    add_executable(yabi_test_cuda_parallel
        "${CMAKE_CURRENT_SOURCE_DIR}/component/cuda_parallel.cpp")
    target_include_directories(yabi_test_cuda_parallel PRIVATE ${CUDA_INCLUDE_DIR})
    target_link_libraries(yabi_test_cuda_parallel PRIVATE yabi ${CUSOLVER_LIBRARY} ${CUDART_LIBRARY})
    add_test(NAME cuda_parallel COMMAND yabi_test_cuda_parallel)
    set_tests_properties(cuda_parallel PROPERTIES LABELS "component;cuda")
endif()

# T36d — external open-source sparse solver adapter (AMGCL), registered only when
# YABI_USE_AMGCL=ON, so the default build stays dependency-free.
if(YABI_USE_AMGCL)
    add_executable(yabi_test_amgcl_solver
        "${CMAKE_CURRENT_SOURCE_DIR}/component/amgcl_solver.cpp")
    target_include_directories(yabi_test_amgcl_solver PRIVATE ${AMGCL_INCLUDE_DIR} ${BOOST_INCLUDE_DIR})
    target_link_libraries(yabi_test_amgcl_solver PRIVATE yabi)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(yabi_test_amgcl_solver PRIVATE OpenMP::OpenMP_CXX)
    endif()
    add_test(NAME amgcl_solver COMMAND yabi_test_amgcl_solver)
    set_tests_properties(amgcl_solver PROPERTIES LABELS "component;amgcl")
endif()

# T36d — external Ginkgo sparse-solver adapter, registered only when
# YABI_USE_GINKGO=ON, so the default build stays dependency-free.
if(YABI_USE_GINKGO)
    add_executable(yabi_test_ginkgo_solver
        "${CMAKE_CURRENT_SOURCE_DIR}/component/ginkgo_solver.cpp")
    target_include_directories(yabi_test_ginkgo_solver PRIVATE ${GINKGO_INCLUDE_DIR})
    target_link_libraries(yabi_test_ginkgo_solver PRIVATE yabi
        ${GINKGO_LIBRARY} ${GINKGO_DEVICE_LIBRARY} ${GINKGO_OMP_LIBRARY} OpenMP::OpenMP_CXX)
    target_link_options(yabi_test_ginkgo_solver PRIVATE "-Wl,-rpath,${GINKGO_LIBRARY_DIR}")
    add_test(NAME ginkgo_solver COMMAND yabi_test_ginkgo_solver)
    set_tests_properties(ginkgo_solver PROPERTIES LABELS "component;ginkgo")
endif()

# T36d — external cuDSS sparse direct solver adapter, registered only when
# YABI_USE_CUDSS=ON, so the default build stays dependency-free.
if(YABI_USE_CUDSS)
    add_executable(yabi_test_cudss_solver
        "${CMAKE_CURRENT_SOURCE_DIR}/component/cudss_solver.cpp")
    target_include_directories(yabi_test_cudss_solver PRIVATE ${CUDSS_INCLUDE_DIR} ${CUDA_INCLUDE_DIR})
    target_link_libraries(yabi_test_cudss_solver PRIVATE yabi ${CUDSS_LIBRARY} ${CUDART_LIBRARY})
    target_link_options(yabi_test_cudss_solver PRIVATE "-Wl,-rpath,${CUDSS_LIBRARY_DIR}")
    add_test(NAME cudss_solver COMMAND yabi_test_cudss_solver)
    set_tests_properties(cudss_solver PROPERTIES LABELS "component;cudss")
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
