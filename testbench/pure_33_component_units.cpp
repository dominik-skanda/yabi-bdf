// Focused unit tests for the single-responsibility components extracted from
// BDFIntegrator (T30).
//
// CppAD-free by design: this file is built and run in both the CppAD and the
// pure (YABI_USE_CPPAD=OFF) configuration.  Each component gets at least one
// test that exercises its own state/operations in isolation from the integrator
// façade.
#include <yabi/log.hpp>

#include <yabi/detail/adjoint_engine.hpp>
#include <yabi/detail/dense_linear_solver.ipp>
#include <yabi/detail/integrator_config.hpp>
#include <yabi/detail/newton_solver.hpp>
#include <yabi/detail/nordsieck_state.hpp>
#include <yabi/detail/replay_tape.hpp>
#include <yabi/detail/sensitivity_engine.hpp>
#include <yabi/detail/step_controller.hpp>

#include <cmath>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char *what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << std::endl;
        ++failures;
    }
}

bool near(double actual, double expected, double tol = 1e-12) {
    return std::fabs(actual - expected) <= tol;
}

// ---------------------------------------------------------------------------
// IntegratorConfig / IntegratorStatistics
// ---------------------------------------------------------------------------
void testConfig() {
    yabi::detail::IntegratorConfig<double> config;

    check(config.START_STEP_SIZE_SAFETY == 0.90, "config.START_STEP_SIZE_SAFETY");
    check(config.NEWTON_STEP_REDUCTION == 0.25, "config.NEWTON_STEP_REDUCTION");
    check(config.STEP_NOT_ACCEPTED_REDUCTION == 0.25, "config.STEP_NOT_ACCEPTED_REDUCTION");
    check(config.STEP_NOT_ACCEPTED_SAVETY_STEP == 0.50, "config.STEP_NOT_ACCEPTED_SAVETY_STEP");
    check(config.ORDER_STAY_SAVETY_STEP == 0.90, "config.ORDER_STAY_SAVETY_STEP");
    check(config.ORDER_UP_SAVETY_STEP == 0.90, "config.ORDER_UP_SAVETY_STEP");
    check(config.ORDER_DOWN_SAVETY_STEP == 0.90, "config.ORDER_DOWN_SAVETY_STEP");
    check(config.allowedSTEPS == 10, "config.allowedSTEPS");
    check(config.NEWTON_SAFETY == -1.00, "config.NEWTON_SAFETY");
    check(config.NEWTON_CONVERGED == 0.01, "config.NEWTON_CONVERGED");
    check(config.ACCEPT_NEWTON_CONVERGENCE_RATE == 0.30, "config.ACCEPT_NEWTON_CONVERGENCE_RATE");
    check(config.allowedNEWTON == 10, "config.allowedNEWTON");
    check(config.MAXNEWTONITERATION == 3, "config.MAXNEWTONITERATION");
    check(config.MAXORDER == 6, "config.MAXORDER");
    check(config.MINORDER == 1, "config.MINORDER");
    check(config.MACHINEPRECISION == std::numeric_limits<double>::epsilon(),
          "config.MACHINEPRECISION");
    check(config.HITSOLUTION == false, "config.HITSOLUTION");
    check(config.ERROR_WEIGHTS_INDEX == 1, "config.ERROR_WEIGHTS_INDEX");
    check(config.MINTAYLOR == 1, "config.MINTAYLOR");
    check(config.updateEveryP == false, "config.updateEveryP");
    check(config.RESTRICT_ORDER_1_RATIO == 3.000, "config.RESTRICT_ORDER_1_RATIO");
    check(config.RESTRICT_ORDER_2_RATIO == 2.414, "config.RESTRICT_ORDER_2_RATIO");
    check(config.RESTRICT_ORDER_3_RATIO == 1.618, "config.RESTRICT_ORDER_3_RATIO");
    check(config.RESTRICT_ORDER_4_RATIO == 1.280, "config.RESTRICT_ORDER_4_RATIO");
    check(config.RESTRICT_ORDER_5_RATIO == 1.127, "config.RESTRICT_ORDER_5_RATIO");
    check(config.RESTRICT_ORDER_6_RATIO == 1.064, "config.RESTRICT_ORDER_6_RATIO");

    // Config is plain mutable state, independent of the integrator.
    config.MAXORDER = 4;
    config.HITSOLUTION = true;
    check(config.MAXORDER == 4, "config.MAXORDER writable");
    check(config.HITSOLUTION == true, "config.HITSOLUTION writable");

    yabi::detail::IntegratorStatistics stats;
    check(stats.acceptedSteps == 0 && stats.rejectedSteps == 0 && stats.rejectedNewtonSteps == 0 &&
              stats.updateJacobian == 0 && stats.updateP == 0 && stats.newtonIterations == 0,
          "statistics zero-initialised");
}

// ---------------------------------------------------------------------------
// ReplayTape / ReplayTapeStore
// ---------------------------------------------------------------------------
void testReplayTape() {
    yabi::detail::ReplayTapeStore<double> store;
    check(store.empty(), "store starts empty");
    check(store.size() == 0, "store size starts at 0");

    yabi::detail::ReplayTape<double> *first = store.create();
    yabi::detail::ReplayTape<double> *second = store.create();
    yabi::detail::ReplayTape<double> *third = store.create();

    check(!store.empty(), "store not empty after create");
    check(store.size() == 3, "store size after three creates");

    // Nodes are individually heap-allocated, so later growth must not move them.
    first->timePoint = 1.0;
    second->timePoint = 2.0;
    third->timePoint = 3.0;
    first->forwardLINK = second;
    second->backLINK = first;
    second->forwardLINK = third;
    third->backLINK = second;
    store.create(); // force the unique_ptr vector to grow
    check(first->timePoint == 1.0 && second->timePoint == 2.0 && third->timePoint == 3.0,
          "node addresses stable across store growth");
    check(first->forwardLINK == second && third->backLINK == second,
          "tape links preserved across store growth");
    check(store.size() == 4, "store size after growth");

    // Vector-owned per-step data is independent per node.
    first->endSolution.assign(2, 0.5);
    second->endSolution.assign(2, 1.5);
    check(first->endSolution[0] == 0.5 && second->endSolution[0] == 1.5,
          "per-node buffers are independent");

    store.clear();
    check(store.empty() && store.size() == 0, "store clear releases every node");
}

// ---------------------------------------------------------------------------
// NordsieckState
// ---------------------------------------------------------------------------
void testNordsieckState() {
    yabi::detail::NordsieckState<double> nordsieck;
    nordsieck.initialize(2);

    check(nordsieck.z_predictor.size() == 14, "z_predictor size n*7");
    check(nordsieck.z_corrector.size() == 14, "z_corrector size n*7");
    check(nordsieck.z_corrector_old.size() == 14, "z_corrector_old size n*7");
    check(nordsieck.corrector.size() == 2, "corrector size n");
    check(nordsieck.corrector_old.size() == 2, "corrector_old size n");
    check(nordsieck.solution_Vector.size() == 2, "solution_Vector size n");
    check(nordsieck.corrector[0] == 1.0 && nordsieck.corrector[1] == 1.0,
          "initialize sets unit initial correction");
    check(nordsieck.z_predictor[0] == 0.0 && nordsieck.z_corrector[0] == 0.0,
          "initialize zeroes the history");

    // l_i for order 1: {1, h/h0}.
    nordsieck.order_corrector = 1;
    nordsieck.step_size_h = 0.5;
    nordsieck.stepSizeSet_new[0] = 0.5;
    check(nordsieck.calculate_l_vector(), "calculate_l_vector order 1");
    check(near(nordsieck.l_i[0], 1.0) && near(nordsieck.l_i[1], 1.0), "l_i order 1 values");

    // l_i for order 2: {1, 3, 2} with h=0.5, h0=0.5, h1=0.25.
    nordsieck.order_corrector = 2;
    nordsieck.stepSizeSet_new[0] = 0.5;
    nordsieck.stepSizeSet_new[1] = 0.25;
    check(nordsieck.calculate_l_vector(), "calculate_l_vector order 2");
    check(near(nordsieck.l_i[0], 1.0) && near(nordsieck.l_i[1], 3.0) && near(nordsieck.l_i[2], 2.0),
          "l_i order 2 values");

    // Predictor for order 1: z_pred[:,0] = z_corr[:,0] + z_corr[:,1],
    // z_pred[:,1] = z_corr[:,1] (constant+linear extrapolation).
    nordsieck.order_corrector = 1;
    nordsieck.z_corrector[0] = 1.0; // col 0, species 0
    nordsieck.z_corrector[1] = 2.0; // col 0, species 1
    nordsieck.z_corrector[2] = 0.5; // col 1, species 0
    nordsieck.z_corrector[3] = -1.0;
    check(nordsieck.calculate_next_predictor(2, 0, nullptr, nullptr),
          "calculate_next_predictor order 1");
    check(near(nordsieck.z_predictor[0], 1.5) && near(nordsieck.z_predictor[1], 1.0) &&
              near(nordsieck.z_predictor[2], 0.5) && near(nordsieck.z_predictor[3], -1.0),
          "predictor constant+linear extrapolation");

    // Corrector: z_corr = z_pred + corrector * l_i^T.
    nordsieck.order_corrector = 1;
    nordsieck.z_predictor[0] = 1.0;
    nordsieck.z_predictor[1] = 2.0;
    nordsieck.z_predictor[2] = 3.0;
    nordsieck.z_predictor[3] = 4.0;
    nordsieck.corrector[0] = 0.5;
    nordsieck.corrector[1] = -0.5;
    nordsieck.l_i[0] = 1.0;
    nordsieck.l_i[1] = 0.0;
    check(nordsieck.calculate_next_corrector(2, 0, nullptr, nullptr, nullptr),
          "calculate_next_corrector");
    check(near(nordsieck.z_corrector[0], 1.5) && near(nordsieck.z_corrector[1], 1.5) &&
              near(nordsieck.z_corrector[2], 3.0) && near(nordsieck.z_corrector[3], 4.0),
          "corrector adds correction*l_i to predictor");

    // Dense output: at t_i[1] only the 0-th coefficient contributes; one step
    // later the first coefficient contributes with weight 1.
    nordsieck.order_corrector = 1;
    nordsieck.step_size_h = 1.0;
    nordsieck.t_i[1] = 5.0;
    nordsieck.z_corrector[0] = 1.0;
    nordsieck.z_corrector[1] = 2.0;
    nordsieck.z_corrector[2] = 3.0;
    nordsieck.z_corrector[3] = 4.0;
    check(nordsieck.calculate_output_vector(2, 0, nullptr, nullptr, 5.0),
          "calculate_output_vector at t_i[1]");
    check(near(nordsieck.solution_Vector[0], 1.0) && near(nordsieck.solution_Vector[1], 2.0),
          "dense output at t_i[1]");
    check(nordsieck.calculate_output_vector(2, 0, nullptr, nullptr, 6.0),
          "calculate_output_vector one step ahead");
    check(near(nordsieck.solution_Vector[0], 4.0) && near(nordsieck.solution_Vector[1], 6.0),
          "dense output one step ahead");
}

// ---------------------------------------------------------------------------
// StepController
// ---------------------------------------------------------------------------
void testStepController() {
    yabi::detail::NordsieckState<double> nordsieck;
    yabi::detail::IntegratorConfig<double> config;
    yabi::detail::StepController<double> controller(nordsieck, config);

    // Mode 0: max(existing weight, |y|, atol).
    double weights0[2] = {0.1, 5.0};
    const double y0[2] = {-2.0, 1.0};
    const double atol0[2] = {0.5, 0.3};
    check(
        yabi::detail::StepController<double>::calculateErrorWeights(2, 0, weights0, y0, atol0, 0.5),
        "calculateErrorWeights mode 0");
    check(near(weights0[0], 2.0) && near(weights0[1], 5.0),
          "mode 0 raises to max(|y|, atol, weight)");

    // Mode 1: |y| + atol/rtol.
    double weights1[2] = {0.0, 0.0};
    const double y1[2] = {-2.0, 1.0};
    const double atol1[2] = {0.5, 0.3};
    check(
        yabi::detail::StepController<double>::calculateErrorWeights(2, 1, weights1, y1, atol1, 0.5),
        "calculateErrorWeights mode 1");
    check(near(weights1[0], 3.0) && near(weights1[1], 1.6), "mode 1 uses |y| + atol/rtol");
}

// ---------------------------------------------------------------------------
// Shared dummy dependencies for the Newton/sensitivity/adjoint engines.
// The engines only borrow these in their constructors; the tests below touch
// only engine-owned state.
// ---------------------------------------------------------------------------
struct Fixture {
    yabi::detail::IntegratorConfig<double> config;
    yabi::detail::IntegratorStatistics stats;
    yabi::detail::NordsieckState<double> nordsieck;
    std::unique_ptr<yabi::AbstractRHS<double>> RHS;
    std::unique_ptr<yabi::LinearSolver<double>> linearSolver;
    unsigned int numOfSpecies = 2;
    unsigned int RHS_type = 0;
    unsigned int sizeOfParameter = 1;
    std::vector<double> parameter;
    std::vector<double> error_weights;
    std::vector<double> absolut_error;
    double relative_error = 0.0;
    double **error_weights_derivatives = nullptr;
    double **absolut_error_derivatives = nullptr;
    double *relative_error_derivatives = nullptr;
    double t_ende = 0.0;
    double minimalEta = 0.0;
    double reachableError = 0.0;
    std::vector<double> reachableError_derivatives;
    unsigned int numOfDerivatives = 0;
    bool direct = false;
    bool taping = false;
    bool internalError = false;
    double **z_predictor_derivatives = nullptr;
    double **corrector_derivatives = nullptr;
    double **p_seed_derivatives = nullptr;
    yabi::detail::ReplayTape<double> *startTAPE = nullptr;
    yabi::detail::ReplayTape<double> *endTAPE = nullptr;
    yabi::detail::ReplayTapeStore<double> replayTapeStore;
    std::list<std::unique_ptr<yabi::AbstractRHS<double>>> cppADFunList;
    std::list<std::unique_ptr<yabi::AbstractRHS<double>>>::iterator intermediateCppADFunIterator;
    yabi::AbstractRHS<double> *start_CppAD_tape = nullptr;
    double startTape_stepSize = 0.0;
    double start_integration_time = 0.0;
    yabi::detail::ReplayTape<double> *intermediateTAPE = nullptr;
    int blasONE = 1;
    double blasScalarONE = 1.0;
    double blasScalarMinusONE = -1.0;
    double blasScalarZERO = 0.0;

    yabi::detail::StepController<double> stepController;
    yabi::detail::SensitivityEngine<double> sensitivity;
    yabi::detail::NewtonSolver<double> newton;
    yabi::detail::AdjointEngine<double> adjoint;

    Fixture()
        : stepController(nordsieck, config),
          sensitivity(nordsieck, config, RHS, numOfSpecies, RHS_type, sizeOfParameter, parameter,
                      start_integration_time, direct, internalError, startTAPE, intermediateTAPE,
                      start_CppAD_tape, startTape_stepSize, cppADFunList,
                      intermediateCppADFunIterator, blasONE, blasScalarONE, blasScalarMinusONE,
                      blasScalarZERO),
          newton(nordsieck, config, stats, RHS, linearSolver, numOfSpecies, RHS_type,
                 sizeOfParameter, parameter, error_weights, absolut_error, relative_error,
                 error_weights_derivatives, absolut_error_derivatives, relative_error_derivatives,
                 t_ende, minimalEta, reachableError, reachableError_derivatives, numOfDerivatives,
                 direct, taping, z_predictor_derivatives, corrector_derivatives, p_seed_derivatives,
                 startTAPE, endTAPE, replayTapeStore, cppADFunList, intermediateCppADFunIterator,
                 internalError),
          adjoint(sensitivity, nordsieck, newton, RHS, numOfSpecies, RHS_type, sizeOfParameter,
                  parameter, config, direct, internalError, startTape_stepSize, start_CppAD_tape,
                  cppADFunList, intermediateCppADFunIterator, blasONE, blasScalarONE,
                  blasScalarMinusONE, blasScalarZERO) {
        intermediateCppADFunIterator = cppADFunList.end();
        nordsieck.initialize(numOfSpecies);
        parameter.assign(sizeOfParameter, 0.0);
        error_weights.assign(numOfSpecies, 1.0);
        absolut_error.assign(numOfSpecies, 0.0);
        newton.P.assign(numOfSpecies * numOfSpecies, 0.0);
        newton.J.assign(numOfSpecies * numOfSpecies, 0.0);
        newton.ip.assign(numOfSpecies, 0);
        newton.scaling_D1.assign(numOfSpecies, 0);
        newton.scaling_D2.assign(numOfSpecies, 0);
    }
};

// ---------------------------------------------------------------------------
// NewtonSolver
// ---------------------------------------------------------------------------
void testNewtonSolver() {
    Fixture f;

    check(f.newton.P.size() == 4, "NewtonSolver owns P");
    check(f.newton.J.size() == 4, "NewtonSolver owns J");
    check(f.newton.ip.size() == 2, "NewtonSolver owns pivots");
    check(f.newton.scaling_D1.size() == 2 && f.newton.scaling_D2.size() == 2,
          "NewtonSolver owns scalings");
    check(!f.newton.setJacobian && !f.newton.setNewP && !f.newton.newtonIteration_Started,
          "NewtonSolver iteration flags start clear");
    check(f.newton.convergenceIndicator == 0.0 && f.newton.newtonFAIL == 0 &&
              f.newton.stepFAIL == 0,
          "NewtonSolver diagnostics start at zero");

    // P = [[2, 1], [1, 3]] (column-major), unit error weights.  The historical
    // scaling sweeps rows: D1 = -ceil(logb(row_sum)) = {-1, -2}, then
    // P <- scalbn(P, D1) per row.
    f.newton.P[0] = 2.0;
    f.newton.P[1] = 1.0;
    f.newton.P[2] = 1.0;
    f.newton.P[3] = 3.0;
    check(f.newton.calculateNewtonScaling(), "calculateNewtonScaling succeeds");
    check(f.newton.scaling_D2[0] == 0 && f.newton.scaling_D2[1] == 0, "unit weights leave D2 zero");
    check(f.newton.scaling_D1[0] == -1 && f.newton.scaling_D1[1] == -2,
          "row scalings from row sums");
    check(near(f.newton.P[0], 1.0) && near(f.newton.P[1], 0.25) && near(f.newton.P[2], 0.5) &&
              near(f.newton.P[3], 0.75),
          "P scaled by row scalings");
}

// ---------------------------------------------------------------------------
// SensitivityEngine
// ---------------------------------------------------------------------------
void testSensitivityEngine() {
    Fixture f;

    f.sensitivity.initialize(3);
    check(f.sensitivity.forward_solution_Vector.size() == 3,
          "SensitivityEngine sizes forward solution vector");
    check(f.sensitivity.correctorVector_tape.size() == 3,
          "SensitivityEngine sizes replay corrector vector");
    check(f.sensitivity.forward_solution_Vector[0] == 0.0, "forward solution starts at zero");

    f.sensitivity.forward_solution_Vector[0] = 7.0;
    f.sensitivity.forward_solution_Vector[2] = -3.0;
    f.sensitivity.zeroForwardSolution();
    check(f.sensitivity.forward_solution_Vector[0] == 0.0 &&
              f.sensitivity.forward_solution_Vector[2] == 0.0,
          "zeroForwardSolution clears every entry");

    // A new tape invalidates the forward replay and the previous order.
    f.sensitivity.getTAPE = true;
    f.sensitivity.lastForwardOrder = 5;
    f.sensitivity.beginNewTape();
    check(!f.sensitivity.getTAPE && f.sensitivity.lastForwardOrder == 0,
          "beginNewTape resets replay state");

    // Per-tape replay buffers are released on tape reset.
    yabi::detail::ReplayTape<double> node;
    f.sensitivity.adjointTapePointer = &node;
    f.sensitivity.z_predictor_tape.push_back(1.0);
    f.sensitivity.z_corrector_tape.push_back(2.0);
    f.sensitivity.P_tape.push_back(3.0);
    f.sensitivity.ip_tape.push_back(4);
    f.sensitivity.scaling_D1_tape.push_back(5);
    f.sensitivity.scaling_D2_tape.push_back(6);
    f.sensitivity.resetForwardBuffers();
    check(f.sensitivity.adjointTapePointer == nullptr, "resetForwardBuffers clears cursor");
    check(f.sensitivity.z_predictor_tape.empty() && f.sensitivity.z_corrector_tape.empty() &&
              f.sensitivity.P_tape.empty() && f.sensitivity.ip_tape.empty() &&
              f.sensitivity.scaling_D1_tape.empty() && f.sensitivity.scaling_D2_tape.empty(),
          "resetForwardBuffers releases replay buffers");

    // Releasing with no simultaneous derivatives is a safe no-op.
    check(f.sensitivity.freeSimultaneousDerivatives(),
          "freeSimultaneousDerivatives with no derivatives");
    check(f.sensitivity.numOfDerivatives == 0 && f.sensitivity.z_predictor_derivatives == nullptr,
          "freeSimultaneousDerivatives keeps empty state null");
}

// ---------------------------------------------------------------------------
// AdjointEngine
// ---------------------------------------------------------------------------
void testAdjointEngine() {
    Fixture f;

    check(f.adjoint.solution_adjoint_species == nullptr, "AdjointEngine species starts null");
    check(f.adjoint.solution_adjoint_parameter == nullptr, "AdjointEngine parameter starts null");
    check(f.adjoint.stored_adjoint_order == 0, "AdjointEngine order starts at zero");

    // Simulate a completed reverse sweep and check release() owns the cleanup.
    f.adjoint.solution_adjoint_species = new double *[f.numOfSpecies];
    for (unsigned int i = 0; i < f.numOfSpecies; i++)
        f.adjoint.solution_adjoint_species[i] = new double[2]{1.0, 2.0};
    f.adjoint.solution_adjoint_parameter = new double *[f.sizeOfParameter];
    for (unsigned int i = 0; i < f.sizeOfParameter; i++)
        f.adjoint.solution_adjoint_parameter[i] = new double[2]{3.0, 4.0};
    f.adjoint.stored_adjoint_order = 2;

    f.adjoint.release();
    check(f.adjoint.solution_adjoint_species == nullptr, "release clears species arrays");
    check(f.adjoint.solution_adjoint_parameter == nullptr, "release clears parameter arrays");
    check(f.adjoint.stored_adjoint_order == 0, "release clears stored order");

    // Idempotent and safe on already-empty state.
    f.adjoint.release();
    check(f.adjoint.solution_adjoint_species == nullptr, "release is idempotent");
}

// ---------------------------------------------------------------------------
// LinearSolver (DenseLUSolver)
// ---------------------------------------------------------------------------
void testDenseLinearSolver() {
    yabi::DenseLUSolver<double> solver;
    check(solver.initialize(2), "DenseLUSolver::initialize");

    // P = [[2, 1], [1, 3]] (column-major), b = [5, 6] => x = [1.8, 1.4].
    double P[4] = {2.0, 1.0, 1.0, 3.0};
    int ipiv[2] = {0, 0};
    long d1[2] = {0, 0};
    long d2[2] = {0, 0};
    check(solver.factorize(P, ipiv, d1, d2, 2), "DenseLUSolver::factorize");

    double b[2] = {5.0, 6.0};
    check(solver.solve(P, ipiv, d1, d2, b, 2, 1) == 0, "DenseLUSolver::solve");
    check(near(b[0], 1.8) && near(b[1], 1.4), "DenseLUSolver solves P*x=b");

    // Multiple right-hand sides: same matrix, b = [5, 7; 6, 10] column-major.
    double P2[4] = {2.0, 1.0, 1.0, 3.0};
    int ipiv2[2] = {0, 0};
    check(solver.factorize(P2, ipiv2, d1, d2, 2), "DenseLUSolver::factorize (2 rhs)");
    double rhs[4] = {5.0, 6.0, 7.0, 10.0};
    check(solver.solve(P2, ipiv2, d1, d2, rhs, 2, 2) == 0, "DenseLUSolver::solve (2 rhs)");
    check(near(rhs[0], 1.8) && near(rhs[1], 1.4) && near(rhs[2], 2.2) && near(rhs[3], 2.6),
          "DenseLUSolver solves both right-hand sides");
}

} // namespace

int main() {
    testConfig();
    testReplayTape();
    testNordsieckState();
    testStepController();
    testNewtonSolver();
    testSensitivityEngine();
    testAdjointEngine();
    testDenseLinearSolver();

    if (failures != 0) {
        std::cerr << failures << " component check(s) failed" << std::endl;
        return 1;
    }

    std::cout << "PASS: component unit tests (config, replay tape, Nordsieck, step control, "
                 "Newton, sensitivity, adjoint, linear solver)"
              << std::endl;
    return 0;
}
