#include <yabi/log.hpp>
#include <yabi/linear_algebra.hpp>
#include <yabi/linear_solver.hpp>
#include <yabi/detail/dense_linear_solver.ipp>
/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  Copyright (C) 2010-2012 Dominik Skanda
 *
 *  SPDX-License-Identifier: Apache-2.0
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
#ifndef YABI_BDF_INTEGRATOR_HPP
    #define YABI_BDF_INTEGRATOR_HPP

    #include <yabi/abstract_rhs.hpp>
    #ifndef YABI_NO_CPPAD
        #include <yabi/cppad_rhs.hpp>
    #endif

    #include <cmath>
    #include <cstddef>
    #include <cstdlib>
    #include <iostream>
    #include <limits>
    #include <list>
    #include <memory>
    #include <string>
    #include <vector>

    #include <yabi/detail/adjoint_engine.hpp>
    #include <yabi/detail/continuous_adjoint.hpp>
    #include <yabi/detail/integrator_config.hpp>
    #include <yabi/detail/newton_solver.hpp>
    #include <yabi/detail/nordsieck_state.hpp>
    #include <yabi/detail/replay_tape.hpp>
    #include <yabi/detail/sensitivity_engine.hpp>
    #include <yabi/detail/step_controller.hpp>
    #include <yabi/jacobian_backend.hpp>
    #include <yabi/linear_algebra.hpp>

namespace yabi {

// Optional sensitivity/seed configuration for resetIntegration.  Defaults
// reproduce the common "no sensitivity" tail
// (..., false, false, NULL, NULL, NULL, NULL, 0).
template <typename Scalar> struct IntegrationOptions {
    bool taping = false;
    bool direct = false;
    Scalar **y_seed_list = nullptr;
    Scalar **p_seed_list = nullptr;
    Scalar *relative_error_derivatives = nullptr;
    Scalar **absolut_error_derivatives = nullptr;
    unsigned int numOfderivatives = 0;
};

template <class Scalar> class BDFIntegrator {
    typedef std::list<std::unique_ptr<AbstractRHS<Scalar>>> cppADFunListType;

  public:
    BDFIntegrator(const AbstractRHS<Scalar> &RHS);
    #ifndef YABI_NO_CPPAD
    // Convenience: wrap a CppAD ADFun (the historical constructor).
    BDFIntegrator(CppAD::ADFun<Scalar> &RHS) : BDFIntegrator<Scalar>(CppADRHS<Scalar>(RHS)) {}
    #endif
    ~BDFIntegrator();
    bool resetIntegration(Scalar *y_start, Scalar max_StepLength, Scalar relative_error,
                          Scalar *absolut_error, bool taping, bool direct, Scalar **y_seed_list,
                          Scalar **p_seed_list, Scalar *relative_error_derivatives,
                          Scalar **absolut_error_derivatives,
                          unsigned int numOfderivatives); // autonom
    bool resetIntegration(Scalar *y_start, Scalar T_start, Scalar max_StepLength_,
                          Scalar relative_error_, Scalar *absolut_error_, bool taping, bool direct,
                          Scalar **y_seed_list, Scalar **p_seed_list,
                          Scalar *relative_error_derivatives, Scalar **absolut_error_derivatives,
                          unsigned int numOfderivatives); // time dependent
    bool resetIntegration(Scalar *y_start, Scalar T_start, Scalar *parameter,
                          Scalar max_StepLength_, Scalar relative_error_, Scalar *absolut_error_,
                          bool taping, bool direct, Scalar **y_seed_list, Scalar **p_seed_list,
                          Scalar *relative_error_derivatives, Scalar **absolut_error_derivatives,
                          unsigned int numOfderivatives); // time dependent and parameter dependent
    // Additive convenience overloads: the same three signatures with the seven
    // trailing sensitivity-seed arguments replaced by a single options struct.
    bool resetIntegration(Scalar *y_start, Scalar max_StepLength, Scalar relative_error,
                          Scalar *absolut_error, const IntegrationOptions<Scalar> &options); // autonom
    bool resetIntegration(Scalar *y_start, Scalar T_start, Scalar max_StepLength,
                          Scalar relative_error, Scalar *absolut_error,
                          const IntegrationOptions<Scalar> &options); // time dependent
    bool resetIntegration(Scalar *y_start, Scalar T_start, Scalar *parameter,
                          Scalar max_StepLength, Scalar relative_error, Scalar *absolut_error,
                          const IntegrationOptions<Scalar> &options); // time + parameter
    bool performIntegration(const Scalar t_ende, unsigned int max_StepSize);
    bool getSolutionVector(Scalar *solution_Vector);
    // Dense output: evaluate the Nordsieck interpolant at t, which must lie
    // within the last accepted step [t_i[1] - step_size_h, t_i[1]].  Returns
    // false (and logs) if integration has not run or t is outside that step.
    bool getSolutionAt(Scalar t, Scalar *solution_Vector_at_t);
    bool getSimultaneousDerivativesVector(Scalar **solution_Vector_derivatives);
    bool getForwardSolutionVector(Scalar *forward_solution_Vector);
    bool getReverseSolutionVector(Scalar **solution_adjoint_species,
                                  Scalar **solution_adjoint_parameter);
    bool forward(unsigned int order, Scalar *y_seed, Scalar *parameter_seed, bool reset);
    bool reverse(unsigned int order, Scalar *seed_weights);
    // First-order *continuous* adjoint (T25): integrate lambda' = -J(y)^T lambda
    // backward from t_end, giving d(sum_i seed_weights[i] y_i(T)) / d y0 (and
    // /dp) stably for long/stiff problems — the stable complement to the
    // discrete-adjoint reverse() (IND).  Reuses the forward BDF in reversed
    // time with a recorded trajectory; requires the forward integration to have
    // run to t_ende first.  adjoint_parameter may be null (then only the state
    // adjoint is computed).
    bool continuousAdjoint(Scalar *seed_weights, Scalar *adjoint_y0, Scalar *adjoint_parameter);
    // Second-order *continuous* adjoint (T26): the Hessian-vector product of
    // g = sum_i seed_weights[i] y_i(t_end) with the direction seed_direction,
    // computed stably for long/stiff problems (the stable complement to the
    // discrete-adjoint reverse(2)).  Requires performIntegration to have run
    // and the backend to provide a Hessian-in-direction hook (hessianVectorProduct).
    bool secondOrderAdjoint(Scalar *seed_weights, Scalar *seed_direction, Scalar *hessian_y0);
    // Second-order *forward* sensitivity (T27): d^2 y / d tau^2 in the seed
    // direction, computed stably for long/stiff problems (the stable complement
    // to forward(2), which is A-stability-bound).  Requires performIntegration
    // to have run and the backend to provide a second-order directional
    // derivative hook (secondOrderDirectional).
    bool secondOrderForward(Scalar *seed_direction, Scalar *second_derivative);
    // Higher-order *forward* sensitivity (T40): d^k y / d tau^k in the seed
    // direction for arbitrary k >= 1, computed stably for long/stiff problems
    // (the stable complement to forward(k), which is A-stability-bound at
    // order >= 3).  k = 1 is the first-order tangent; k = 2 coincides with
    // secondOrderForward.  For k >= 2 the backend must provide the k-th-order
    // directional source via AbstractRHS::orderKDirectional (CppAD automatic,
    // pure UserODERHS via the setOrderKDirectional analytic hook; fails loudly
    // without it).  Requires performIntegration to have run.
    bool forwardOrder(unsigned int k, Scalar *seed_direction, Scalar *kth_derivative);
    bool setStepSettings(Scalar NEWTON_STEP_REDUCTION, Scalar STEP_NOT_ACCEPTED_REDUCTION,
                         Scalar STEP_NOT_ACCEPTED_SAVETY_STEP, Scalar ORDER_STAY_SAVETY_STEP,
                         Scalar ORDER_UP_SAVETY_STEP, Scalar ORDER_DOWN_SAVETY_STEP,
                         unsigned int allowedSTEPS);
    bool setNewtonSettings(Scalar NEWTON_SAFETY, Scalar NEWTON_CONVERGED,
                           Scalar ACCEPT_NEWTON_CONVERGENCE_RATE, unsigned int allowedNEWTON,
                           unsigned int MAXNEWTONITERATION, bool updateEveryP);
    // Opt-in damped Newton (T45): backtrack (halve) the Newton correction while
    // the BDF residual fails to decrease, up to `max_backtracks` times.  Applies
    // to the plain state solve (non-taped); off by default so the historical
    // full-step behaviour is preserved bit-for-bit.  Returns false if
    // `max_backtracks == 0`.
    bool setNewtonLineSearch(bool enable, unsigned int max_backtracks = 6);
    bool setAlgorithmSettings(unsigned int MAXORDER, Scalar MACHINEPRECISION, bool HITSOLUTION,
                              unsigned int ERROR_WEIGHTS_INDEX);
    // Force a fixed BDF order (1..6): sets both MINORDER and MAXORDER to q, so
    // the controller ramps up to q and then stays there (no increase above, no
    // decrease below).  Call after (or instead of) setAlgorithmSettings.
    bool setFixedOrder(unsigned int order);
    bool setRestrictOrderRatio(Scalar restrictRatioOrder1, Scalar restrictRatioOrder2,
                               Scalar restrictRatioOrder3, Scalar restrictRatioOrder4,
                               Scalar restrictRatioOrder5, Scalar restrictRatioOrder6);
    bool setMinTaylorCapacity(unsigned int MINTAYLOR);
    bool setMinMemoryADFunObjectes(unsigned int MINnumberADFunObjects);

    // Thread-parallel sensitivity propagation (T35): set the number of threads
    // used for the independent forward/simultaneous sensitivity directions.
    // 1 = serial (default); >1 = chunked parallel solve+update.  The direction
    // residual (CppAD tape replay) stays serial.  Requires a thread-safe linear
    // solver (the default dense LAPACK solve is).
    bool setSensitivityThreads(unsigned int threads);

    // Set a custom linear solver (e.g. for future sparse support). The
    // integrator takes ownership of the passed solver.  Installing a solver
    // whose CSC path is the production path (UMFPACK/KLU) automatically
    // enables native sparse assembly; call setSparseAssembly(false) afterwards
    // to keep the dense assembly path with that solver.
    void setLinearSolver(LinearSolver<Scalar> *solver) {
        linearSolver.reset(solver);
        if (linearSolver)
            linearSolver->initialize(numOfSpecies);
        // Auto: reflect the solver's preferences by only ever *enabling* a
        // toggle (never disabling) — the historical behaviour, which does not
        // override a manual setSparseAssembly/setMatrixFreeAssembly toggle.  An
        // explicit JacobianBackend keeps its toggles untouched; the solve
        // validates the choice.
        if (newtonSolver.jacobianBackend == JacobianBackend::Auto) {
            if (linearSolver->prefersSparseFactorization())
                newtonSolver.sparseAssembly = true;
            if (linearSolver->supportsMatrixFree())
                newtonSolver.matrixFreeAssembly = true;
        }
    }
    LinearSolver<Scalar> *getLinearSolver() const { return linearSolver.get(); }

    // Explicitly select how the Jacobian is formed for the plain state Newton
    // solve (T33).  Dense/Sparse/MatrixFree force that path and fail loudly when
    // the RHS + installed solver cannot satisfy it; Auto (the default) keeps the
    // historical capability-driven selection.  The sparse/matrix-free paths cover
    // the plain state solve only — sensitivity solves always use the dense J/H
    // path (documented, no silent loss).
    void setJacobianBackend(JacobianBackend backend) {
        newtonSolver.jacobianBackend = backend;
        switch (backend) {
        case JacobianBackend::Dense:
            newtonSolver.sparseAssembly = false;
            newtonSolver.matrixFreeAssembly = false;
            break;
        case JacobianBackend::Sparse:
            newtonSolver.sparseAssembly = true;
            newtonSolver.matrixFreeAssembly = false;
            break;
        case JacobianBackend::MatrixFree:
            newtonSolver.sparseAssembly = false;
            newtonSolver.matrixFreeAssembly = true;
            break;
        case JacobianBackend::Auto:
        default:
            applyAutoBackend();
            break;
        }
    }
    JacobianBackend getJacobianBackend() const { return newtonSolver.jacobianBackend; }

    // Native sparse Newton assembly (T31): detect the Jacobian pattern once,
    // assemble J/P in CSC and reuse the solver's symbolic factorization while
    // the pattern is unchanged.  It is additionally gated on the RHS exposing
    // a sparsity pattern and on the solve being a plain state solve (no
    // sensitivity replay tape).  With the default DenseLUSolver this uses the
    // CSC reference path, which is useful for validation but not faster.
    void setSparseAssembly(bool enabled) { newtonSolver.sparseAssembly = enabled; }
    bool getSparseAssembly() const { return newtonSolver.sparseAssembly; }

    // Matrix-free state Newton solve (T32): when a Krylov solver is installed,
    // P*v is evaluated through the RHS Jacobian-vector product instead of a
    // stored dense J.  J/P are still assembled in CSC for scaling and the
    // preconditioner, and the path is used only for plain state solves; the
    // sensitivity equations keep their explicit J/H path.
    void setMatrixFreeAssembly(bool enabled) { newtonSolver.matrixFreeAssembly = enabled; }
    bool getMatrixFreeAssembly() const { return newtonSolver.matrixFreeAssembly; }

    bool printStatistic(std::ostream &outstream = std::cout, std::string delim = " ") {
        outstream << "INTEGRATION STATISTIC:" << delim << " " << std::endl
                  << "ACCEPTED STEPS:" << delim << stats.acceptedSteps << std::endl
                  << "REJECTED STEPS:" << delim << stats.rejectedSteps << std::endl
                  << "REJECTED NEWTON STEPS:" << delim << stats.rejectedNewtonSteps << std::endl
                  << "JACOBIAN UPDATED:" << delim << stats.updateJacobian << std::endl
                  << "MATRIX P UPDATED:" << delim << stats.updateP << std::endl;
        return true;
    };

    // ---- Diagnostics (programmatic; the quantities printStatistic and the
    // step/order controller already track, exposed as read-only getters) ----

    // BDF order of the last accepted step (1..MAXORDER).
    unsigned int getCurrentOrder() const {
        return nordsieck.order_corrector;
    }

    // Step size of the last accepted step (== the size of the next step).
    Scalar getCurrentStepSize() const {
        return nordsieck.step_size_h;
    }

    // Total nonlinear (Newton) iterations performed across the integration.
    unsigned int getNewtonIterations() const { return stats.newtonIterations; }

    // Accepted / rejected step counters.
    unsigned int getAcceptedSteps() const { return stats.acceptedSteps; }
    unsigned int getRejectedSteps() const { return stats.rejectedSteps; }
    unsigned int getRejectedNewtonSteps() const { return stats.rejectedNewtonSteps; }

    // Number of Jacobian (re)evaluations.  Each is a full J assembly.
    unsigned int getJacobianEvaluations() const { return stats.updateJacobian; }

    // Number of Newton-matrix (P) updates performed from a *reused* Jacobian
    // (i.e. P re-formed and re-factorized without reassembling J).
    unsigned int getMatrixPUpdates() const { return stats.updateP; }

    // Total number of Newton-matrix (LU) factorizations.  Every Jacobian
    // assembly re-forms and factorizes P, and every P-only update factorizes
    // again, so the total is updateJacobian + updateP.
    unsigned int getFactorizations() const { return stats.updateJacobian + stats.updateP; }

  private:
    bool setSimultaneousDerivatives(Scalar **y_seed_derivatives_, Scalar **p_seed_derivatives_,
                                    Scalar *relative_error_derivatives_,
                                    Scalar **absolut_error_derivatives_,
                                    unsigned int numOfDerivatives_);
    bool freeSimultaneousDerivatives();

    // Re-derive the sparse/matrix-free assembly toggles from the installed
    // solver's capabilities (the historical Auto behaviour; T33).
    void applyAutoBackend() {
        newtonSolver.sparseAssembly = linearSolver && linearSolver->prefersSparseFactorization();
        newtonSolver.matrixFreeAssembly = linearSolver && linearSolver->supportsMatrixFree();
    }
    std::unique_ptr<AbstractRHS<Scalar>> RHS;

    // Pluggable linear solver (default DenseLUSolver). Mathematically identical
    // to original; allows future sparse solvers without changing Newton logic.
    std::unique_ptr<LinearSolver<Scalar>> linearSolver;

    inline bool calculate_next_predictor();
    bool calculate_corrector_vector();
    inline bool calculate_next_corrector();
    unsigned int accept_corrector_vector();
    inline bool calculate_output_vector();
    bool prepare_next_step();
    inline bool calculate_l_vector();
    bool estimateStartStepsize();
    inline bool calculate_newton_scaling();
    inline bool calculate_error_weights(Scalar *errorWeights, Scalar *yWeight,
                                        Scalar *absoluteError, Scalar &relativeError);

    // Simultaneous-derivative state lives in sensitivityEngine (see
    // detail/sensitivity_engine.hpp).

    bool isIntegrattionStarted;
    bool isIntegrattionPerformed;

    unsigned int numOfSpecies;

    std::vector<Scalar> error_weights;
    std::vector<Scalar> absolut_error;
    Scalar relative_error;

    Scalar t_ende;

    Scalar minimalEta;

    Scalar max_StepLength;
    Scalar start_integration_time;
    unsigned int RHS_type; // 0 - Autonom; 1 time dependent; 2 time and parameter dependent;
    std::vector<Scalar> parameter;
    unsigned int sizeOfParameter;

    Scalar reachableError;

    // SENSITIVITY GENERATION

    // forward

    // Per-step replay record (see detail/replay_tape.hpp) and the pool that
    // owns every node of the current integration.
    using ReplayTape = detail::ReplayTape<Scalar>;
    using ReplayTapeStore = detail::ReplayTapeStore<Scalar>;

    // Allocate a node, initialise it, append it after endTAPE and make it the
    // new end.  Callers keep their own "can we append?" checks.
    ReplayTape *createReplayTapeNode();

    ReplayTape *startTAPE;
    ReplayTape *intermediateTAPE;
    ReplayTape *endTAPE;
    ReplayTapeStore replayTapeStore;

    cppADFunListType cppADFunList;
    typename cppADFunListType::iterator intermediateCppADFunIterator;

    bool direct;

    AbstractRHS<Scalar> *start_CppAD_tape;
    Scalar startTape_stepSize;

    bool taping;
    bool internalError;

    bool deleteOldTape();
    bool createNewTape();

    // LAPACK WRAPPER

    int blasONE;
    Scalar blasScalarONE;
    Scalar blasScalarMinusONE;
    Scalar blasScalarZERO;

    // Nordsieck history array, step/order/time state and its algebra (see
    // detail/nordsieck_state.hpp).
    detail::NordsieckState<Scalar> nordsieck;

    // Integrator configuration and run statistics (own single-responsibility
    // value types; see detail/integrator_config.hpp).
    detail::IntegratorConfig<Scalar> config;
    detail::IntegratorStatistics stats;

    // Step-size/order selection on top of the Nordsieck state (see
    // detail/step_controller.hpp).
    detail::StepController<Scalar> stepController;

    // Forward/simultaneous sensitivity state and replay (see
    // detail/sensitivity_engine.hpp).  Declared before NewtonSolver because the
    // Newton code borrows the simultaneous-derivative mirrors.
    detail::SensitivityEngine<Scalar> sensitivityEngine;

    // Newton iteration, Jacobian assembly and Newton-owned matrix state (see
    // detail/newton_solver.hpp).
    detail::NewtonSolver<Scalar> newtonSolver;

    // Reverse/discrete-adjoint replay and its result arrays (see
    // detail/adjoint_engine.hpp).  Declared after the forward engine and the
    // Newton solver because it borrows both.
    detail::AdjointEngine<Scalar> adjointEngine;

    // Forward trajectory recorded during performIntegration for the continuous
    // adjoint (see detail/continuous_adjoint.hpp).  Cleared at the start of
    // each performIntegration run.
    detail::Trajectory<Scalar> trajectory_;
};

} // namespace yabi

    // Implementation of the class template.  The algorithm lives in these files
    // so that the public header shows the interface and nothing else.
    #include <yabi/detail/core.ipp>
    #include <yabi/detail/newton.ipp>
    #include <yabi/detail/sensitivity_forward.ipp>
    #include <yabi/detail/sensitivity_reverse.ipp>
    #include <yabi/detail/sensitivity_setup.ipp>
    #include <yabi/detail/step_control.ipp>

    // Only double is instantiated by the library.  Other scalar types can be
    // instantiated by a client by defining YABI_NO_EXTERN_TEMPLATE before
    // including this header.
    #ifndef YABI_NO_EXTERN_TEMPLATE
extern template class yabi::BDFIntegrator<double>;
    #endif

#endif
