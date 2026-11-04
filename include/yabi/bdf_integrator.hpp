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

    #include <yabi/detail/integrator_config.hpp>
    #include <yabi/detail/nordsieck_state.hpp>
    #include <yabi/detail/replay_tape.hpp>
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
    bool setStepSettings(Scalar NEWTON_STEP_REDUCTION, Scalar STEP_NOT_ACCEPTED_REDUCTION,
                         Scalar STEP_NOT_ACCEPTED_SAVETY_STEP, Scalar ORDER_STAY_SAVETY_STEP,
                         Scalar ORDER_UP_SAVETY_STEP, Scalar ORDER_DOWN_SAVETY_STEP,
                         unsigned int allowedSTEPS);
    bool setNewtonSettings(Scalar NEWTON_SAFETY, Scalar NEWTON_CONVERGED,
                           Scalar ACCEPT_NEWTON_CONVERGENCE_RATE, unsigned int allowedNEWTON,
                           unsigned int MAXNEWTONITERATION, bool updateEveryP);
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

    // Set a custom linear solver (e.g. for future sparse support). The
    // integrator takes ownership of the passed solver.
    void setLinearSolver(LinearSolver<Scalar> *solver) {
        linearSolver.reset(solver);
        if (linearSolver) {
            linearSolver->initialize(numOfSpecies);
        }
    }
    LinearSolver<Scalar> *getLinearSolver() const { return linearSolver.get(); }

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

    // Simultaneous Derivatives
    Scalar **z_predictor_derivatives;
    Scalar **z_corrector_derivatives;
    Scalar **z_corrector_derivatives_old;
    Scalar **y_seed_derivatives;
    Scalar **p_seed_derivatives;

    unsigned int numOfDerivatives;

    bool isIntegrattionStarted;
    bool isIntegrattionPerformed;

    unsigned int numOfSpecies;

    // Simultaneous Derivatives
    Scalar **corrector_derivatives;

    // Simultaneous Derivatives
    Scalar **corrector_old_derivatives;

    std::vector<Scalar> error_weights;
    std::vector<Scalar> absolut_error;
    Scalar relative_error;
    // Simultaneous Derivatives
    Scalar **error_weights_derivatives;
    Scalar **absolut_error_derivatives;
    Scalar *relative_error_derivatives; // non-owning: points into the user's seed array

    Scalar t_ende;
    // Simultaneous Derivatives
    Scalar **solution_Vector_derivatives;

    Scalar minimalEta;

    // Simultaneous Derivatives

    Scalar max_StepLength;
    Scalar start_integration_time;
    std::vector<Scalar> P;
    std::vector<Scalar> J;
    std::vector<int> ip;
    std::vector<long> scaling_D1;
    std::vector<long> scaling_D2;
    bool setJacobian;
    bool setNewP;
    bool newtonIteration_Started;
    Scalar convergenceIndicator;
    unsigned newtonFAIL;
    unsigned stepFAIL;
    unsigned int RHS_type; // 0 - Autonom; 1 time dependent; 2 time and parameter dependent;
    std::vector<Scalar> parameter;
    unsigned int sizeOfParameter;

    Scalar reachableError;
    std::vector<Scalar> reachableError_derivatives;

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
    std::vector<Scalar> correctorVector_tape;
    Scalar startTape_stepSize;

    Scalar l_i_tape[7];
    Scalar time_tape;
    Scalar stepSizeSet_tape[6];
    std::vector<Scalar> z_predictor_tape;
    std::vector<Scalar> z_corrector_tape;
    Scalar step_size_h_tape;
    std::vector<Scalar> forward_solution_Vector;
    unsigned int order_corrector_tape;
    bool taping;
    bool getTAPE;
    unsigned int lastForwardOrder;
    std::vector<Scalar> P_tape;
    std::vector<int> ip_tape;
    std::vector<long> scaling_D1_tape;
    std::vector<long> scaling_D2_tape;
    bool internalError;

    bool deleteOldTape();
    bool createNewTape();
    inline bool calculate_next_predictor_forward();
    inline bool calculate_l_vector_forward();
    bool calculate_corrector_vector_forward(unsigned int order, Scalar *y_seed,
                                            Scalar *parameter_seed);
    inline bool calculate_next_corrector_forward();
    inline bool calculate_output_vector_forward();

    // adjoint

    ReplayTape *adjointTapePointer;

    Scalar **solution_adjoint_species;
    Scalar **solution_adjoint_parameter;

    unsigned int possible_adjoint_order;
    unsigned int stored_adjoint_order;
    bool start_adjoint_from_forward_mode;

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
