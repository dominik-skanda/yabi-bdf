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

    #include <cppad/cppad.hpp>

    #include <cmath>
    #include <cstddef>
    #include <cstdlib>
    #include <iostream>
    #include <limits>
    #include <list>
    #include <string>
    #include <vector>

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
    typedef std::list<CppAD::ADFun<Scalar> *> cppADFunListType;

  public:
    BDFIntegrator(CppAD::ADFun<Scalar> &RHS);
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
        if (linearSolver) {
            delete linearSolver;
        }
        linearSolver = solver;
        if (linearSolver) {
            linearSolver->initialize(numOfSpecies);
        }
    }
    LinearSolver<Scalar> *getLinearSolver() const { return linearSolver; }

    bool printStatistic(std::ostream &outstream = std::cout, std::string delim = " ") {
        outstream << "INTEGRATION STATISTIC:" << delim << " " << std::endl
                  << "ACCEPTED STEPS:" << delim << acceptedSteps << std::endl
                  << "REJECTED STEPS:" << delim << rejectedSteps << std::endl
                  << "REJECTED NEWTON STEPS:" << delim << rejectedNewtonSteps << std::endl
                  << "JACOBIAN UPDATED:" << delim << updateJacobian << std::endl
                  << "MATRIX P UPDATED:" << delim << updateP << std::endl;
        return true;
    };

    // ---- Diagnostics (programmatic; the quantities printStatistic and the
    // step/order controller already track, exposed as read-only getters) ----

    // BDF order of the last accepted step (1..MAXORDER).
    unsigned int getCurrentOrder() const { return order_corrector; }

    // Step size of the last accepted step (== the size of the next step).
    Scalar getCurrentStepSize() const { return step_size_h; }

    // Total nonlinear (Newton) iterations performed across the integration.
    unsigned int getNewtonIterations() const { return newtonIterations; }

    // Accepted / rejected step counters.
    unsigned int getAcceptedSteps() const { return acceptedSteps; }
    unsigned int getRejectedSteps() const { return rejectedSteps; }
    unsigned int getRejectedNewtonSteps() const { return rejectedNewtonSteps; }

    // Number of Jacobian (re)evaluations.  Each is a full J assembly.
    unsigned int getJacobianEvaluations() const { return updateJacobian; }

    // Number of Newton-matrix (P) updates performed from a *reused* Jacobian
    // (i.e. P re-formed and re-factorized without reassembling J).
    unsigned int getMatrixPUpdates() const { return updateP; }

    // Total number of Newton-matrix (LU) factorizations.  Every Jacobian
    // assembly re-forms and factorizes P, and every P-only update factorizes
    // again, so the total is updateJacobian + updateP.
    unsigned int getFactorizations() const { return updateJacobian + updateP; }

  private:
    bool setSimultaneousDerivatives(Scalar **y_seed_derivatives_, Scalar **p_seed_derivatives_,
                                    Scalar *relative_error_derivatives_,
                                    Scalar **absolut_error_derivatives_,
                                    unsigned int numOfDerivatives_);
    bool freeSimultaneousDerivatives();
    CppAD::ADFun<Scalar> *RHS;

    // Pluggable linear solver (default DenseLUSolver). Mathematically identical
    // to original; allows future sparse solvers without changing Newton logic.
    LinearSolver<Scalar> *linearSolver;

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

    Scalar *z_predictor;
    Scalar *z_corrector;
    Scalar *z_corrector_old;

    // Simultaneous Derivatives
    Scalar **z_predictor_derivatives;
    Scalar **z_corrector_derivatives;
    Scalar **z_corrector_derivatives_old;
    Scalar **y_seed_derivatives;
    Scalar **p_seed_derivatives;

    unsigned int numOfDerivatives;

    bool isIntegrattionStarted;
    bool isIntegrattionPerformed;

    unsigned int order_corrector;
    unsigned int num_of_steps_per_order;
    unsigned int numOfSpecies;
    Scalar step_size_h;
    Scalar step_size_h_old;
    Scalar t_i[2];
    Scalar l_i[7];

    Scalar *corrector;
    // Simultaneous Derivatives
    Scalar **corrector_derivatives;

    Scalar *corrector_old;
    // Simultaneous Derivatives
    Scalar **corrector_old_derivatives;

    Scalar *error_weights;
    Scalar *absolut_error;
    Scalar relative_error;
    // Simultaneous Derivatives
    Scalar **error_weights_derivatives;
    Scalar **absolut_error_derivatives;
    Scalar *relative_error_derivatives;

    Scalar t_ende;
    Scalar *solution_Vector;
    // Simultaneous Derivatives
    Scalar **solution_Vector_derivatives;

    Scalar minimalEta;

    // Simultaneous Derivatives

    Scalar c_n_old;
    Scalar max_StepLength;
    Scalar start_integration_time;
    Scalar *P;
    Scalar *J;
    int *ip;
    long *scaling_D1;
    long *scaling_D2;
    bool setJacobian;
    bool setNewP;
    bool newtonIteration_Started;
    Scalar convergenceIndicator;
    unsigned newtonFAIL;
    unsigned stepFAIL;
    unsigned int RHS_type; // 0 - Autonom; 1 time dependent; 2 time and parameter dependent;
    Scalar *parameter;
    unsigned int sizeOfParameter;

    Scalar stepSizeSet_new[6];
    Scalar stepSizeSet_old[6];

    Scalar reachableError;
    Scalar *reachableError_derivatives;

    // SENSITIVITY GENERATION

    // forward

    struct bdf_TAPE {
        Scalar timePoint;
        Scalar stepSize;
        CppAD::ADFun<Scalar> **CppAD_tape;
        CppAD::ADFun<Scalar> *direct_fun;
        Scalar *endSolution;
        Scalar *P;
        int *ip;
        long *scaling_D1;
        long *scaling_D2;
        // NAVIGATION
        bdf_TAPE *forwardLINK;
        bdf_TAPE *backLINK;
        bool calculate_output;
        bool orderUP;
        bool orderDOWN;
        unsigned int NewtonIterates;

        Scalar *old_parameter_seed;
        Scalar *old_y_seed;
        bool updatedTaylorFunction;
        unsigned int lastTaylorOrder;
    };

    bdf_TAPE *startTAPE;
    bdf_TAPE *intermediateTAPE;
    bdf_TAPE *endTAPE;

    cppADFunListType cppADFunList;
    typename cppADFunListType::iterator intermediateCppADFunIterator;

    bool direct;

    CppAD::ADFun<Scalar> *start_CppAD_tape;
    Scalar *correctorVector_tape;
    Scalar startTape_stepSize;

    Scalar l_i_tape[7];
    Scalar time_tape;
    Scalar stepSizeSet_tape[6];
    Scalar *z_predictor_tape;
    Scalar *z_corrector_tape;
    Scalar step_size_h_tape;
    Scalar *forward_solution_Vector;
    unsigned int order_corrector_tape;
    bool taping;
    bool getTAPE;
    unsigned int lastForwardOrder;
    Scalar *P_tape;
    int *ip_tape;
    long *scaling_D1_tape;
    long *scaling_D2_tape;
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

    bdf_TAPE *adjointTapePointer;

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

    // Integrator Configuration

    Scalar START_STEP_SIZE_SAFETY;
    Scalar NEWTON_STEP_REDUCTION;
    Scalar STEP_NOT_ACCEPTED_REDUCTION;
    Scalar STEP_NOT_ACCEPTED_SAVETY_STEP;
    Scalar ORDER_STAY_SAVETY_STEP;
    Scalar ORDER_UP_SAVETY_STEP;
    Scalar ORDER_DOWN_SAVETY_STEP;
    unsigned int allowedSTEPS;

    Scalar NEWTON_SAFETY; // negative Value rate is estimated automatically
    Scalar NEWTON_CONVERGED;
    Scalar ACCEPT_NEWTON_CONVERGENCE_RATE;
    unsigned int allowedNEWTON;
    unsigned int MAXNEWTONITERATION;

    unsigned int MAXORDER;
    unsigned int MINORDER;
    Scalar MACHINEPRECISION;
    bool HITSOLUTION;
    unsigned int ERROR_WEIGHTS_INDEX;
    unsigned int MINTAYLOR;

    bool updateEveryP;

    Scalar RESTRICT_ORDER_1_RATIO;
    Scalar RESTRICT_ORDER_2_RATIO;
    Scalar RESTRICT_ORDER_3_RATIO;
    Scalar RESTRICT_ORDER_4_RATIO;
    Scalar RESTRICT_ORDER_5_RATIO;
    Scalar RESTRICT_ORDER_6_RATIO;

    // Statistic

    unsigned int acceptedSteps;
    unsigned int rejectedSteps;
    unsigned int rejectedNewtonSteps;
    unsigned int updateJacobian;
    unsigned int updateP;
    unsigned int newtonIterations;
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
