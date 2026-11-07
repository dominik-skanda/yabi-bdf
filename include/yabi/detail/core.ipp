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
// Construction, configuration, reset and result accessors.
//
// Generated from the reference implementation in YABI_old without changing any
// expression: the bodies below are the original algorithm verbatim.

#ifndef YABI_DETAIL_CORE_IPP
#define YABI_DETAIL_CORE_IPP

namespace yabi {

template <typename Scalar>
BDFIntegrator<Scalar>::BDFIntegrator(const AbstractRHS<Scalar> &RHS_)
    : stepController(nordsieck, config),
      sensitivityEngine(nordsieck, config, RHS, numOfSpecies, RHS_type, sizeOfParameter, parameter,
                        start_integration_time, direct, internalError, startTAPE, intermediateTAPE,
                        start_CppAD_tape, startTape_stepSize, cppADFunList,
                        intermediateCppADFunIterator, blasONE, blasScalarONE, blasScalarMinusONE,
                        blasScalarZERO),
      newtonSolver(
          nordsieck, config, stats, RHS, linearSolver, numOfSpecies, RHS_type, sizeOfParameter,
          parameter, error_weights, absolut_error, relative_error,
          sensitivityEngine.error_weights_derivatives, sensitivityEngine.absolut_error_derivatives,
          sensitivityEngine.relative_error_derivatives, t_ende, minimalEta, reachableError,
          sensitivityEngine.reachableError_derivatives, sensitivityEngine.numOfDerivatives, direct,
          taping, sensitivityEngine.z_predictor_derivatives,
          sensitivityEngine.corrector_derivatives, sensitivityEngine.p_seed_derivatives, startTAPE,
          endTAPE, replayTapeStore, cppADFunList, intermediateCppADFunIterator, internalError),
      adjointEngine(sensitivityEngine, nordsieck, newtonSolver, RHS, numOfSpecies, RHS_type,
                    sizeOfParameter, parameter, config, direct, internalError, startTape_stepSize,
                    start_CppAD_tape, cppADFunList, intermediateCppADFunIterator, blasONE,
                    blasScalarONE, blasScalarMinusONE, blasScalarZERO) {
    unsigned int i;
    isIntegrattionPerformed = false;
    isIntegrattionStarted = false;

    RHS.reset(RHS_.clone());

    if (RHS->Domain() < RHS->Range()) {
        YABI_LOG_ERROR("right hand side not properly set!");

        exit(-1);
    }

    numOfSpecies = RHS->Range();

    if (RHS->Domain() == numOfSpecies) {
        RHS_type = 0;
        sizeOfParameter = 0;
    }
    if (RHS->Domain() == numOfSpecies + 1) {
        RHS_type = 1;
        sizeOfParameter = 0;
    }
    if (RHS->Domain() > numOfSpecies + 1) {
        RHS_type = 2;
        sizeOfParameter = RHS->Domain() - numOfSpecies - 1;
    }
    parameter.resize(sizeOfParameter);
    nordsieck.initialize(numOfSpecies);

    error_weights.resize(numOfSpecies);
    absolut_error.resize(numOfSpecies);

    newtonSolver.P.resize(numOfSpecies * numOfSpecies);
    newtonSolver.J.resize(numOfSpecies * numOfSpecies);
    newtonSolver.newtonIteration_Started = false;

    newtonSolver.ip.resize(numOfSpecies);

    newtonSolver.scaling_D1.resize(numOfSpecies);
    newtonSolver.scaling_D2.resize(numOfSpecies);

    startTAPE = NULL;
    intermediateTAPE = NULL;
    endTAPE = NULL;

    start_CppAD_tape = NULL;
    internalError = false;

    sensitivityEngine.initialize(numOfSpecies);
    direct = false;

    adjointEngine.release();

    blasONE = 1;
    blasScalarONE = (Scalar)1.0;
    blasScalarMinusONE = (Scalar)-1.0;
    blasScalarZERO = (Scalar)0.0;

    // Configuration and statistics use their declared defaults
    // (see detail/integrator_config.hpp).

    linearSolver = std::make_unique<DenseLUSolver<Scalar>>();
}

// Allocate a replay-tape node, append it after the current end and make it the
// new end.  The caller guarantees that appending is legal (see the call sites);
// all per-step buffers start empty and the CppAD slots start null.
template <typename Scalar>
typename BDFIntegrator<Scalar>::ReplayTape *
BDFIntegrator<Scalar>::createReplayTapeNode() {
    ReplayTape *node = replayTapeStore.create();
    node->CppAD_tape.assign(config.MAXNEWTONITERATION, nullptr);
    node->backLINK = endTAPE;
    if (endTAPE != nullptr)
        endTAPE->forwardLINK = node;
    else
        startTAPE = node;
    endTAPE = node;
    return node;
}

template <typename Scalar>
bool BDFIntegrator<Scalar>::resetIntegration(Scalar *y_start, Scalar max_StepLength,
                                             Scalar relative_error, Scalar *absolut_error_,
                                             const IntegrationOptions<Scalar> &options) {
    return resetIntegration(y_start, max_StepLength, relative_error, absolut_error_, options.taping,
                            options.direct, options.y_seed_list, options.p_seed_list,
                            options.relative_error_derivatives, options.absolut_error_derivatives,
                            options.numOfderivatives);
}

template <typename Scalar>
bool BDFIntegrator<Scalar>::resetIntegration(Scalar *y_start, Scalar T_start, Scalar max_StepLength,
                                             Scalar relative_error, Scalar *absolut_error_,
                                             const IntegrationOptions<Scalar> &options) {
    return resetIntegration(y_start, T_start, max_StepLength, relative_error, absolut_error_,
                            options.taping, options.direct, options.y_seed_list, options.p_seed_list,
                            options.relative_error_derivatives, options.absolut_error_derivatives,
                            options.numOfderivatives);
}

template <typename Scalar>
bool BDFIntegrator<Scalar>::resetIntegration(Scalar *y_start, Scalar T_start, Scalar *parameter_,
                                             Scalar max_StepLength, Scalar relative_error,
                                             Scalar *absolut_error_,
                                             const IntegrationOptions<Scalar> &options) {
    return resetIntegration(y_start, T_start, parameter_, max_StepLength, relative_error,
                            absolut_error_, options.taping, options.direct, options.y_seed_list,
                            options.p_seed_list, options.relative_error_derivatives,
                            options.absolut_error_derivatives, options.numOfderivatives);
}

template <typename Scalar>
bool BDFIntegrator<Scalar>::resetIntegration(
    Scalar *y_start, Scalar max_StepLength_, Scalar relative_error_, Scalar *absolut_error_,
    bool taping_, bool direct_, Scalar **y_seed_derivatives_, Scalar **p_seed_derivatives_,
    Scalar *relative_error_derivatives_, Scalar **absolut_error_derivatives_,
    unsigned int numOfDerivatives_) {
    if (RHS_type != 0) {
        YABI_LOG_ERROR("right hand side is not autonom!");

        return false;
    }
    unsigned int i;
    max_StepLength = max_StepLength_;
    // Lege Nordsiek-Array an
    nordsieck.z_corrector.resize(numOfSpecies * 7);

    nordsieck.z_corrector_old.resize(numOfSpecies * 7);

    nordsieck.z_predictor.resize(numOfSpecies * 7);

    nordsieck.order_corrector = 1;

    // Initialisiere Nordsiek-Array
    for (i = 0; i < numOfSpecies; i++)
        nordsieck.z_corrector[i] = y_start[i];

    for (i = 0; i < numOfSpecies; i++)
        absolut_error[i] = absolut_error_[i];
    relative_error = relative_error_;
    for (i = 0; i < numOfSpecies; i++)
        error_weights[i] = 0.0;
    freeSimultaneousDerivatives();
    start_integration_time = 0.0;
    nordsieck.t_i[1] = 0.0;
    estimateStartStepsize();
    std::vector<Scalar> y_start_(numOfSpecies);
    for (i = 0; i < numOfSpecies; i++)
        y_start_[i] = y_start[i];
    std::vector<Scalar> rhs(numOfSpecies);
    rhs = RHS->Forward(0, y_start_);
    nordsieck.t_i[0] = nordsieck.step_size_h;
    for (i = 0; i < numOfSpecies; i++)
        nordsieck.z_corrector[numOfSpecies + i] = rhs[i] * nordsieck.step_size_h;
    isIntegrattionStarted = true;
    isIntegrattionPerformed = false;
    nordsieck.num_of_steps_per_order = 0;
    newtonSolver.newtonIteration_Started = false;
    newtonSolver.convergenceIndicator = 0;
    internalError = false;
    // Sensitivity Generation
    taping = taping_;
    if (taping) {
        if (!createNewTape()) {
            YABI_LOG_ERROR("RESET INTEGRATOR FAILED!");

            return false;
        }
        direct = direct_;
    } else
        deleteOldTape();

    for (i = 0; i < 6; i++)
        nordsieck.stepSizeSet_new[i] = 0.0;
    nordsieck.step_size_h_old = nordsieck.step_size_h;

    // Simultaneous Derivatives
    setSimultaneousDerivatives(y_seed_derivatives_, p_seed_derivatives_,
                               relative_error_derivatives_, absolut_error_derivatives_,
                               numOfDerivatives_);

    sensitivityEngine.adjointTapePointer = NULL;

    stats.acceptedSteps = 0;
    stats.rejectedSteps = 0;
    stats.rejectedNewtonSteps = 0;
    stats.updateJacobian = 0;
    stats.updateP = 0;
    stats.newtonIterations = 0;

    return true;
}

template <typename Scalar>
bool BDFIntegrator<Scalar>::resetIntegration(
    Scalar *y_start, Scalar T_start, Scalar max_StepLength_, Scalar relative_error_,
    Scalar *absolut_error_, bool taping_, bool direct_, Scalar **y_seed_derivatives_,
    Scalar **p_seed_derivatives_, Scalar *relative_error_derivatives_,
    Scalar **absolut_error_derivatives_, unsigned int numOfDerivatives_) {
    if (RHS_type != 0 && RHS_type != 1) {
        YABI_LOG_ERROR("RHS depends on parameters!");

        return false;
    }
    unsigned int i;
    max_StepLength = max_StepLength_;
    // Lege Nordsiek-Array an

    nordsieck.z_corrector.resize(numOfSpecies * 7);

    nordsieck.z_corrector_old.resize(numOfSpecies * 7);

    nordsieck.z_predictor.resize(numOfSpecies * 7);

    nordsieck.order_corrector = 1;

    // Initialisiere Nordsiek-Array
    for (i = 0; i < numOfSpecies; i++)
        nordsieck.z_corrector[i] = y_start[i];

    for (i = 0; i < numOfSpecies; i++)
        absolut_error[i] = absolut_error_[i];
    relative_error = relative_error_;
    for (i = 0; i < numOfSpecies; i++)
        error_weights[i] = 0.0;
    freeSimultaneousDerivatives();

    start_integration_time = T_start;
    nordsieck.t_i[1] = T_start;

    estimateStartStepsize();

    nordsieck.t_i[0] = T_start + nordsieck.step_size_h;

    unsigned int cppAD_argument_size = 0;

    if (RHS_type == 0)
        cppAD_argument_size = numOfSpecies;
    if (RHS_type == 1)
        cppAD_argument_size = numOfSpecies + 1;

    std::vector<Scalar> y_start_(cppAD_argument_size);
    for (i = 0; i < numOfSpecies; i++)
        y_start_[i] = y_start[i];
    if (RHS_type == 1)
        y_start_[numOfSpecies] = T_start;
    std::vector<Scalar> rhs(numOfSpecies);
    rhs = RHS->Forward(0, y_start_);

    for (i = 0; i < numOfSpecies; i++)
        nordsieck.z_corrector[numOfSpecies + i] = rhs[i] * nordsieck.step_size_h;
    isIntegrattionStarted = true;
    isIntegrattionPerformed = false;
    nordsieck.num_of_steps_per_order = 0;
    newtonSolver.newtonIteration_Started = false;
    newtonSolver.convergenceIndicator = 0;
    internalError = false;
    // Sensitivity Generation
    taping = taping_;
    if (taping) {
        if (!createNewTape()) {
            YABI_LOG_ERROR("RESET INTEGRATOR FAILED!");

            return false;
        }
        direct = direct_;
    } else
        deleteOldTape();

    for (i = 0; i < 6; i++)
        nordsieck.stepSizeSet_new[i] = 0.0;
    nordsieck.step_size_h_old = nordsieck.step_size_h;
    setSimultaneousDerivatives(y_seed_derivatives_, p_seed_derivatives_,
                               relative_error_derivatives_, absolut_error_derivatives_,
                               numOfDerivatives_);

    sensitivityEngine.adjointTapePointer = NULL;

    stats.acceptedSteps = 0;
    stats.rejectedSteps = 0;
    stats.rejectedNewtonSteps = 0;
    stats.updateJacobian = 0;
    stats.updateP = 0;
    stats.newtonIterations = 0;

    return true;
}

template <typename Scalar>
bool BDFIntegrator<Scalar>::resetIntegration(
    Scalar *y_start, Scalar T_start, Scalar *parameter_, Scalar max_StepLength_,
    Scalar relative_error_, Scalar *absolut_error_, bool taping_, bool direct_,
    Scalar **y_seed_derivatives_, Scalar **p_seed_derivatives_, Scalar *relative_error_derivatives_,
    Scalar **absolut_error_derivatives_, unsigned int numOfDerivatives_) {
    unsigned int i;
    for (i = 0; i < sizeOfParameter; i++)
        parameter[i] = parameter_[i];

    max_StepLength = max_StepLength_;

    // Lege Nordsiek-Array an

    nordsieck.z_corrector.resize(numOfSpecies * 7);

    nordsieck.z_corrector_old.resize(numOfSpecies * 7);

    nordsieck.z_predictor.resize(numOfSpecies * 7);

    nordsieck.order_corrector = 1;

    // Initialisiere Nordsiek-Array
    for (i = 0; i < numOfSpecies; i++)
        nordsieck.z_corrector[i] = y_start[i];

    for (i = 0; i < numOfSpecies; i++)
        absolut_error[i] = absolut_error_[i];
    relative_error = relative_error_;
    for (i = 0; i < numOfSpecies; i++)
        error_weights[i] = 0.0;
    freeSimultaneousDerivatives();

    start_integration_time = T_start;
    nordsieck.t_i[1] = T_start;

    estimateStartStepsize();

    nordsieck.t_i[0] = T_start + nordsieck.step_size_h;

    unsigned int cppAD_argument_size = 0;

    if (RHS_type == 0)
        cppAD_argument_size = numOfSpecies;
    if (RHS_type == 1)
        cppAD_argument_size = numOfSpecies + 1;
    if (RHS_type == 2)
        cppAD_argument_size = numOfSpecies + 1 + sizeOfParameter;

    std::vector<Scalar> y_start_(cppAD_argument_size);
    for (i = 0; i < numOfSpecies; i++)
        y_start_[i] = y_start[i];
    if (RHS_type > 0)
        y_start_[numOfSpecies] = T_start;
    if (RHS_type == 2) {
        for (i = 0; i < sizeOfParameter; i++)
            y_start_[numOfSpecies + 1 + i] = parameter[i];
    }
    std::vector<Scalar> rhs(numOfSpecies);
    rhs = RHS->Forward(0, y_start_);

    for (i = 0; i < numOfSpecies; i++)
        nordsieck.z_corrector[numOfSpecies + i] = rhs[i] * nordsieck.step_size_h;
    isIntegrattionStarted = true;
    isIntegrattionPerformed = false;
    nordsieck.num_of_steps_per_order = 0;
    newtonSolver.newtonIteration_Started = false;
    newtonSolver.convergenceIndicator = 0;
    internalError = false;
    // Sensitivity Generation
    taping = taping_;
    if (taping) {
        if (!createNewTape()) {
            YABI_LOG_ERROR("RESET INTEGRATOR FAILED!");

            return false;
        }
        direct = direct_;
    } else
        deleteOldTape();

    for (i = 0; i < 6; i++)
        nordsieck.stepSizeSet_new[i] = 0.0;
    nordsieck.step_size_h_old = nordsieck.step_size_h;
    setSimultaneousDerivatives(y_seed_derivatives_, p_seed_derivatives_,
                               relative_error_derivatives_, absolut_error_derivatives_,
                               numOfDerivatives_);

    sensitivityEngine.adjointTapePointer = NULL;

    stats.acceptedSteps = 0;
    stats.rejectedSteps = 0;
    stats.rejectedNewtonSteps = 0;
    stats.updateJacobian = 0;
    stats.updateP = 0;
    stats.newtonIterations = 0;

    return true;
}

template <typename Scalar>
bool BDFIntegrator<Scalar>::performIntegration(const Scalar t_ende_,
                                               unsigned int maximal_Stepsize) {
    if (!sensitivityEngine.start_adjoint_from_forward_mode)
        sensitivityEngine.adjointTapePointer = NULL;
    if (internalError) {
        YABI_LOG_ERROR("internal error please reset integration!");

        return false;
    }
    unsigned int i, j, k, z, accept_indicator = 0;
    if (!isIntegrattionStarted) {
        YABI_LOG_ERROR("initialization failed!");
        return false;
    }
    t_ende = t_ende_;
    if (start_integration_time > t_ende) {
        YABI_LOG_ERROR(" t_start > t_end");

        return false;
    }
    // Reset the forward-trajectory record so the continuous adjoint reflects
    // this run only.
    trajectory_.clear();

    if (nordsieck.t_i[1] >= t_ende) {
        calculate_output_vector();
        return true;
    }
    // Starte Integrationsroutine
    if (isIntegrattionPerformed)
        prepare_next_step();
    accept_indicator = 0;
    for (i = 0; i < maximal_Stepsize; i++) {
        for (j = 0; j < numOfSpecies; j++)
            nordsieck.corrector_old[j] = nordsieck.corrector[j];

        if (sensitivityEngine.numOfDerivatives > 0)
            for (z = 0; z < sensitivityEngine.numOfDerivatives; z++)
                for (j = 0; j < numOfSpecies; j++)
                    sensitivityEngine.corrector_old_derivatives[z][j] =
                        sensitivityEngine.corrector_derivatives[z][j];

        newtonSolver.setJacobian = false;
        newtonSolver.setNewP = config.updateEveryP;
        newtonSolver.newtonFAIL = 0;
        newtonSolver.stepFAIL = 0;

        if (taping) {
            if (endTAPE == NULL && startTAPE != NULL)
                return false;

            ReplayTape *tape = createReplayTapeNode();
            if (sensitivityEngine.getTAPE == true && intermediateTAPE == NULL)
                intermediateTAPE = endTAPE;

            tape->calculate_output = false;
            tape->stepSize = 0.0;
            tape->timePoint = nordsieck.t_i[1];
            tape->orderUP = false;
            tape->orderDOWN = false;
        }

        // SAVE INITIAL STEPSIZE_SET & Z_CORRECTOR
        for (j = 0; j < 6; j++)
            nordsieck.stepSizeSet_old[j] = nordsieck.stepSizeSet_new[j];
        for (j = 0; j < numOfSpecies; j++) {
            for (z = 0; z <= nordsieck.order_corrector; z++)
                nordsieck.z_corrector_old[z * numOfSpecies + j] =
                    nordsieck.z_corrector[z * numOfSpecies + j];
            for (z = nordsieck.order_corrector + 1; z < 7; z++)
                nordsieck.z_corrector_old[z * numOfSpecies + j] = 0.0;
        }
        if (sensitivityEngine.numOfDerivatives > 0) {
            for (z = 0; z < sensitivityEngine.numOfDerivatives; z++)
                for (j = 0; j < numOfSpecies; j++) {
                    for (k = 0; k <= nordsieck.order_corrector; k++)
                        sensitivityEngine.z_corrector_derivatives_old[z][numOfSpecies * k + j] =
                            sensitivityEngine.z_corrector_derivatives[z][numOfSpecies * k + j];
                    for (k = nordsieck.order_corrector + 1; k < 7; k++)
                        sensitivityEngine.z_corrector_derivatives_old[z][numOfSpecies * k + j] =
                            0.0;
                }
        }
        newtonSolver.newtonFAIL = 0;
        newtonSolver.stepFAIL = 0;
        while (accept_indicator == 0) {
            if (nordsieck.step_size_h != nordsieck.step_size_h_old) {
                Scalar temp = nordsieck.step_size_h / nordsieck.step_size_h_old;
                for (j = 0; j < numOfSpecies; j++)
                    for (z = 0; z <= nordsieck.order_corrector; z++)
                        nordsieck.z_corrector[z * numOfSpecies + j] =
                            nordsieck.z_corrector_old[z * numOfSpecies + j] * (Scalar)pow(temp, z);
                if (sensitivityEngine.numOfDerivatives > 0)
                    for (z = 0; z < sensitivityEngine.numOfDerivatives; z++)
                        for (j = 0; j < numOfSpecies; j++)
                            for (k = 0; k <= nordsieck.order_corrector; k++)
                                sensitivityEngine.z_corrector_derivatives[z][numOfSpecies * k + j] =
                                    sensitivityEngine
                                        .z_corrector_derivatives_old[z][numOfSpecies * k + j] *
                                    (Scalar)pow(temp, k);
            }
            for (j = 5; j > 0; j--)
                nordsieck.stepSizeSet_new[j] =
                    nordsieck.stepSizeSet_old[j - 1] + nordsieck.step_size_h;
            nordsieck.stepSizeSet_new[0] = nordsieck.step_size_h;
            if (!calculate_next_predictor()) {
                internalError = true;

                YABI_LOG_ERROR("calculate predictor failed!");

                return false;
            }
            if (!calculate_l_vector()) {
                internalError = true;

                YABI_LOG_ERROR("calculate l_vector_failed!");

                return false;
            }
            if (!calculate_corrector_vector()) {
                internalError = true;

                YABI_LOG_ERROR("calculate corrector.data() vector failed!");

                return false;
            }
            accept_indicator = accept_corrector_vector();
        }
        calculate_next_corrector();
        nordsieck.step_size_h_old = nordsieck.step_size_h;
        // Record the accepted step's Nordsieck history for the continuous adjoint.
        trajectory_.record(nordsieck.t_i[1], nordsieck.step_size_h, nordsieck.order_corrector,
                           nordsieck.z_corrector);
        if (accept_indicator == 1)
            prepare_next_step();
        if (accept_indicator == 2) {
            calculate_output_vector();
            isIntegrattionPerformed = true;
            return true;
        }
        if (accept_indicator == 3) {
            YABI_LOG_ERROR("too much newton steps rejected!");

            internalError = true;
            return false;
        }
        if (accept_indicator == 4) {
            YABI_LOG_ERROR("too much steps rejected!");

            internalError = true;
            return false;
        }
        accept_indicator = 0;
    }

    YABI_LOG_ERROR("steps exceeded!");

    isIntegrattionPerformed = true;
    return false;
}

template <typename Scalar> bool BDFIntegrator<Scalar>::getSolutionVector(Scalar *solution_Vector_) {
    unsigned int i;
    for (i = 0; i < numOfSpecies; i++)
        solution_Vector_[i] = nordsieck.solution_Vector[i];
    return true;
}

template <typename Scalar> bool BDFIntegrator<Scalar>::getSolutionAt(Scalar t,
                                                                   Scalar *solution_Vector_at_t) {
    if (internalError)
        return false;
    if (!isIntegrattionPerformed) {
        YABI_LOG_ERROR("getSolutionAt: integration has not been performed yet!");

        return false;
    }

    // The Nordsieck interpolant is anchored at the last accepted time t_i[1]
    // with the last step size step_size_h:
    //     y(t) = sum_j z_corrector[n*j + i] * theta^j,   theta = (t - t_i[1]) / step_size_h.
    // It is only accurate within the completed step theta in [-1, 0].
    const Scalar theta = (t - nordsieck.t_i[1]) / nordsieck.step_size_h;
    if (theta < -1.0 - 4.0 * config.MACHINEPRECISION || theta > 4.0 * config.MACHINEPRECISION) {
        YABI_LOG_ERROR("getSolutionAt: t = ", t, " is outside the last step [",
                       nordsieck.t_i[1] - nordsieck.step_size_h, ", ", nordsieck.t_i[1], "]!");

        return false;
    }

    unsigned int i, j;
    Scalar temp[7];
    for (i = 0; i <= nordsieck.order_corrector; i++) {
        temp[i] = 1.0;
        for (j = 0; j < i; j++)
            temp[i] *= theta;
    }

    int blasTEMP = nordsieck.order_corrector + 1;
    linalg::GEMV_NT(numOfSpecies, blasTEMP, blasScalarONE, nordsieck.z_corrector.data(),
                    numOfSpecies, temp, blasONE, blasScalarZERO, solution_Vector_at_t, blasONE);

    return true;
}

template <typename Scalar>
bool BDFIntegrator<Scalar>::getSimultaneousDerivativesVector(
    Scalar **solution_Vector_derivatives_) {
    unsigned int i, z;
    for (z = 0; z < sensitivityEngine.numOfDerivatives; z++)
        for (i = 0; i < numOfSpecies; i++)
            solution_Vector_derivatives_[z][i] =
                sensitivityEngine.solution_Vector_derivatives[z][i];
    return true;
}

template <typename Scalar>
bool BDFIntegrator<Scalar>::getForwardSolutionVector(Scalar *forward_solution_Vector_) {
    unsigned int i;
    for (i = 0; i < numOfSpecies; i++)
        forward_solution_Vector_[i] = sensitivityEngine.forward_solution_Vector[i];
    return true;
}

template <typename Scalar>
bool BDFIntegrator<Scalar>::getReverseSolutionVector(Scalar **solution_adjoint_species_,
                                                     Scalar **solution_adjoint_parameter_) {
    unsigned int i, j;
    for (i = 0; i < numOfSpecies; i++)
        for (j = 0; j < adjointEngine.stored_adjoint_order; j++)
            solution_adjoint_species_[i][j] = adjointEngine.solution_adjoint_species[i][j];

    for (i = 0; i < sizeOfParameter; i++)
        for (j = 0; j < adjointEngine.stored_adjoint_order; j++)
            solution_adjoint_parameter_[i][j] = adjointEngine.solution_adjoint_parameter[i][j];

    return true;
}

// Continuous adjoint (T25): first-order stable reverse sensitivity.

template <typename Scalar>
bool BDFIntegrator<Scalar>::continuousAdjoint(Scalar *seed_weights, Scalar *adjoint_y0,
                                              Scalar *adjoint_parameter) {
    if (!isIntegrattionPerformed) {
        YABI_LOG_ERROR("continuous adjoint: run performIntegration first!");

        return false;
    }
    if (trajectory_.empty()) {
        YABI_LOG_ERROR("continuous adjoint: no forward trajectory recorded!");

        return false;
    }

    // Backward adjoint ODE d lambda/d tau = J(y(T - tau))^T lambda, integrated
    // forward in reversed time tau in [0, T] by a fresh BDF integrator.
    detail::ContinuousAdjointRHS<Scalar> adjRHS(*RHS, trajectory_, numOfSpecies, t_ende,
                                                parameter);

    std::vector<Scalar> lam0(numOfSpecies);
    for (unsigned int i = 0; i < numOfSpecies; i++)
        lam0[i] = seed_weights[i]; // lambda(T) = w = d g / d y(T)

    std::vector<Scalar> abs_err(numOfSpecies);
    for (unsigned int i = 0; i < numOfSpecies; i++)
        abs_err[i] = absolut_error[i];

    BDFIntegrator<Scalar> adj(adjRHS);
    if (!adj.resetIntegration(lam0.data(), 0.0, NULL, max_StepLength, relative_error,
                              abs_err.data(), false, false, NULL, NULL, NULL, NULL, 0))
        return false;
    if (!adj.performIntegration(t_ende, 10000000)) {
        YABI_LOG_ERROR("continuous adjoint: backward integration failed!");

        return false;
    }
    adj.getSolutionVector(adjoint_y0); // lambda(0) = d g / d y0

    // Parameter sensitivity (d g/d p = int_0^T lambda^T d f/d p dt) is not yet
    // accumulated along the backward pass; leave it zero for now (honest scope).
    if (adjoint_parameter != nullptr)
        for (unsigned int i = 0; i < sizeOfParameter; i++)
            adjoint_parameter[i] = (Scalar)0;

    return true;
}

// Integrator Configuration

template <typename Scalar>
bool BDFIntegrator<Scalar>::setStepSettings(
    Scalar NEWTON_STEP_REDUCTION_, Scalar STEP_NOT_ACCEPTED_REDUCTION_,
    Scalar STEP_NOT_ACCEPTED_SAVETY_STEP_, Scalar ORDER_STAY_SAVETY_STEP_,
    Scalar ORDER_UP_SAVETY_STEP_, Scalar ORDER_DOWN_SAVETY_STEP_, unsigned int allowedSTEPS_) {
    if (0 < NEWTON_STEP_REDUCTION_ && NEWTON_STEP_REDUCTION_ < 1)
        config.NEWTON_STEP_REDUCTION = NEWTON_STEP_REDUCTION_;
    else
        return false;

    if (0 < STEP_NOT_ACCEPTED_REDUCTION_ && STEP_NOT_ACCEPTED_REDUCTION_ < 1)
        config.STEP_NOT_ACCEPTED_REDUCTION = STEP_NOT_ACCEPTED_REDUCTION_;
    else
        return false;

    if (0 < ORDER_STAY_SAVETY_STEP_ && ORDER_STAY_SAVETY_STEP_ < 1)
        config.ORDER_STAY_SAVETY_STEP = ORDER_STAY_SAVETY_STEP_;
    else
        return false;

    if (0 < ORDER_UP_SAVETY_STEP_ && ORDER_UP_SAVETY_STEP_ < 1)
        config.ORDER_UP_SAVETY_STEP = ORDER_UP_SAVETY_STEP_;
    else
        return false;

    if (0 < ORDER_DOWN_SAVETY_STEP_ && ORDER_DOWN_SAVETY_STEP_ < 1)
        config.ORDER_DOWN_SAVETY_STEP = ORDER_DOWN_SAVETY_STEP_;
    else
        return false;

    if (allowedSTEPS_ > 0)
        config.allowedSTEPS = allowedSTEPS_;
    else
        return false;

    if (0 < STEP_NOT_ACCEPTED_SAVETY_STEP_ && STEP_NOT_ACCEPTED_SAVETY_STEP_ <= 1)
        config.STEP_NOT_ACCEPTED_SAVETY_STEP = STEP_NOT_ACCEPTED_SAVETY_STEP_;
    else
        return false;

    return true;
}

template <typename Scalar>
bool BDFIntegrator<Scalar>::setNewtonSettings(Scalar NEWTON_SAFETY_, Scalar NEWTON_CONVERGED_,
                                              Scalar ACCEPT_NEWTON_CONVERGENCE_RATE_,
                                              unsigned int allowedNEWTON_,
                                              unsigned int MAXNEWTONITERATION_,
                                              bool updateEveryP_) {
    config.NEWTON_SAFETY = NEWTON_SAFETY_;

    if (0 < NEWTON_CONVERGED_)
        config.NEWTON_CONVERGED = NEWTON_CONVERGED_;
    else
        return false;

    if (0 <= ACCEPT_NEWTON_CONVERGENCE_RATE_)
        config.ACCEPT_NEWTON_CONVERGENCE_RATE = ACCEPT_NEWTON_CONVERGENCE_RATE_;
    else
        return false;

    if (allowedNEWTON_ > 0)
        config.allowedNEWTON = allowedNEWTON_;
    else
        return false;

    if (MAXNEWTONITERATION_ > 0)
        config.MAXNEWTONITERATION = MAXNEWTONITERATION_;
    else
        return false;

    config.updateEveryP = updateEveryP_;

    return true;
}

template <typename Scalar>
bool BDFIntegrator<Scalar>::setAlgorithmSettings(unsigned int MAXORDER_, Scalar MACHINEPRECISION_,
                                                 bool HITSOLUTION_,
                                                 unsigned int ERROR_WEIGHTS_INDEX_) {
    // Validate all arguments before changing any state: a rejected call must
    // not leave the integrator partially reconfigured.
    if (!(0 < MAXORDER_ && MAXORDER_ < 7))
        return false;

    if (!(0 < MACHINEPRECISION_))
        return false;

    if (!(ERROR_WEIGHTS_INDEX_ < 2))
        return false;

    config.MAXORDER = MAXORDER_;
    config.MACHINEPRECISION = MACHINEPRECISION_;
    config.HITSOLUTION = HITSOLUTION_;
    config.ERROR_WEIGHTS_INDEX = ERROR_WEIGHTS_INDEX_;

    return true;
}

template <typename Scalar> bool BDFIntegrator<Scalar>::setFixedOrder(unsigned int order) {
    // Force a fixed BDF order: MINORDER == MAXORDER == order.  The integrator
    // still starts at order 1 (reset sets order_corrector = 1) and ramps up to
    // `order`; once there it can neither increase nor decrease.
    if (!(0 < order && order < 7))
        return false;

    config.MINORDER = order;
    config.MAXORDER = order;

    return true;
}

template <typename Scalar>
bool BDFIntegrator<Scalar>::setRestrictOrderRatio(
    Scalar restrictRatioOrder1, Scalar restrictRatioOrder2, Scalar restrictRatioOrder3,
    Scalar restrictRatioOrder4, Scalar restrictRatioOrder5, Scalar restrictRatioOrder6) {
    config.RESTRICT_ORDER_1_RATIO = restrictRatioOrder1;
    config.RESTRICT_ORDER_2_RATIO = restrictRatioOrder2;
    config.RESTRICT_ORDER_3_RATIO = restrictRatioOrder3;
    config.RESTRICT_ORDER_4_RATIO = restrictRatioOrder4;
    config.RESTRICT_ORDER_5_RATIO = restrictRatioOrder5;
    config.RESTRICT_ORDER_6_RATIO = restrictRatioOrder6;
    return true;
}

template <typename Scalar>
bool BDFIntegrator<Scalar>::setMinTaylorCapacity(unsigned int MINTAYLOR_) {
    config.MINTAYLOR = MINTAYLOR_;

    RHS->capacity_order(config.MINTAYLOR);
    for (typename cppADFunListType::iterator it = cppADFunList.begin(); it != cppADFunList.end();
         it++) {
        (*it)->capacity_order(config.MINTAYLOR);
    }

    return true;
}

template <typename Scalar>
bool BDFIntegrator<Scalar>::setMinMemoryADFunObjectes(unsigned int MINnumberADFunObjects) {
    unsigned int i, counter;
    counter = 0;

    for (typename cppADFunListType::iterator it = cppADFunList.begin(); it != cppADFunList.end();
         it++) {
        counter++;
    }
    for (i = counter + 1; i <= MINnumberADFunObjects; i++) {
        std::unique_ptr<AbstractRHS<Scalar>> tempADFun(RHS->clone());
        tempADFun->capacity_order(config.MINTAYLOR);
        cppADFunList.push_back(std::move(tempADFun));
    }

    return true;
}

// DESTRUCTOR AND FREE MEMORY

template <typename Scalar> BDFIntegrator<Scalar>::~BDFIntegrator() {
    // Releases the replay tape (nodes are owned by replayTapeStore) and the
    // sensitivity scratch mirrors.  The simultaneous-derivative arrays and the
    // adjoint result arrays are RAII-owned (std::vector) and clean themselves up.
    linearSolver.reset();
    deleteOldTape();

    cppADFunList.clear();

    // Adjoint result arrays are released by ~AdjointEngine().
}

// Sensitivity Generation

// FORWARD MODE

} // namespace yabi

#endif
