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

template <typename Scalar> BDFIntegrator<Scalar>::BDFIntegrator(CppAD::ADFun<Scalar> &RHS_) {
    unsigned int i;
    isIntegrattionPerformed = false;
    isIntegrattionStarted = false;

    RHS = new CppAD::ADFun<Scalar>();
    *RHS = RHS_;

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
    parameter = new Scalar[sizeOfParameter];
    corrector = new Scalar[numOfSpecies];

    z_corrector = NULL;
    z_predictor = NULL;
    z_corrector_old = NULL;

    for (i = 0; i < numOfSpecies; i++)
        corrector[i] = 1.0;
    corrector_old = new Scalar[numOfSpecies];
    error_weights = new Scalar[numOfSpecies];
    absolut_error = new Scalar[numOfSpecies];
    solution_Vector = new Scalar[numOfSpecies];
    for (i = 0; i < numOfSpecies; i++)
        solution_Vector[i] = 0;

    c_n_old = 1;

    P = new Scalar[numOfSpecies * numOfSpecies];
    J = new Scalar[numOfSpecies * numOfSpecies];
    newtonIteration_Started = false;

    ip = new int[numOfSpecies];

    scaling_D1 = new long[numOfSpecies];
    scaling_D2 = new long[numOfSpecies];

    startTAPE = NULL;
    intermediateTAPE = NULL;
    endTAPE = NULL;

    start_CppAD_tape = NULL;
    z_predictor_tape = NULL;
    z_corrector_tape = NULL;
    correctorVector_tape = new Scalar[numOfSpecies];
    internalError = false;
    P_tape = NULL;
    ip_tape = NULL;
    scaling_D1_tape = NULL;
    scaling_D2_tape = NULL;

    forward_solution_Vector = new Scalar[numOfSpecies];
    for (i = 0; i < numOfSpecies; i++)
        forward_solution_Vector[i] = 0;
    direct = false;

    // Simultaneous Derivatives

    numOfDerivatives = 0;
    z_predictor_derivatives = NULL;
    z_corrector_derivatives = NULL;
    corrector_derivatives = NULL;
    corrector_old_derivatives = NULL;
    solution_Vector_derivatives = NULL;
    y_seed_derivatives = NULL;
    p_seed_derivatives = NULL;
    z_corrector_derivatives_old = NULL;

    error_weights_derivatives = NULL;
    absolut_error_derivatives = NULL;
    relative_error_derivatives = NULL;

    reachableError_derivatives = NULL;

    adjointTapePointer = NULL;

    solution_adjoint_species = NULL;
    solution_adjoint_parameter = NULL;

    possible_adjoint_order = 0;
    start_adjoint_from_forward_mode = true;

    stored_adjoint_order = 0;

    blasONE = 1;
    blasScalarONE = (Scalar)1.0;
    blasScalarMinusONE = (Scalar)-1.0;
    blasScalarZERO = (Scalar)0.0;

    // Integrator Configuration

    START_STEP_SIZE_SAFETY = (Scalar)0.90;
    NEWTON_STEP_REDUCTION = (Scalar)0.25;
    STEP_NOT_ACCEPTED_REDUCTION = (Scalar)0.25;
    STEP_NOT_ACCEPTED_SAVETY_STEP = (Scalar)0.50;

    ORDER_STAY_SAVETY_STEP = (Scalar)0.90;
    ORDER_UP_SAVETY_STEP = (Scalar)0.90;
    ORDER_DOWN_SAVETY_STEP = (Scalar)0.90;
    allowedSTEPS = 10;

    NEWTON_SAFETY = (Scalar)-1.00;
    NEWTON_CONVERGED = (Scalar)0.01;
    ACCEPT_NEWTON_CONVERGENCE_RATE = (Scalar)0.30;
    allowedNEWTON = 10;
    MAXNEWTONITERATION = 3;

    MAXORDER = 6;
    MACHINEPRECISION = std::numeric_limits<Scalar>::epsilon();
    HITSOLUTION = false;
    updateEveryP = false;

    linearSolver = new DenseLUSolver<Scalar>();
    ERROR_WEIGHTS_INDEX = 1;

    MINTAYLOR = 1;

    RESTRICT_ORDER_1_RATIO = (Scalar)3.000;
    RESTRICT_ORDER_2_RATIO = (Scalar)2.414;
    RESTRICT_ORDER_3_RATIO = (Scalar)1.618;
    RESTRICT_ORDER_4_RATIO = (Scalar)1.280;
    RESTRICT_ORDER_5_RATIO = (Scalar)1.127;
    RESTRICT_ORDER_6_RATIO = (Scalar)1.064;

    // Statistic

    acceptedSteps = 0;
    rejectedSteps = 0;
    rejectedNewtonSteps = 0;
    updateJacobian = 0;
    updateP = 0;
    newtonIterations = 0;
}

template <typename Scalar>
bool BDFIntegrator<Scalar>::resetIntegration(Scalar *y_start, Scalar max_StepLength,
                                             Scalar relative_error, Scalar *absolut_error,
                                             const IntegrationOptions<Scalar> &options) {
    return resetIntegration(y_start, max_StepLength, relative_error, absolut_error, options.taping,
                            options.direct, options.y_seed_list, options.p_seed_list,
                            options.relative_error_derivatives, options.absolut_error_derivatives,
                            options.numOfderivatives);
}

template <typename Scalar>
bool BDFIntegrator<Scalar>::resetIntegration(Scalar *y_start, Scalar T_start, Scalar max_StepLength,
                                             Scalar relative_error, Scalar *absolut_error,
                                             const IntegrationOptions<Scalar> &options) {
    return resetIntegration(y_start, T_start, max_StepLength, relative_error, absolut_error,
                            options.taping, options.direct, options.y_seed_list, options.p_seed_list,
                            options.relative_error_derivatives, options.absolut_error_derivatives,
                            options.numOfderivatives);
}

template <typename Scalar>
bool BDFIntegrator<Scalar>::resetIntegration(Scalar *y_start, Scalar T_start, Scalar *parameter,
                                             Scalar max_StepLength, Scalar relative_error,
                                             Scalar *absolut_error,
                                             const IntegrationOptions<Scalar> &options) {
    return resetIntegration(y_start, T_start, parameter, max_StepLength, relative_error,
                            absolut_error, options.taping, options.direct, options.y_seed_list,
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
    if (z_corrector == NULL)
        z_corrector = new Scalar[numOfSpecies * 7];

    if (z_corrector_old == NULL)
        z_corrector_old = new Scalar[numOfSpecies * 7];

    if (z_predictor == NULL)
        z_predictor = new Scalar[numOfSpecies * 7];

    order_corrector = 1;

    // Initialisiere Nordsiek-Array
    for (i = 0; i < numOfSpecies; i++)
        z_corrector[i] = y_start[i];

    for (i = 0; i < numOfSpecies; i++)
        absolut_error[i] = absolut_error_[i];
    relative_error = relative_error_;
    for (i = 0; i < numOfSpecies; i++)
        error_weights[i] = 0.0;
    freeSimultaneousDerivatives();
    start_integration_time = 0.0;
    t_i[1] = 0.0;
    estimateStartStepsize();
    CppAD::vector<Scalar> y_start_(numOfSpecies);
    for (i = 0; i < numOfSpecies; i++)
        y_start_[i] = y_start[i];
    CppAD::vector<Scalar> rhs(numOfSpecies);
    rhs = RHS->Forward(0, y_start_);
    t_i[0] = step_size_h;
    for (i = 0; i < numOfSpecies; i++)
        z_corrector[numOfSpecies + i] = rhs[i] * step_size_h;
    isIntegrattionStarted = true;
    isIntegrattionPerformed = false;
    num_of_steps_per_order = 0;
    newtonIteration_Started = false;
    convergenceIndicator = 0;
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
        stepSizeSet_new[i] = 0.0;
    step_size_h_old = step_size_h;

    // Simultaneous Derivatives
    setSimultaneousDerivatives(y_seed_derivatives_, p_seed_derivatives_,
                               relative_error_derivatives_, absolut_error_derivatives_,
                               numOfDerivatives_);

    adjointTapePointer = NULL;

    acceptedSteps = 0;
    rejectedSteps = 0;
    rejectedNewtonSteps = 0;
    updateJacobian = 0;
    updateP = 0;
    newtonIterations = 0;

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

    if (z_corrector == NULL)
        z_corrector = new Scalar[numOfSpecies * 7];

    if (z_corrector_old == NULL)
        z_corrector_old = new Scalar[numOfSpecies * 7];

    if (z_predictor == NULL)
        z_predictor = new Scalar[numOfSpecies * 7];

    order_corrector = 1;

    // Initialisiere Nordsiek-Array
    for (i = 0; i < numOfSpecies; i++)
        z_corrector[i] = y_start[i];

    for (i = 0; i < numOfSpecies; i++)
        absolut_error[i] = absolut_error_[i];
    relative_error = relative_error_;
    for (i = 0; i < numOfSpecies; i++)
        error_weights[i] = 0.0;
    freeSimultaneousDerivatives();

    start_integration_time = T_start;
    t_i[1] = T_start;

    estimateStartStepsize();

    t_i[0] = T_start + step_size_h;

    unsigned int cppAD_argument_size = 0;

    if (RHS_type == 0)
        cppAD_argument_size = numOfSpecies;
    if (RHS_type == 1)
        cppAD_argument_size = numOfSpecies + 1;

    CppAD::vector<Scalar> y_start_(cppAD_argument_size);
    for (i = 0; i < numOfSpecies; i++)
        y_start_[i] = y_start[i];
    if (RHS_type == 1)
        y_start_[numOfSpecies] = T_start;
    CppAD::vector<Scalar> rhs(numOfSpecies);
    rhs = RHS->Forward(0, y_start_);

    for (i = 0; i < numOfSpecies; i++)
        z_corrector[numOfSpecies + i] = rhs[i] * step_size_h;
    isIntegrattionStarted = true;
    isIntegrattionPerformed = false;
    num_of_steps_per_order = 0;
    newtonIteration_Started = false;
    convergenceIndicator = 0;
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
        stepSizeSet_new[i] = 0.0;
    step_size_h_old = step_size_h;
    setSimultaneousDerivatives(y_seed_derivatives_, p_seed_derivatives_,
                               relative_error_derivatives_, absolut_error_derivatives_,
                               numOfDerivatives_);

    adjointTapePointer = NULL;

    acceptedSteps = 0;
    rejectedSteps = 0;
    rejectedNewtonSteps = 0;
    updateJacobian = 0;
    updateP = 0;
    newtonIterations = 0;

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

    if (z_corrector == NULL)
        z_corrector = new Scalar[numOfSpecies * 7];

    if (z_corrector_old == NULL)
        z_corrector_old = new Scalar[numOfSpecies * 7];

    if (z_predictor == NULL)
        z_predictor = new Scalar[numOfSpecies * 7];

    order_corrector = 1;

    // Initialisiere Nordsiek-Array
    for (i = 0; i < numOfSpecies; i++)
        z_corrector[i] = y_start[i];

    for (i = 0; i < numOfSpecies; i++)
        absolut_error[i] = absolut_error_[i];
    relative_error = relative_error_;
    for (i = 0; i < numOfSpecies; i++)
        error_weights[i] = 0.0;
    freeSimultaneousDerivatives();

    start_integration_time = T_start;
    t_i[1] = T_start;

    estimateStartStepsize();

    t_i[0] = T_start + step_size_h;

    unsigned int cppAD_argument_size = 0;

    if (RHS_type == 0)
        cppAD_argument_size = numOfSpecies;
    if (RHS_type == 1)
        cppAD_argument_size = numOfSpecies + 1;
    if (RHS_type == 2)
        cppAD_argument_size = numOfSpecies + 1 + sizeOfParameter;

    CppAD::vector<Scalar> y_start_(cppAD_argument_size);
    for (i = 0; i < numOfSpecies; i++)
        y_start_[i] = y_start[i];
    if (RHS_type > 0)
        y_start_[numOfSpecies] = T_start;
    if (RHS_type == 2) {
        for (i = 0; i < sizeOfParameter; i++)
            y_start_[numOfSpecies + 1 + i] = parameter[i];
    }
    CppAD::vector<Scalar> rhs(numOfSpecies);
    rhs = RHS->Forward(0, y_start_);

    for (i = 0; i < numOfSpecies; i++)
        z_corrector[numOfSpecies + i] = rhs[i] * step_size_h;
    isIntegrattionStarted = true;
    isIntegrattionPerformed = false;
    num_of_steps_per_order = 0;
    newtonIteration_Started = false;
    convergenceIndicator = 0;
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
        stepSizeSet_new[i] = 0.0;
    step_size_h_old = step_size_h;
    setSimultaneousDerivatives(y_seed_derivatives_, p_seed_derivatives_,
                               relative_error_derivatives_, absolut_error_derivatives_,
                               numOfDerivatives_);

    adjointTapePointer = NULL;

    acceptedSteps = 0;
    rejectedSteps = 0;
    rejectedNewtonSteps = 0;
    updateJacobian = 0;
    updateP = 0;
    newtonIterations = 0;

    return true;
}

template <typename Scalar>
bool BDFIntegrator<Scalar>::performIntegration(const Scalar t_ende_,
                                               unsigned int maximal_Stepsize) {
    if (!start_adjoint_from_forward_mode)
        adjointTapePointer = NULL;
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

    if (t_i[1] >= t_ende) {
        calculate_output_vector();
        return true;
    }
    // Starte Integrationsroutine
    if (isIntegrattionPerformed)
        prepare_next_step();
    accept_indicator = 0;
    for (i = 0; i < maximal_Stepsize; i++) {
        for (j = 0; j < numOfSpecies; j++)
            corrector_old[j] = corrector[j];

        if (numOfDerivatives > 0)
            for (z = 0; z < numOfDerivatives; z++)
                for (j = 0; j < numOfSpecies; j++)
                    corrector_old_derivatives[z][j] = corrector_derivatives[z][j];

        setJacobian = false;
        setNewP = updateEveryP;
        newtonFAIL = 0;
        stepFAIL = 0;

        if (taping) {
            if (endTAPE == NULL)
                if (startTAPE == NULL) {
                    startTAPE = new bdf_TAPE;
                    startTAPE->backLINK = NULL;
                    startTAPE->forwardLINK = NULL;
                    endTAPE = startTAPE;
                } else
                    return false;
            else {
                endTAPE->forwardLINK = new bdf_TAPE;
                endTAPE->forwardLINK->backLINK = endTAPE;
                endTAPE->forwardLINK->forwardLINK = NULL;
                endTAPE = endTAPE->forwardLINK;
            }

            if (getTAPE == true)
                if (intermediateTAPE == NULL)
                    intermediateTAPE = endTAPE;

            endTAPE->P = NULL;
            endTAPE->endSolution = NULL;
            endTAPE->direct_fun = NULL;

            endTAPE->old_parameter_seed = NULL;
            endTAPE->old_y_seed = NULL;
            endTAPE->updatedTaylorFunction = false;
            endTAPE->lastTaylorOrder = 0;

            endTAPE->CppAD_tape = new CppAD::ADFun<Scalar> *[MAXNEWTONITERATION];
            for (j = 0; j < MAXNEWTONITERATION; j++)
                endTAPE->CppAD_tape[j] = NULL;
            endTAPE->ip = NULL;
            endTAPE->scaling_D1 = NULL;
            endTAPE->scaling_D2 = NULL;
            endTAPE->calculate_output = false;
            endTAPE->stepSize = 0.0;
            endTAPE->timePoint = t_i[1];
            endTAPE->orderUP = false;
            endTAPE->orderDOWN = false;
        }

        // SAVE INITIAL STEPSIZE_SET & Z_CORRECTOR
        for (j = 0; j < 6; j++)
            stepSizeSet_old[j] = stepSizeSet_new[j];
        for (j = 0; j < numOfSpecies; j++) {
            for (z = 0; z <= order_corrector; z++)
                z_corrector_old[z * numOfSpecies + j] = z_corrector[z * numOfSpecies + j];
            for (z = order_corrector + 1; z < 7; z++)
                z_corrector_old[z * numOfSpecies + j] = 0.0;
        }
        if (numOfDerivatives > 0) {
            for (z = 0; z < numOfDerivatives; z++)
                for (j = 0; j < numOfSpecies; j++) {
                    for (k = 0; k <= order_corrector; k++)
                        z_corrector_derivatives_old[z][numOfSpecies * k + j] =
                            z_corrector_derivatives[z][numOfSpecies * k + j];
                    for (k = order_corrector + 1; k < 7; k++)
                        z_corrector_derivatives_old[z][numOfSpecies * k + j] = 0.0;
                }
        }
        newtonFAIL = 0;
        stepFAIL = 0;
        while (accept_indicator == 0) {
            if (step_size_h != step_size_h_old) {
                Scalar temp = step_size_h / step_size_h_old;
                for (j = 0; j < numOfSpecies; j++)
                    for (z = 0; z <= order_corrector; z++)
                        z_corrector[z * numOfSpecies + j] =
                            z_corrector_old[z * numOfSpecies + j] * (Scalar)pow(temp, z);
                if (numOfDerivatives > 0)
                    for (z = 0; z < numOfDerivatives; z++)
                        for (j = 0; j < numOfSpecies; j++)
                            for (k = 0; k <= order_corrector; k++)
                                z_corrector_derivatives[z][numOfSpecies * k + j] =
                                    z_corrector_derivatives_old[z][numOfSpecies * k + j] *
                                    (Scalar)pow(temp, k);
            }
            for (j = 5; j > 0; j--)
                stepSizeSet_new[j] = stepSizeSet_old[j - 1] + step_size_h;
            stepSizeSet_new[0] = step_size_h;
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

                YABI_LOG_ERROR("calculate corrector vector failed!");

                return false;
            }
            accept_indicator = accept_corrector_vector();
        }
        calculate_next_corrector();
        step_size_h_old = step_size_h;
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
        solution_Vector_[i] = solution_Vector[i];
    return true;
}

template <typename Scalar>
bool BDFIntegrator<Scalar>::getSimultaneousDerivativesVector(
    Scalar **solution_Vector_derivatives_) {
    unsigned int i, z;
    for (z = 0; z < numOfDerivatives; z++)
        for (i = 0; i < numOfSpecies; i++)
            solution_Vector_derivatives_[z][i] = solution_Vector_derivatives[z][i];
    return true;
}

template <typename Scalar>
bool BDFIntegrator<Scalar>::getForwardSolutionVector(Scalar *forward_solution_Vector_) {
    unsigned int i;
    for (i = 0; i < numOfSpecies; i++)
        forward_solution_Vector_[i] = forward_solution_Vector[i];
    return true;
}

template <typename Scalar>
bool BDFIntegrator<Scalar>::getReverseSolutionVector(Scalar **solution_adjoint_species_,
                                                     Scalar **solution_adjoint_parameter_) {
    unsigned int i, j;
    for (i = 0; i < numOfSpecies; i++)
        for (j = 0; j < stored_adjoint_order; j++)
            solution_adjoint_species_[i][j] = solution_adjoint_species[i][j];

    for (i = 0; i < sizeOfParameter; i++)
        for (j = 0; j < stored_adjoint_order; j++)
            solution_adjoint_parameter_[i][j] = solution_adjoint_parameter[i][j];

    return true;
}

// Integrator Configuration

template <typename Scalar>
bool BDFIntegrator<Scalar>::setStepSettings(
    Scalar NEWTON_STEP_REDUCTION_, Scalar STEP_NOT_ACCEPTED_REDUCTION_,
    Scalar STEP_NOT_ACCEPTED_SAVETY_STEP_, Scalar ORDER_STAY_SAVETY_STEP_,
    Scalar ORDER_UP_SAVETY_STEP_, Scalar ORDER_DOWN_SAVETY_STEP_, unsigned int allowedSTEPS_) {
    if (0 < NEWTON_STEP_REDUCTION_ && NEWTON_STEP_REDUCTION_ < 1)
        NEWTON_STEP_REDUCTION = NEWTON_STEP_REDUCTION_;
    else
        return false;

    if (0 < STEP_NOT_ACCEPTED_REDUCTION_ && STEP_NOT_ACCEPTED_REDUCTION_ < 1)
        STEP_NOT_ACCEPTED_REDUCTION = STEP_NOT_ACCEPTED_REDUCTION_;
    else
        return false;

    if (0 < ORDER_STAY_SAVETY_STEP_ && ORDER_STAY_SAVETY_STEP_ < 1)
        ORDER_STAY_SAVETY_STEP = ORDER_STAY_SAVETY_STEP_;
    else
        return false;

    if (0 < ORDER_UP_SAVETY_STEP_ && ORDER_UP_SAVETY_STEP_ < 1)
        ORDER_UP_SAVETY_STEP = ORDER_UP_SAVETY_STEP_;
    else
        return false;

    if (0 < ORDER_DOWN_SAVETY_STEP_ && ORDER_DOWN_SAVETY_STEP_ < 1)
        ORDER_DOWN_SAVETY_STEP = ORDER_DOWN_SAVETY_STEP_;
    else
        return false;

    if (allowedSTEPS_ > 0)
        allowedSTEPS = allowedSTEPS_;
    else
        return false;

    if (0 < STEP_NOT_ACCEPTED_SAVETY_STEP_ && STEP_NOT_ACCEPTED_SAVETY_STEP_ <= 1)
        STEP_NOT_ACCEPTED_SAVETY_STEP = STEP_NOT_ACCEPTED_SAVETY_STEP_;
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
    NEWTON_SAFETY = NEWTON_SAFETY_;

    if (0 < NEWTON_CONVERGED_)
        NEWTON_CONVERGED = NEWTON_CONVERGED_;
    else
        return false;

    if (0 <= ACCEPT_NEWTON_CONVERGENCE_RATE_)
        ACCEPT_NEWTON_CONVERGENCE_RATE = ACCEPT_NEWTON_CONVERGENCE_RATE_;
    else
        return false;

    if (allowedNEWTON_ > 0)
        allowedNEWTON = allowedNEWTON_;
    else
        return false;

    if (MAXNEWTONITERATION_ > 0)
        MAXNEWTONITERATION = MAXNEWTONITERATION_;
    else
        return false;

    updateEveryP = updateEveryP_;

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

    MAXORDER = MAXORDER_;
    MACHINEPRECISION = MACHINEPRECISION_;
    HITSOLUTION = HITSOLUTION_;
    ERROR_WEIGHTS_INDEX = ERROR_WEIGHTS_INDEX_;

    return true;
}

template <typename Scalar>
bool BDFIntegrator<Scalar>::setRestrictOrderRatio(
    Scalar restrictRatioOrder1, Scalar restrictRatioOrder2, Scalar restrictRatioOrder3,
    Scalar restrictRatioOrder4, Scalar restrictRatioOrder5, Scalar restrictRatioOrder6) {
    RESTRICT_ORDER_1_RATIO = restrictRatioOrder1;
    RESTRICT_ORDER_2_RATIO = restrictRatioOrder2;
    RESTRICT_ORDER_3_RATIO = restrictRatioOrder3;
    RESTRICT_ORDER_4_RATIO = restrictRatioOrder4;
    RESTRICT_ORDER_5_RATIO = restrictRatioOrder5;
    RESTRICT_ORDER_6_RATIO = restrictRatioOrder6;
    return true;
}

template <typename Scalar>
bool BDFIntegrator<Scalar>::setMinTaylorCapacity(unsigned int MINTAYLOR_) {
    MINTAYLOR = MINTAYLOR_;

    RHS->capacity_order(MINTAYLOR);
    for (typename cppADFunListType::iterator it = cppADFunList.begin(); it != cppADFunList.end();
         it++) {
        (*it)->capacity_order(MINTAYLOR);
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
        CppAD::ADFun<Scalar> *tempADFun = new CppAD::ADFun<Scalar>();
        (*tempADFun) = (*RHS);
        tempADFun->capacity_order(MINTAYLOR);
        cppADFunList.push_back(tempADFun);
    }

    return true;
}

// DESTRUCTOR AND FREE MEMORY

template <typename Scalar> BDFIntegrator<Scalar>::~BDFIntegrator() {
    unsigned int i;

    delete[] forward_solution_Vector;
    if (P_tape != NULL)
        delete[] P_tape;
    if (linearSolver) {
        delete linearSolver;
        linearSolver = NULL;
    }
    P_tape = NULL;
    if (ip_tape != NULL)
        delete[] ip_tape;
    ip_tape = NULL;
    if (scaling_D1_tape != NULL)
        delete[] scaling_D1_tape;
    scaling_D1_tape = NULL;
    if (scaling_D2_tape != NULL)
        delete[] scaling_D2_tape;
    scaling_D2_tape = NULL;
    delete[] correctorVector_tape;
    if (z_corrector_tape != NULL) {
        delete[] z_corrector_tape;
        z_corrector_tape = NULL;
    }
    if (z_predictor_tape != NULL) {
        delete[] z_predictor_tape;
        z_predictor_tape = NULL;
    }
    if (z_corrector_old != NULL) {
        delete[] z_corrector_old;
        z_corrector_old = NULL;
    }

    deleteOldTape();
    delete[] parameter;
    delete[] ip;
    delete[] scaling_D1;
    delete[] scaling_D2;
    if (linearSolver) {
        delete linearSolver;
        linearSolver = NULL;
    }
    delete[] P;
    delete[] J;
    delete[] error_weights;
    delete[] absolut_error;
    delete[] solution_Vector;
    delete[] corrector_old;
    delete[] corrector;
    delete RHS;
    if (z_predictor != NULL) {
        delete[] z_predictor;
        z_predictor = NULL;
    }
    if (z_corrector != NULL) {
        delete[] z_corrector;
        z_corrector = NULL;
    }

    // Simultaneous Derivatives

    if (error_weights_derivatives != NULL) {
        for (i = 0; i < numOfDerivatives; i++)
            delete[] error_weights_derivatives[i];
        delete[] error_weights_derivatives;
    }
    error_weights_derivatives = NULL;
    absolut_error_derivatives = NULL;
    relative_error_derivatives = NULL;

    if (z_predictor_derivatives != NULL) {
        for (i = 0; i < numOfDerivatives; i++)
            delete[] z_predictor_derivatives[i];
        delete[] z_predictor_derivatives;
    }
    z_predictor_derivatives = NULL;
    if (z_corrector_derivatives != NULL) {
        for (i = 0; i < numOfDerivatives; i++)
            delete[] z_corrector_derivatives[i];
        delete[] z_corrector_derivatives;
    }
    z_corrector_derivatives = NULL;
    if (z_corrector_derivatives_old != NULL) {
        for (i = 0; i < numOfDerivatives; i++)
            delete[] z_corrector_derivatives_old[i];
        delete[] z_corrector_derivatives_old;
    }
    z_corrector_derivatives_old = NULL;
    if (corrector_derivatives != NULL) {
        for (i = 0; i < numOfDerivatives; i++)
            delete[] corrector_derivatives[i];
        delete[] corrector_derivatives;
    }
    corrector_derivatives = NULL;
    if (corrector_old_derivatives != NULL) {
        for (i = 0; i < numOfDerivatives; i++)
            delete[] corrector_old_derivatives[i];
        delete[] corrector_old_derivatives;
    }
    corrector_old_derivatives = NULL;
    if (solution_Vector_derivatives != NULL) {
        for (i = 0; i < numOfDerivatives; i++)
            delete[] solution_Vector_derivatives[i];
        delete[] solution_Vector_derivatives;
    }
    solution_Vector_derivatives = NULL;

    for (typename cppADFunListType::iterator it = cppADFunList.begin(); it != cppADFunList.end();
         it++) {
        delete (*it);
        (*it) = NULL;
    }
    cppADFunList.resize(0);

    if (reachableError_derivatives != NULL)
        delete[] reachableError_derivatives;

    // adjoint
    if (solution_adjoint_species != NULL) {
        for (i = 0; i < numOfSpecies; i++)
            delete[] solution_adjoint_species[i];
        delete[] solution_adjoint_species;
    }
    if (solution_adjoint_parameter != NULL) {
        for (i = 0; i < sizeOfParameter; i++)
            delete[] solution_adjoint_parameter[i];
        delete[] solution_adjoint_parameter;
    }
}

// Sensitivity Generation

// FORWARD MODE

} // namespace yabi

#endif
