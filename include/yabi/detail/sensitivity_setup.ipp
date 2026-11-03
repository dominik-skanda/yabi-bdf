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
// Sensitivity storage and forward-tape lifetime.
//
// Generated from the reference implementation in YABI_old without changing any
// expression: the bodies below are the original algorithm verbatim.

#ifndef YABI_DETAIL_SENSITIVITY_SETUP_IPP
#define YABI_DETAIL_SENSITIVITY_SETUP_IPP

namespace yabi {

template <typename Scalar>
bool BDFIntegrator<Scalar>::setSimultaneousDerivatives(Scalar **y_seed_derivatives_,
                                                       Scalar **p_seed_derivatives_,
                                                       Scalar *relative_error_derivatives_,
                                                       Scalar **absolut_error_derivatives_,
                                                       unsigned int numOfDerivatives_) {
    unsigned int i, j, k;
    numOfDerivatives = numOfDerivatives_;
    if (numOfDerivatives > 0 && y_seed_derivatives_ != NULL) {
        relative_error_derivatives = relative_error_derivatives_;
        absolut_error_derivatives = absolut_error_derivatives_;
        y_seed_derivatives = y_seed_derivatives_;
        p_seed_derivatives = p_seed_derivatives_;
        reachableError_derivatives = new Scalar[numOfDerivatives];
        z_predictor_derivatives = new Scalar *[numOfDerivatives];
        z_corrector_derivatives = new Scalar *[numOfDerivatives];
        z_corrector_derivatives_old = new Scalar *[numOfDerivatives];
        corrector_derivatives = new Scalar *[numOfDerivatives];
        corrector_old_derivatives = new Scalar *[numOfDerivatives];
        solution_Vector_derivatives = new Scalar *[numOfDerivatives];
        error_weights_derivatives = new Scalar *[numOfDerivatives];
        for (i = 0; i < numOfDerivatives; i++) {
            z_predictor_derivatives[i] = new Scalar[numOfSpecies * 7];
            z_corrector_derivatives[i] = new Scalar[numOfSpecies * 7];
            z_corrector_derivatives_old[i] = new Scalar[numOfSpecies * 7];
            corrector_derivatives[i] = new Scalar[numOfSpecies];
            corrector_old_derivatives[i] = new Scalar[numOfSpecies];
            solution_Vector_derivatives[i] = new Scalar[numOfSpecies];
            error_weights_derivatives[i] = new Scalar[numOfSpecies];
            for (j = 0; j < numOfSpecies; j++) {
                error_weights_derivatives[i][j] = 0.0;
                for (k = 0; k < 7; k++)
                    z_predictor_derivatives[i][numOfSpecies * k + j] = 0.0;
                z_corrector_derivatives[i][j] = y_seed_derivatives[i][j];
                z_corrector_derivatives_old[i][j] = y_seed_derivatives[i][j];
            }

            unsigned int cppAD_argument_size = 0;
            if (RHS_type == 0)
                cppAD_argument_size = numOfSpecies;
            if (RHS_type == 1)
                cppAD_argument_size = numOfSpecies + 1;
            if (RHS_type == 2)
                cppAD_argument_size = numOfSpecies + 1 + sizeOfParameter;

            std::vector<Scalar> cppAD_argument(cppAD_argument_size);
            for (j = 0; j < numOfSpecies; j++)
                cppAD_argument[j] = y_seed_derivatives[i][j];

            if (RHS_type > 0)
                cppAD_argument[numOfSpecies] = 0.0;

            if (RHS_type > 1)
                for (j = 0; j < sizeOfParameter; j++)
                    cppAD_argument[numOfSpecies + 1 + j] = p_seed_derivatives[i][j];

            std::vector<Scalar> df(numOfSpecies);
            df = RHS->Forward(1, cppAD_argument);

            for (j = 0; j < numOfSpecies; j++) {
                z_corrector_derivatives[i][numOfSpecies + j] = step_size_h * df[j];
                z_corrector_derivatives_old[i][numOfSpecies + j] = step_size_h * df[j];
            }
            for (j = 0; j < numOfSpecies; j++)
                for (k = 2; k < 7; k++) {
                    z_corrector_derivatives[i][numOfSpecies * k + j] = 0.0;
                    z_corrector_derivatives_old[i][numOfSpecies * k + j] = 0.0;
                }
        }
    } else
        numOfDerivatives = 0;
    return true;
}

template <typename Scalar> bool BDFIntegrator<Scalar>::freeSimultaneousDerivatives() {
    unsigned int i;

    if (error_weights_derivatives != NULL) {
        for (i = 0; i < numOfDerivatives; i++)
            delete[] error_weights_derivatives[i];
        delete[] error_weights_derivatives;
    }
    error_weights_derivatives = NULL;

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

    if (z_corrector_derivatives_old != NULL) {
        for (i = 0; i < numOfDerivatives; i++)
            delete[] z_corrector_derivatives_old[i];
        delete[] z_corrector_derivatives_old;
    }
    z_corrector_derivatives_old = NULL;

    if (reachableError_derivatives != NULL)
        delete[] reachableError_derivatives;
    reachableError_derivatives = NULL;

    numOfDerivatives = 0;
    return true;
}

template <typename Scalar> bool BDFIntegrator<Scalar>::deleteOldTape() {
    if (z_corrector_tape != NULL) {
        delete[] z_corrector_tape;
        z_corrector_tape = NULL;
    }
    if (z_predictor_tape != NULL) {
        delete[] z_predictor_tape;
        z_predictor_tape = NULL;
    }
    while (endTAPE != NULL) {
        if (endTAPE->endSolution != NULL)
            delete[] endTAPE->endSolution;
        if (endTAPE->direct_fun != NULL)
            endTAPE->direct_fun = NULL;

        if (endTAPE->old_parameter_seed != NULL)
            delete[] endTAPE->old_parameter_seed;

        if (endTAPE->old_y_seed != NULL)
            delete[] endTAPE->old_y_seed;

        if (endTAPE->CppAD_tape != NULL) {
            delete[] endTAPE->CppAD_tape;
        }
        if (endTAPE->P != NULL)
            delete[] endTAPE->P;
        if (endTAPE->ip != NULL)
            delete[] endTAPE->ip;
        if (endTAPE->scaling_D1 != NULL)
            delete[] endTAPE->scaling_D1;
        if (endTAPE->scaling_D2 != NULL)
            delete[] endTAPE->scaling_D2;
        intermediateTAPE = endTAPE->backLINK;
        delete endTAPE;
        endTAPE = intermediateTAPE;
    }
    startTAPE = NULL;
    intermediateTAPE = NULL;
    endTAPE = NULL;

    start_CppAD_tape = NULL;

    if (P_tape != NULL)
        delete[] P_tape;
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

    return true;
}

template <typename Scalar> bool BDFIntegrator<Scalar>::createNewTape() {
    deleteOldTape();
    intermediateCppADFunIterator = cppADFunList.begin();
    getTAPE = false;
    lastForwardOrder = 0;

    if (start_CppAD_tape == NULL) {
        if (intermediateCppADFunIterator == cppADFunList.end()) {
            std::unique_ptr<AbstractRHS<Scalar>> tempADFun(RHS->clone());
    tempADFun->capacity_order(MINTAYLOR);
    cppADFunList.push_back(std::move(tempADFun));
            intermediateCppADFunIterator--;
        }
        start_CppAD_tape = intermediateCppADFunIterator->get();
        intermediateCppADFunIterator++;
    } else {
        YABI_LOG_ERROR("CREATE NEW TAPE FAILED!");

        return false;
    }

    startTape_stepSize = step_size_h;
    unsigned int i;

    unsigned int cppAD_argument_size = 0;

    if (RHS_type == 0)
        cppAD_argument_size = numOfSpecies;
    if (RHS_type == 1)
        cppAD_argument_size = numOfSpecies + 1;
    if (RHS_type == 2)
        cppAD_argument_size = numOfSpecies + 1 + sizeOfParameter;

    std::vector<Scalar> cppAD_argument(cppAD_argument_size);

    for (i = 0; i < numOfSpecies; i++)
        cppAD_argument[i] = z_corrector[i];
    if (RHS_type >= 1)
        cppAD_argument[numOfSpecies] = start_integration_time;
    if (RHS_type == 2)
        for (i = 0; i < sizeOfParameter; i++)
            cppAD_argument[numOfSpecies + 1 + i] = parameter[i];

    start_CppAD_tape->Forward(0, cppAD_argument);

    for (i = 0; i < numOfSpecies; i++)
        forward_solution_Vector[i] = 0.0;

    return true;
}

} // namespace yabi

#endif
