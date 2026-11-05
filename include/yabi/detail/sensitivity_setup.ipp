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
    return sensitivityEngine.setSimultaneousDerivatives(
        y_seed_derivatives_, p_seed_derivatives_, relative_error_derivatives_,
        absolut_error_derivatives_, numOfDerivatives_);
}

template <typename Scalar> bool BDFIntegrator<Scalar>::freeSimultaneousDerivatives() {
    return sensitivityEngine.freeSimultaneousDerivatives();
}

template <typename Scalar> bool BDFIntegrator<Scalar>::deleteOldTape() {
    // The store owns every node (including order-change markers), so releasing
    // the whole tape is a single clear().
    replayTapeStore.clear();

    startTAPE = nullptr;
    intermediateTAPE = nullptr;
    endTAPE = nullptr;
    sensitivityEngine.resetForwardBuffers();

    start_CppAD_tape = nullptr;

    return true;
}

template <typename Scalar> bool BDFIntegrator<Scalar>::createNewTape() {
    deleteOldTape();
    intermediateCppADFunIterator = cppADFunList.begin();
    sensitivityEngine.beginNewTape();

    if (start_CppAD_tape == NULL) {
        if (intermediateCppADFunIterator == cppADFunList.end()) {
            std::unique_ptr<AbstractRHS<Scalar>> tempADFun(RHS->clone());
    tempADFun->capacity_order(config.MINTAYLOR);
    cppADFunList.push_back(std::move(tempADFun));
            intermediateCppADFunIterator--;
        }
        start_CppAD_tape = intermediateCppADFunIterator->get();
        intermediateCppADFunIterator++;
    } else {
        YABI_LOG_ERROR("CREATE NEW TAPE FAILED!");

        return false;
    }

    startTape_stepSize = nordsieck.step_size_h;
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
        cppAD_argument[i] = nordsieck.z_corrector[i];
    if (RHS_type >= 1)
        cppAD_argument[numOfSpecies] = start_integration_time;
    if (RHS_type == 2)
        for (i = 0; i < sizeOfParameter; i++)
            cppAD_argument[numOfSpecies + 1 + i] = parameter[i];

    start_CppAD_tape->Forward(0, cppAD_argument);

    sensitivityEngine.zeroForwardSolution();

    return true;
}

} // namespace yabi

#endif
