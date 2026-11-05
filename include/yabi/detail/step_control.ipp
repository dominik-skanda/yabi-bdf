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
// Step-size selection, predictor, order control and output.
//
// Generated from the reference implementation in YABI_old without changing any
// expression: the bodies below are the original algorithm verbatim.

#ifndef YABI_DETAIL_STEP_CONTROL_IPP
#define YABI_DETAIL_STEP_CONTROL_IPP

namespace yabi {

template <typename Scalar>
inline bool BDFIntegrator<Scalar>::calculate_error_weights(Scalar *errorWeights, Scalar *yWeight,
                                                           Scalar *absoluteError,
                                                           Scalar &relativeError) {
    return detail::StepController<Scalar>::calculateErrorWeights(
        numOfSpecies, config.ERROR_WEIGHTS_INDEX, errorWeights, yWeight, absoluteError,
        relativeError);
}

template <typename Scalar> bool BDFIntegrator<Scalar>::estimateStartStepsize() {
    return stepController.estimateStartStepsize(*RHS, numOfSpecies, RHS_type, sizeOfParameter,
                                                parameter, error_weights, absolut_error,
                                                relative_error, max_StepLength);
}

template <typename Scalar> inline bool BDFIntegrator<Scalar>::calculate_l_vector() {
    if (!isIntegrattionStarted)
        return false;
    return nordsieck.calculate_l_vector();
}

template <typename Scalar> inline bool BDFIntegrator<Scalar>::calculate_next_predictor() {
    return nordsieck.calculate_next_predictor(numOfSpecies, sensitivityEngine.numOfDerivatives,
                                              sensitivityEngine.z_corrector_derivatives,
                                              sensitivityEngine.z_predictor_derivatives);
}

template <typename Scalar> bool BDFIntegrator<Scalar>::prepare_next_step() {
    detail::StepDecision decision = stepController.prepareNextStep(
        numOfSpecies, sensitivityEngine.numOfDerivatives, error_weights, reachableError,
        sensitivityEngine.reachableError_derivatives, sensitivityEngine.z_corrector_derivatives,
        sensitivityEngine.corrector_derivatives, sensitivityEngine.corrector_old_derivatives,
        sensitivityEngine.error_weights_derivatives, minimalEta, max_StepLength, t_ende);

    if (taping) {
        if (decision.orderUp && decision.orderDown)
            return false;

        if (!decision.orderUp && !decision.orderDown)
            return true;

        if (endTAPE == NULL)
            return false;

        ReplayTape *tape = createReplayTapeNode();
        if (sensitivityEngine.getTAPE == true && intermediateTAPE == NULL)
            intermediateTAPE = endTAPE;

        tape->calculate_output = false;
        tape->stepSize = nordsieck.step_size_h;
        tape->orderUP = decision.orderUp;
        tape->orderDOWN = decision.orderDown;
    }
    return true;
}

template <typename Scalar> inline bool BDFIntegrator<Scalar>::calculate_output_vector() {
    if (internalError)
        return false;

    nordsieck.calculate_output_vector(numOfSpecies, sensitivityEngine.numOfDerivatives,
                                      sensitivityEngine.z_corrector_derivatives,
                                      sensitivityEngine.solution_Vector_derivatives, t_ende);

    if (taping) {
        if (endTAPE == NULL && startTAPE != NULL)
            return false;

        ReplayTape *tape = createReplayTapeNode();
        if (sensitivityEngine.getTAPE == true && intermediateTAPE == NULL)
            intermediateTAPE = endTAPE;

        tape->calculate_output = true;
        tape->stepSize = nordsieck.step_size_h;
        tape->timePoint = t_ende;
        tape->orderUP = false;
        tape->orderDOWN = false;

        sensitivityEngine.adjointTapePointer = endTAPE;
        sensitivityEngine.possible_adjoint_order = 1;
        sensitivityEngine.start_adjoint_from_forward_mode = false;
    }
    return true;
}

// Write out Output

} // namespace yabi

#endif
