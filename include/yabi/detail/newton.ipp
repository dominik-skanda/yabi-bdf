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
// Newton iteration, Jacobian assembly and corrector.data() acceptance.
//
// Generated from the reference implementation in YABI_old without changing any
// expression: the bodies below are the original algorithm verbatim.

#ifndef YABI_DETAIL_NEWTON_IPP
#define YABI_DETAIL_NEWTON_IPP

namespace yabi {

template <typename Scalar> inline bool BDFIntegrator<Scalar>::calculate_newton_scaling() {
    return newtonSolver.calculateNewtonScaling();
}

template <typename Scalar> bool BDFIntegrator<Scalar>::calculate_corrector_vector() {
    return newtonSolver.calculateCorrectorVector();
}

template <typename Scalar> inline bool BDFIntegrator<Scalar>::calculate_next_corrector() {
    return nordsieck.calculate_next_corrector(numOfSpecies, numOfDerivatives, corrector_derivatives,
                                              z_predictor_derivatives, z_corrector_derivatives);
}

template <typename Scalar> unsigned int BDFIntegrator<Scalar>::accept_corrector_vector() {
    return newtonSolver.acceptCorrectorVector();
}

} // namespace yabi

#endif
