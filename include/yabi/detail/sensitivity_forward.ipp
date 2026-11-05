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
// Forward-mode sensitivity replay.
//
// Generated from the reference implementation in YABI_old without changing any
// expression: the bodies below are the original algorithm verbatim.

#ifndef YABI_DETAIL_SENSITIVITY_FORWARD_IPP
#define YABI_DETAIL_SENSITIVITY_FORWARD_IPP

namespace yabi {

template <typename Scalar>
bool BDFIntegrator<Scalar>::forward(unsigned int order, Scalar *y_seed, Scalar *parameter_seed,
                                    bool reset) {
    return sensitivityEngine.forward(order, y_seed, parameter_seed, reset);
}

// Reverse Mode

} // namespace yabi

#endif
