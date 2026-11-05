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
// Reverse/adjoint sensitivity replay: public façade over detail::AdjointEngine.
//
// The algorithm lives in detail/adjoint_engine.hpp and is the original replay
// body verbatim; this file only forwards.

#ifndef YABI_DETAIL_SENSITIVITY_REVERSE_IPP
#define YABI_DETAIL_SENSITIVITY_REVERSE_IPP

namespace yabi {

template <typename Scalar>
bool BDFIntegrator<Scalar>::reverse(unsigned int order, Scalar *seed_weights) {
    return adjointEngine.reverse(order, seed_weights);
}

} // namespace yabi

#endif
