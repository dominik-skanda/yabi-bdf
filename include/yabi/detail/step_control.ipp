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
    unsigned int i;
    if (config.ERROR_WEIGHTS_INDEX == 0) {
        for (i = 0; i < numOfSpecies; i++) {
            if (errorWeights[i] < fabs(yWeight[i]))
                errorWeights[i] = fabs(yWeight[i]);
            if (errorWeights[i] < absoluteError[i])
                errorWeights[i] = absoluteError[i];
        }
    }
    if (config.ERROR_WEIGHTS_INDEX == 1) {
        for (i = 0; i < numOfSpecies; i++)
            errorWeights[i] = fabs(yWeight[i]) + (absoluteError[i] / relativeError);
    }
    return true;
}

template <typename Scalar> bool BDFIntegrator<Scalar>::estimateStartStepsize() {
    unsigned int i, j;

    unsigned int argumentSize = 0;
    if (RHS_type == 0)
        argumentSize = numOfSpecies;
    if (RHS_type == 1)
        argumentSize = numOfSpecies + 1;
    if (RHS_type == 2)
        argumentSize = numOfSpecies + 1 + sizeOfParameter;
    if (argumentSize == 0)
        return false;

    std::vector<Scalar> y_start(argumentSize);
    for (i = 0; i < numOfSpecies; i++)
        y_start[i] = nordsieck.z_corrector[i];
    if (RHS_type >= 1)
        y_start[numOfSpecies] = nordsieck.t_i[1];
    if (RHS_type == 2)
        for (i = 0; i < sizeOfParameter; i++)
            y_start[numOfSpecies + 1 + i] = parameter[i];

    std::vector<Scalar> rhs(numOfSpecies);
    rhs = RHS->Forward(0, y_start);

    // Jacobian F_y(y_start,t_start,p)
    std::vector<Scalar> J(numOfSpecies * numOfSpecies);
    std::vector<Scalar> dx(argumentSize);
    std::vector<Scalar> df(numOfSpecies);
    for (j = 0; j < argumentSize; j++)
        dx[j] = 0.0;
    for (j = 0; j < numOfSpecies; j++) {
        dx[j] = 1.0;
        df = RHS->Forward(1, dx);
        dx[j] = 0.0;
        for (i = 0; i < numOfSpecies; i++)
            J[i * numOfSpecies + j] = df[i];
    }

    // Second derivative of the solution:
    // y'' = F_y * F + F_t  (F_t is zero for autonomous systems).
    std::vector<Scalar> y_ddot(numOfSpecies);
    for (i = 0; i < numOfSpecies; i++) {
        y_ddot[i] = 0.0;
        for (j = 0; j < numOfSpecies; j++)
            y_ddot[i] += J[i * numOfSpecies + j] * rhs[j];
    }
    if (RHS_type >= 1) {
        std::vector<Scalar> dx_time(argumentSize);
        for (j = 0; j < argumentSize; j++)
            dx_time[j] = 0.0;
        dx_time[numOfSpecies] = 1.0;
        RHS->Forward(0, y_start);
        df = RHS->Forward(1, dx_time);
        for (i = 0; i < numOfSpecies; i++)
            y_ddot[i] += df[i];
    }

    // CALCULATE ERROR WEIGHTS
    calculate_error_weights(error_weights.data(), &y_start[0], absolut_error.data(), relative_error);

    nordsieck.step_size_h = 0.0;
    Scalar temp = 0.0;
    for (i = 0; i < numOfSpecies; i++) {
        temp = (y_ddot[i] / error_weights[i]);
        nordsieck.step_size_h += temp * temp;
    }
    nordsieck.step_size_h = sqrt(nordsieck.step_size_h) / (Scalar)numOfSpecies;

    // If the second derivative vanishes (e.g. y_start=0 and F(y_start)=0),
    // the estimate is not defined.  Fall back to the maximum step size as
    // described in the dissertation.
    if (!(nordsieck.step_size_h > 0.0) || !std::isfinite(nordsieck.step_size_h))
        nordsieck.step_size_h = max_StepLength;
    else {
        nordsieck.step_size_h =
            sqrt((2.0 * relative_error * config.START_STEP_SIZE_SAFETY) / nordsieck.step_size_h);
        if (!(nordsieck.step_size_h > 0.0) || !std::isfinite(nordsieck.step_size_h))
            nordsieck.step_size_h = max_StepLength;
    }
    if (nordsieck.step_size_h > max_StepLength)
        nordsieck.step_size_h = max_StepLength;
    // For purely autonomous systems the start step heuristic can be overly
    // aggressive for very stiff problems (e.g. Robertson). Be more
    // conservative to resolve fast transients.
    if (RHS_type == 0) {
        if (max_StepLength > 0) {
            if (nordsieck.step_size_h > max_StepLength * 1e-4)
                nordsieck.step_size_h = max_StepLength * 1e-4;
        }
        if (!(nordsieck.step_size_h > 0.0) || !std::isfinite(nordsieck.step_size_h)) {
            if (max_StepLength > 0)
                nordsieck.step_size_h = max_StepLength * 1e-4;
            else
                nordsieck.step_size_h = 1e-12;
        }
    }
    return true;
}

template <typename Scalar> inline bool BDFIntegrator<Scalar>::calculate_l_vector() {
    if (!isIntegrattionStarted)
        return false;
    return nordsieck.calculate_l_vector();
}

template <typename Scalar> inline bool BDFIntegrator<Scalar>::calculate_next_predictor() {
    return nordsieck.calculate_next_predictor(numOfSpecies, numOfDerivatives,
                                              z_corrector_derivatives, z_predictor_derivatives);
}

template <typename Scalar> bool BDFIntegrator<Scalar>::prepare_next_step() {
    unsigned int i, j, z;
    bool isOrderUp = false;
    bool isOrderDown = false;

    // c_n
    Scalar c_n = 1.0;
    for (i = 1; i < nordsieck.order_corrector; i++)
        c_n *=
            nordsieck.stepSizeSet_new[i] / (nordsieck.stepSizeSet_new[i] - nordsieck.step_size_h);
    c_n += 1.0;
    for (i = 1; i <= nordsieck.order_corrector; i++)
        c_n = c_n / (i + 1);

    for (i = 0; i < nordsieck.order_corrector; i++)
        c_n *= nordsieck.stepSizeSet_new[i] / nordsieck.step_size_h;

    if (nordsieck.num_of_steps_per_order < nordsieck.order_corrector + 1) {
        Scalar eta = minimalEta;

        if (config.HITSOLUTION)
            if ((eta * nordsieck.step_size_h) + nordsieck.t_i[1] > t_ende)
                eta = (t_ende - nordsieck.t_i[1]) / nordsieck.step_size_h;

        if (eta * nordsieck.step_size_h < max_StepLength) {
            nordsieck.step_size_h = nordsieck.step_size_h * eta;
            if (nordsieck.step_size_h < 4.0 * config.MACHINEPRECISION * nordsieck.t_i[1]) {
                YABI_LOG_ERROR("relaxing step size!");

                nordsieck.step_size_h = 4.0 * config.MACHINEPRECISION * nordsieck.t_i[1];
            }
            nordsieck.t_i[0] = nordsieck.step_size_h + nordsieck.t_i[1];
        } else {
            eta = max_StepLength / nordsieck.step_size_h;
            nordsieck.step_size_h = nordsieck.step_size_h * eta;
            if (nordsieck.step_size_h < 4.0 * config.MACHINEPRECISION * nordsieck.t_i[1]) {
                YABI_LOG_ERROR("relaxing step size!");

                nordsieck.step_size_h = 4.0 * config.MACHINEPRECISION * nordsieck.t_i[1];
            }
            nordsieck.t_i[0] = nordsieck.step_size_h + nordsieck.t_i[1];
        }
    } else {
        unsigned int i;

        // Ordnungssteuerung

        Scalar normWeightedLocalTruncError;

        Scalar eta_q_minus = 0.0;

        Scalar temp1 = 0.0;
        Scalar *temp2 = new Scalar[numOfSpecies];

        if (nordsieck.order_corrector > 1) {
            temp1 = -1.0;
            for (i = 0; i < nordsieck.order_corrector - 1; i++)
                temp1 *= (nordsieck.stepSizeSet_new[i]) / nordsieck.step_size_h;

            temp1 /=
                (nordsieck.l_i[1] - (nordsieck.step_size_h /
                                     nordsieck.stepSizeSet_new[nordsieck.order_corrector - 1]));

            for (i = 0; i < numOfSpecies; i++)
                temp2[i] = nordsieck.z_corrector[numOfSpecies * nordsieck.order_corrector + i] *
                           temp1 / error_weights[i];

            normWeightedLocalTruncError = linalg::NRM2(numOfSpecies, temp2, blasONE) / numOfSpecies;

            Scalar mu = config.ORDER_DOWN_SAVETY_STEP * (reachableError / normWeightedLocalTruncError);
            Scalar sumTemp = 0.0;
            for (i = 1; i <= nordsieck.order_corrector - 1; i++)
                sumTemp += (1.0 / i);

            eta_q_minus = sumTemp;
            sumTemp = 0.0;
            for (i = 0; i < nordsieck.order_corrector - 1; i++)
                sumTemp += (nordsieck.step_size_h / nordsieck.stepSizeSet_new[i]);
            eta_q_minus /= sumTemp;
            for (i = 2; i <= nordsieck.order_corrector - 1; i++)
                eta_q_minus *= nordsieck.stepSizeSet_new[i - 1] / nordsieck.step_size_h / i;
            eta_q_minus = pow(eta_q_minus, 1.0 / nordsieck.order_corrector);
            eta_q_minus *= pow(mu, 1.0 / nordsieck.order_corrector);

            if (numOfDerivatives > 0) {
                Scalar eta_q_minus_derivatives;
                for (z = 0; z < numOfDerivatives; z++) {
                    for (i = 0; i < numOfSpecies; i++)
                        temp2[i] =
                            z_corrector_derivatives[z]
                                                   [numOfSpecies * nordsieck.order_corrector + i] *
                            temp1 / error_weights_derivatives[z][i];

                    normWeightedLocalTruncError =
                        linalg::NRM2(numOfSpecies, temp2, blasONE) / numOfSpecies;

                    mu = config.ORDER_DOWN_SAVETY_STEP *
                         (reachableError_derivatives[z] / normWeightedLocalTruncError);
                    sumTemp = 0.0;
                    for (i = 1; i <= nordsieck.order_corrector - 1; i++)
                        sumTemp += (1.0 / i);

                    eta_q_minus_derivatives = sumTemp;
                    sumTemp = 0.0;
                    for (i = 0; i < nordsieck.order_corrector - 1; i++)
                        sumTemp += (nordsieck.step_size_h / nordsieck.stepSizeSet_new[i]);
                    eta_q_minus_derivatives /= sumTemp;
                    for (i = 2; i <= nordsieck.order_corrector - 1; i++)
                        eta_q_minus_derivatives *=
                            nordsieck.stepSizeSet_new[i - 1] / nordsieck.step_size_h / i;
                    eta_q_minus_derivatives =
                        pow(eta_q_minus_derivatives, 1.0 / nordsieck.order_corrector);
                    eta_q_minus_derivatives *= pow(mu, 1.0 / nordsieck.order_corrector);

                    if (eta_q_minus_derivatives < eta_q_minus)
                        eta_q_minus = eta_q_minus_derivatives;
                }
            }
        }

        Scalar eta_q_plus = 0.0;

        if (nordsieck.order_corrector < config.MAXORDER) {
            temp1 = (Scalar)pow(
                (nordsieck.step_size_h / (nordsieck.stepSizeSet_new[1] - nordsieck.step_size_h)),
                nordsieck.order_corrector + 1);
            temp1 *= (c_n / nordsieck.c_n_old);

            Scalar temp3 = 1.0;
            for (i = 1; i < nordsieck.order_corrector; i++)
                temp3 *= nordsieck.stepSizeSet_new[i] /
                         (nordsieck.stepSizeSet_new[i] - nordsieck.step_size_h);
            temp3 += 1;
            temp3 = -nordsieck.stepSizeSet_new[nordsieck.order_corrector] /
                    (nordsieck.step_size_h * temp3 *
                     (nordsieck.l_i[1] + (nordsieck.step_size_h /
                                          nordsieck.stepSizeSet_new[nordsieck.order_corrector])) *
                     (Scalar)(nordsieck.order_corrector + 2));

            for (i = 0; i < numOfSpecies; i++)
                temp2[i] = temp3 * (nordsieck.corrector[i] - (nordsieck.corrector_old[i] * temp1)) /
                           error_weights[i];

            normWeightedLocalTruncError = linalg::NRM2(numOfSpecies, temp2, blasONE) / numOfSpecies;

            Scalar mu = config.ORDER_UP_SAVETY_STEP * (reachableError / normWeightedLocalTruncError);
            Scalar sumTemp = 0.0;
            for (i = 1; i <= nordsieck.order_corrector + 1; i++)
                sumTemp += (1.0 / i);

            eta_q_plus = sumTemp;
            sumTemp = 0.0;
            for (i = 0; i < nordsieck.order_corrector + 1; i++)
                sumTemp += (nordsieck.step_size_h / nordsieck.stepSizeSet_new[i]);
            eta_q_plus /= sumTemp;
            for (i = 2; i <= nordsieck.order_corrector + 1; i++)
                eta_q_plus *= nordsieck.stepSizeSet_new[i - 1] / nordsieck.step_size_h / i;
            eta_q_plus = pow(eta_q_plus, 1.0 / (nordsieck.order_corrector + 2));
            eta_q_plus *= pow(mu, 1.0 / (nordsieck.order_corrector + 2));

            if (numOfDerivatives > 0) {
                Scalar eta_q_plus_derivatives;
                for (z = 0; z < numOfDerivatives; z++) {
                    for (i = 0; i < numOfSpecies; i++)
                        temp2[i] = temp3 *
                                   (corrector_derivatives[z][i] -
                                    (corrector_old_derivatives[z][i] * temp1)) /
                                   error_weights_derivatives[z][i];

                    normWeightedLocalTruncError =
                        linalg::NRM2(numOfSpecies, temp2, blasONE) / numOfSpecies;

                    mu = config.ORDER_UP_SAVETY_STEP *
                         (reachableError_derivatives[z] / normWeightedLocalTruncError);
                    sumTemp = 0.0;
                    for (i = 1; i <= nordsieck.order_corrector + 1; i++)
                        sumTemp += (1.0 / i);

                    eta_q_plus_derivatives = sumTemp;
                    sumTemp = 0.0;
                    for (i = 0; i < nordsieck.order_corrector + 1; i++)
                        sumTemp += (nordsieck.step_size_h / nordsieck.stepSizeSet_new[i]);
                    eta_q_plus_derivatives /= sumTemp;
                    for (i = 2; i <= nordsieck.order_corrector + 1; i++)
                        eta_q_plus_derivatives *=
                            nordsieck.stepSizeSet_new[i - 1] / nordsieck.step_size_h / i;
                    eta_q_plus_derivatives =
                        pow(eta_q_plus_derivatives, 1.0 / (nordsieck.order_corrector + 2));
                    eta_q_plus_derivatives *= pow(mu, 1.0 / (nordsieck.order_corrector + 2));

                    if (eta_q_plus_derivatives < eta_q_plus)
                        eta_q_plus = eta_q_plus_derivatives;
                }
            }
        }

        delete[] temp2;
        Scalar eta_q = minimalEta;

        if (nordsieck.order_corrector - 1 == 1)
            if (eta_q_minus > config.RESTRICT_ORDER_1_RATIO)
                eta_q_minus = config.RESTRICT_ORDER_1_RATIO;

        if (nordsieck.order_corrector - 1 == 2)
            if (eta_q_minus > config.RESTRICT_ORDER_2_RATIO)
                eta_q_minus = config.RESTRICT_ORDER_2_RATIO;

        if (nordsieck.order_corrector - 1 == 3)
            if (eta_q_minus > config.RESTRICT_ORDER_3_RATIO)
                eta_q_minus = config.RESTRICT_ORDER_3_RATIO;

        if (nordsieck.order_corrector - 1 == 4)
            if (eta_q_minus > config.RESTRICT_ORDER_4_RATIO)
                eta_q_minus = config.RESTRICT_ORDER_4_RATIO;

        if (nordsieck.order_corrector - 1 == 5)
            if (eta_q_minus > config.RESTRICT_ORDER_5_RATIO)
                eta_q_minus = config.RESTRICT_ORDER_5_RATIO;

        if (nordsieck.order_corrector - 1 == 6)
            if (eta_q_minus > config.RESTRICT_ORDER_6_RATIO)
                eta_q_minus = config.RESTRICT_ORDER_6_RATIO;

        if (nordsieck.order_corrector + 1 == 1)
            if (eta_q_plus > config.RESTRICT_ORDER_1_RATIO)
                eta_q_plus = config.RESTRICT_ORDER_1_RATIO;

        if (nordsieck.order_corrector + 1 == 2)
            if (eta_q_plus > config.RESTRICT_ORDER_2_RATIO)
                eta_q_plus = config.RESTRICT_ORDER_2_RATIO;

        if (nordsieck.order_corrector + 1 == 3)
            if (eta_q_plus > config.RESTRICT_ORDER_3_RATIO)
                eta_q_plus = config.RESTRICT_ORDER_3_RATIO;

        if (nordsieck.order_corrector + 1 == 4)
            if (eta_q_plus > config.RESTRICT_ORDER_4_RATIO)
                eta_q_plus = config.RESTRICT_ORDER_4_RATIO;

        if (nordsieck.order_corrector + 1 == 5)
            if (eta_q_plus > config.RESTRICT_ORDER_5_RATIO)
                eta_q_plus = config.RESTRICT_ORDER_5_RATIO;

        if (nordsieck.order_corrector + 1 == 6)
            if (eta_q_plus > config.RESTRICT_ORDER_6_RATIO)
                eta_q_plus = config.RESTRICT_ORDER_6_RATIO;

        if (config.HITSOLUTION)
            if ((eta_q * nordsieck.step_size_h) + nordsieck.t_i[1] > t_ende)
                eta_q = (t_ende - nordsieck.t_i[1]) / nordsieck.step_size_h;

        if (eta_q * nordsieck.step_size_h > max_StepLength)
            eta_q = max_StepLength / nordsieck.step_size_h;

        if (config.HITSOLUTION)
            if ((eta_q_plus * nordsieck.step_size_h) + nordsieck.t_i[1] > t_ende)
                eta_q_plus = (t_ende - nordsieck.t_i[1]) / nordsieck.step_size_h;

        if (eta_q_plus * nordsieck.step_size_h > max_StepLength)
            eta_q_plus = max_StepLength / nordsieck.step_size_h;

        if (config.HITSOLUTION)
            if ((eta_q_minus * nordsieck.step_size_h) + nordsieck.t_i[1] > t_ende)
                eta_q_minus = (t_ende - nordsieck.t_i[1]) / nordsieck.step_size_h;

        if (eta_q_minus * nordsieck.step_size_h > max_StepLength)
            eta_q_minus = max_StepLength / nordsieck.step_size_h;

        if (eta_q < eta_q_minus && nordsieck.order_corrector > config.MINORDER) {
            if (eta_q_minus < eta_q_plus && nordsieck.order_corrector < config.MAXORDER) {
                for (i = 0; i < numOfSpecies; i++)
                    for (j = nordsieck.order_corrector + 1; j < 7; j++)
                        nordsieck.z_corrector[numOfSpecies * j + i] = 0.0;

                if (numOfDerivatives > 0) {
                    for (z = 0; z < numOfDerivatives; z++)
                        for (i = 0; i < numOfSpecies; i++)
                            for (j = nordsieck.order_corrector + 1; j < 7; j++)
                                z_corrector_derivatives[z][numOfSpecies * j + i] = 0.0;
                }

                nordsieck.order_corrector += 1;
                nordsieck.step_size_h = nordsieck.step_size_h * eta_q_plus;
                if (nordsieck.step_size_h < 4.0 * config.MACHINEPRECISION * nordsieck.t_i[1]) {
                    YABI_LOG_ERROR("relaxing step size!");

                    nordsieck.step_size_h = 4.0 * config.MACHINEPRECISION * nordsieck.t_i[1];
                }
                nordsieck.t_i[0] = nordsieck.step_size_h + nordsieck.t_i[1];
                nordsieck.num_of_steps_per_order = 0.0;
                isOrderUp = true;
            } else {
                Scalar *d_kahan = new Scalar[4];
                for (i = 0; i < 4; i++)
                    d_kahan[i] = 0.0;
                Scalar y;
                Scalar t;

                Scalar *d_i = new Scalar[6];
                d_i[0] = 0.0;
                d_i[1] = 0.0;
                for (i = 2; i <= nordsieck.order_corrector; i++)
                    for (j = i; j > 1; j--) {
                        if (j == i) {
                            if (j != 6)
                                d_i[j] = 1.0;
                        } else {
                            // d_i[j]=d_i[j-1]+d_i[j]*stepSizeSet_new[i-3]/step_size_h;
                            y = d_i[j - 1] - (d_kahan[j - 2] * nordsieck.stepSizeSet_new[i - 3] /
                                              nordsieck.step_size_h);
                            t = (d_i[j] * nordsieck.stepSizeSet_new[i - 3] /
                                 nordsieck.step_size_h) +
                                y;
                            d_kahan[j - 2] = (t - d_i[j]) - y;
                            d_i[j] = t;
                        }
                    }

                Scalar *temp = new Scalar[numOfSpecies];
                for (i = 0; i < numOfSpecies; i++)
                    temp[i] = nordsieck.z_corrector[nordsieck.order_corrector * numOfSpecies + i];

                Scalar blasTEMP = -1.0;
                linalg::GER(numOfSpecies, nordsieck.order_corrector, blasTEMP, temp, blasONE, d_i,
                            blasONE, nordsieck.z_corrector.data(), numOfSpecies);

                if (numOfDerivatives > 0) {
                    for (z = 0; z < numOfDerivatives; z++) {
                        for (i = 0; i < numOfSpecies; i++)
                            temp[i] = z_corrector_derivatives[z][nordsieck.order_corrector *
                                                                     numOfSpecies +
                                                                 i];
                        Scalar blasTEMP = -1.0;
                        linalg::GER(numOfSpecies, nordsieck.order_corrector, blasTEMP, temp,
                                    blasONE, d_i, blasONE, z_corrector_derivatives[z],
                                    numOfSpecies);
                    }
                }

                delete[] temp;
                nordsieck.order_corrector -= 1;
                nordsieck.step_size_h = nordsieck.step_size_h * eta_q_minus;
                if (nordsieck.step_size_h < 4.0 * config.MACHINEPRECISION * nordsieck.t_i[1]) {
                    YABI_LOG_ERROR("relaxing step size!");

                    nordsieck.step_size_h = 4.0 * config.MACHINEPRECISION * nordsieck.t_i[1];
                }
                nordsieck.t_i[0] = nordsieck.step_size_h + nordsieck.t_i[1];

                delete[] d_i;
                delete[] d_kahan;
                nordsieck.num_of_steps_per_order = 0;
                isOrderDown = true;
            }
        } else if ((eta_q < eta_q_plus || nordsieck.order_corrector < config.MINORDER) &&
                   nordsieck.order_corrector < config.MAXORDER) {
            for (i = 0; i < numOfSpecies; i++)
                for (j = nordsieck.order_corrector + 1; j < 7; j++)
                    nordsieck.z_corrector[numOfSpecies * j + i] = 0.0;

            if (numOfDerivatives > 0) {
                for (z = 0; z < numOfDerivatives; z++)
                    for (i = 0; i < numOfSpecies; i++)
                        for (j = nordsieck.order_corrector + 1; j < 7; j++)
                            z_corrector_derivatives[z][numOfSpecies * j + i] = 0.0;
            }

            nordsieck.order_corrector += 1;
            nordsieck.step_size_h = nordsieck.step_size_h * eta_q_plus;
            if (nordsieck.step_size_h < 4.0 * config.MACHINEPRECISION * nordsieck.t_i[1]) {
                YABI_LOG_ERROR("relaxing step size!");

                nordsieck.step_size_h = 4.0 * config.MACHINEPRECISION * nordsieck.t_i[1];
            }
            nordsieck.t_i[0] = nordsieck.step_size_h + nordsieck.t_i[1];
            nordsieck.num_of_steps_per_order = 0;
            isOrderUp = true;
        } else {
            if (!std::isinf(eta_q)) {
                nordsieck.step_size_h = nordsieck.step_size_h * eta_q;
                if (nordsieck.step_size_h < 4 * config.MACHINEPRECISION * nordsieck.t_i[1]) {
                    YABI_LOG_ERROR("relaxing step size!");

                    nordsieck.step_size_h = 4.0 * config.MACHINEPRECISION * nordsieck.t_i[1];
                }
                nordsieck.t_i[0] = nordsieck.step_size_h + nordsieck.t_i[1];
            } else {
                eta_q = max_StepLength / nordsieck.step_size_h;
                nordsieck.step_size_h = nordsieck.step_size_h * eta_q;
                if (nordsieck.step_size_h < 4.0 * config.MACHINEPRECISION * nordsieck.t_i[1]) {
                    YABI_LOG_ERROR("relaxing step size!");

                    nordsieck.step_size_h = 4.0 * config.MACHINEPRECISION * nordsieck.t_i[1];
                }
                nordsieck.t_i[0] = nordsieck.step_size_h + nordsieck.t_i[1];
            }
        }
    }
    nordsieck.c_n_old = c_n;
    if (taping) {
        if (isOrderUp == true && isOrderDown == true)
            return false;

        if (isOrderUp == false && isOrderDown == false)
            return true;

        if (endTAPE == NULL)
            return false;

        ReplayTape *tape = createReplayTapeNode();
        if (getTAPE == true && intermediateTAPE == NULL)
            intermediateTAPE = endTAPE;

        tape->calculate_output = false;
        if (isOrderUp) {
            tape->orderDOWN = false;
            tape->orderUP = true;
            tape->stepSize = nordsieck.step_size_h;
        }
        if (isOrderDown) {
            tape->orderUP = false;
            tape->orderDOWN = true;
            tape->stepSize = nordsieck.step_size_h;
        }
    }
    return true;
}

template <typename Scalar> inline bool BDFIntegrator<Scalar>::calculate_output_vector() {
    if (internalError)
        return false;

    nordsieck.calculate_output_vector(numOfSpecies, numOfDerivatives, z_corrector_derivatives,
                                      solution_Vector_derivatives, t_ende);

    if (taping) {
        if (endTAPE == NULL && startTAPE != NULL)
            return false;

        ReplayTape *tape = createReplayTapeNode();
        if (getTAPE == true && intermediateTAPE == NULL)
            intermediateTAPE = endTAPE;

        tape->calculate_output = true;
        tape->stepSize = nordsieck.step_size_h;
        tape->timePoint = t_ende;
        tape->orderUP = false;
        tape->orderDOWN = false;

        adjointTapePointer = endTAPE;
        possible_adjoint_order = 1;
        start_adjoint_from_forward_mode = false;
    }
    return true;
}

// Write out Output

} // namespace yabi

#endif
