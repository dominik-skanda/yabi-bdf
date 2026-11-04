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
    if (ERROR_WEIGHTS_INDEX == 0) {
        for (i = 0; i < numOfSpecies; i++) {
            if (errorWeights[i] < fabs(yWeight[i]))
                errorWeights[i] = fabs(yWeight[i]);
            if (errorWeights[i] < absoluteError[i])
                errorWeights[i] = absoluteError[i];
        }
    }
    if (ERROR_WEIGHTS_INDEX == 1) {
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
        y_start[i] = z_corrector[i];
    if (RHS_type >= 1)
        y_start[numOfSpecies] = t_i[1];
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

    step_size_h = 0.0;
    Scalar temp = 0.0;
    for (i = 0; i < numOfSpecies; i++) {
        temp = (y_ddot[i] / error_weights[i]);
        step_size_h += temp * temp;
    }
    step_size_h = sqrt(step_size_h) / (Scalar)numOfSpecies;

    // If the second derivative vanishes (e.g. y_start=0 and F(y_start)=0),
    // the estimate is not defined.  Fall back to the maximum step size as
    // described in the dissertation.
    if (!(step_size_h > 0.0) || !std::isfinite(step_size_h))
        step_size_h = max_StepLength;
    else {
        step_size_h = sqrt((2.0 * relative_error * START_STEP_SIZE_SAFETY) / step_size_h);
        if (!(step_size_h > 0.0) || !std::isfinite(step_size_h))
            step_size_h = max_StepLength;
    }
    if (step_size_h > max_StepLength)
        step_size_h = max_StepLength;
    // For purely autonomous systems the start step heuristic can be overly
    // aggressive for very stiff problems (e.g. Robertson). Be more
    // conservative to resolve fast transients.
    if (RHS_type == 0) {
        if (max_StepLength > 0) {
            if (step_size_h > max_StepLength * 1e-4)
                step_size_h = max_StepLength * 1e-4;
        }
        if (!(step_size_h > 0.0) || !std::isfinite(step_size_h)) {
            if (max_StepLength > 0)
                step_size_h = max_StepLength * 1e-4;
            else
                step_size_h = 1e-12;
        }
    }
    return true;
}

template <typename Scalar> inline bool BDFIntegrator<Scalar>::calculate_l_vector() {
    unsigned int i, j, k;
    if (!isIntegrattionStarted)
        return false;

    Scalar *l_kahan = new Scalar[6];
    for (i = 0; i < 6; i++)
        l_kahan[i] = 0.0;
    Scalar y;
    Scalar t;

    l_i[0] = 1.0;
    for (i = 1; i <= order_corrector; i++)
        for (j = i; j > 0; j--) {
            if (j == i) {
                l_i[j] = 1.0;
                for (k = 0; k < i; k++)
                    l_i[j] *= (step_size_h / stepSizeSet_new[k]);
            } else {
                // l_i[j]=l_i[j]+(l_i[j-1]*(step_size_h/stepSizeSet_new[i-1]));
                y = (l_i[j - 1] * (step_size_h / stepSizeSet_new[i - 1])) - l_kahan[j - 1];
                t = l_i[j] + y;
                l_kahan[j - 1] = (t - l_i[j]) - y;
                l_i[j] = t;
            }
        }
    delete[] l_kahan;
    return true;
}

template <typename Scalar> inline bool BDFIntegrator<Scalar>::calculate_next_predictor() {
    unsigned int z;

    static Scalar A1[4] = {1., 0., 1., 1.};
    static Scalar A2[9] = {1., 0., 0., 1., 1., 0., 1., 2., 1.};
    static Scalar A3[16] = {1., 0., 0., 0., 1., 1., 0., 0., 1., 2., 1., 0., 1., 3., 3., 1.};
    static Scalar A4[25] = {1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 2., 1.,
                            0., 0., 1., 3., 3., 1., 0., 1., 4., 6., 4., 1.};
    static Scalar A5[36] = {1., 0., 0., 0., 0., 0., 1., 1., 0.,  0.,  0., 0.,
                            1., 2., 1., 0., 0., 0., 1., 3., 3.,  1.,  0., 0.,
                            1., 4., 6., 4., 1., 0., 1., 5., 10., 10., 5., 1.};

    static Scalar A6[49] = {1., 0., 0., 0.,  0.,  0., 0., 1., 1., 0., 0.,  0.,  0.,  0., 1., 2., 1.,
                            0., 0., 0., 0.,  1.,  3., 3., 1., 0., 0., 0.,  1.,  4.,  6., 4., 1., 0.,
                            0., 1., 5., 10., 10., 5., 1., 0., 1., 6., 15., 20., 15., 6., 1.};

    if (order_corrector + 1 > 7)
        return false;

    int blasTEMP;
    switch (order_corrector) {
    case 1:
        blasTEMP = 2;
        linalg::DGEMM_NT_T(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE, z_corrector.data(),
                           numOfSpecies, A1, blasTEMP, blasScalarZERO, z_predictor.data(), numOfSpecies);
        break;
    case 2:
        blasTEMP = 3;
        linalg::DGEMM_NT_T(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE, z_corrector.data(),
                           numOfSpecies, A2, blasTEMP, blasScalarZERO, z_predictor.data(), numOfSpecies);
        break;
    case 3:
        blasTEMP = 4;
        linalg::DGEMM_NT_T(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE, z_corrector.data(),
                           numOfSpecies, A3, blasTEMP, blasScalarZERO, z_predictor.data(), numOfSpecies);
        break;
    case 4:
        blasTEMP = 5;
        linalg::DGEMM_NT_T(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE, z_corrector.data(),
                           numOfSpecies, A4, blasTEMP, blasScalarZERO, z_predictor.data(), numOfSpecies);
        break;
    case 5:
        blasTEMP = 6;
        linalg::DGEMM_NT_T(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE, z_corrector.data(),
                           numOfSpecies, A5, blasTEMP, blasScalarZERO, z_predictor.data(), numOfSpecies);
        break;
    case 6:
        blasTEMP = 7;
        linalg::DGEMM_NT_T(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE, z_corrector.data(),
                           numOfSpecies, A6, blasTEMP, blasScalarZERO, z_predictor.data(), numOfSpecies);
        break;
    }

    if (numOfDerivatives > 0) {
        for (z = 0; z < numOfDerivatives; z++) {
            switch (order_corrector) {
            case 1:
                blasTEMP = 2;
                linalg::DGEMM_NT_T(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE,
                                   z_corrector_derivatives[z], numOfSpecies, A1, blasTEMP,
                                   blasScalarZERO, z_predictor_derivatives[z], numOfSpecies);
                break;
            case 2:
                blasTEMP = 3;
                linalg::DGEMM_NT_T(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE,
                                   z_corrector_derivatives[z], numOfSpecies, A2, blasTEMP,
                                   blasScalarZERO, z_predictor_derivatives[z], numOfSpecies);
                break;
            case 3:
                blasTEMP = 4;
                linalg::DGEMM_NT_T(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE,
                                   z_corrector_derivatives[z], numOfSpecies, A3, blasTEMP,
                                   blasScalarZERO, z_predictor_derivatives[z], numOfSpecies);
                break;
            case 4:
                blasTEMP = 5;
                linalg::DGEMM_NT_T(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE,
                                   z_corrector_derivatives[z], numOfSpecies, A4, blasTEMP,
                                   blasScalarZERO, z_predictor_derivatives[z], numOfSpecies);
                break;
            case 5:
                blasTEMP = 6;
                linalg::DGEMM_NT_T(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE,
                                   z_corrector_derivatives[z], numOfSpecies, A5, blasTEMP,
                                   blasScalarZERO, z_predictor_derivatives[z], numOfSpecies);
                break;
            case 6:
                blasTEMP = 7;
                linalg::DGEMM_NT_T(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE,
                                   z_corrector_derivatives[z], numOfSpecies, A6, blasTEMP,
                                   blasScalarZERO, z_predictor_derivatives[z], numOfSpecies);
                break;
            }
        }
    }

    return true;
}

template <typename Scalar> bool BDFIntegrator<Scalar>::prepare_next_step() {
    unsigned int i, j, z;
    bool isOrderUp = false;
    bool isOrderDown = false;

    // c_n
    Scalar c_n = 1.0;
    for (i = 1; i < order_corrector; i++)
        c_n *= stepSizeSet_new[i] / (stepSizeSet_new[i] - step_size_h);
    c_n += 1.0;
    for (i = 1; i <= order_corrector; i++)
        c_n = c_n / (i + 1);

    for (i = 0; i < order_corrector; i++)
        c_n *= stepSizeSet_new[i] / step_size_h;

    if (num_of_steps_per_order < order_corrector + 1) {
        Scalar eta = minimalEta;

        if (HITSOLUTION)
            if ((eta * step_size_h) + t_i[1] > t_ende)
                eta = (t_ende - t_i[1]) / step_size_h;

        if (eta * step_size_h < max_StepLength) {
            step_size_h = step_size_h * eta;
            if (step_size_h < 4.0 * MACHINEPRECISION * t_i[1]) {
                YABI_LOG_ERROR("relaxing step size!");

                step_size_h = 4.0 * MACHINEPRECISION * t_i[1];
            }
            t_i[0] = step_size_h + t_i[1];
        } else {
            eta = max_StepLength / step_size_h;
            step_size_h = step_size_h * eta;
            if (step_size_h < 4.0 * MACHINEPRECISION * t_i[1]) {
                YABI_LOG_ERROR("relaxing step size!");

                step_size_h = 4.0 * MACHINEPRECISION * t_i[1];
            }
            t_i[0] = step_size_h + t_i[1];
        }
    } else {
        unsigned int i;

        // Ordnungssteuerung

        Scalar normWeightedLocalTruncError;

        Scalar eta_q_minus = 0.0;

        Scalar temp1 = 0.0;
        Scalar *temp2 = new Scalar[numOfSpecies];

        if (order_corrector > 1) {
            temp1 = -1.0;
            for (i = 0; i < order_corrector - 1; i++)
                temp1 *= (stepSizeSet_new[i]) / step_size_h;

            temp1 /= (l_i[1] - (step_size_h / stepSizeSet_new[order_corrector - 1]));

            for (i = 0; i < numOfSpecies; i++)
                temp2[i] =
                    z_corrector[numOfSpecies * order_corrector + i] * temp1 / error_weights[i];

            normWeightedLocalTruncError = linalg::NRM2(numOfSpecies, temp2, blasONE) / numOfSpecies;

            Scalar mu = ORDER_DOWN_SAVETY_STEP * (reachableError / normWeightedLocalTruncError);
            Scalar sumTemp = 0.0;
            for (i = 1; i <= order_corrector - 1; i++)
                sumTemp += (1.0 / i);

            eta_q_minus = sumTemp;
            sumTemp = 0.0;
            for (i = 0; i < order_corrector - 1; i++)
                sumTemp += (step_size_h / stepSizeSet_new[i]);
            eta_q_minus /= sumTemp;
            for (i = 2; i <= order_corrector - 1; i++)
                eta_q_minus *= stepSizeSet_new[i - 1] / step_size_h / i;
            eta_q_minus = pow(eta_q_minus, 1.0 / order_corrector);
            eta_q_minus *= pow(mu, 1.0 / order_corrector);

            if (numOfDerivatives > 0) {
                Scalar eta_q_minus_derivatives;
                for (z = 0; z < numOfDerivatives; z++) {
                    for (i = 0; i < numOfSpecies; i++)
                        temp2[i] = z_corrector_derivatives[z][numOfSpecies * order_corrector + i] *
                                   temp1 / error_weights_derivatives[z][i];

                    normWeightedLocalTruncError =
                        linalg::NRM2(numOfSpecies, temp2, blasONE) / numOfSpecies;

                    mu = ORDER_DOWN_SAVETY_STEP *
                         (reachableError_derivatives[z] / normWeightedLocalTruncError);
                    sumTemp = 0.0;
                    for (i = 1; i <= order_corrector - 1; i++)
                        sumTemp += (1.0 / i);

                    eta_q_minus_derivatives = sumTemp;
                    sumTemp = 0.0;
                    for (i = 0; i < order_corrector - 1; i++)
                        sumTemp += (step_size_h / stepSizeSet_new[i]);
                    eta_q_minus_derivatives /= sumTemp;
                    for (i = 2; i <= order_corrector - 1; i++)
                        eta_q_minus_derivatives *= stepSizeSet_new[i - 1] / step_size_h / i;
                    eta_q_minus_derivatives = pow(eta_q_minus_derivatives, 1.0 / order_corrector);
                    eta_q_minus_derivatives *= pow(mu, 1.0 / order_corrector);

                    if (eta_q_minus_derivatives < eta_q_minus)
                        eta_q_minus = eta_q_minus_derivatives;
                }
            }
        }

        Scalar eta_q_plus = 0.0;

        if (order_corrector < MAXORDER) {
            temp1 = (Scalar)pow((step_size_h / (stepSizeSet_new[1] - step_size_h)),
                                order_corrector + 1);
            temp1 *= (c_n / c_n_old);

            Scalar temp3 = 1.0;
            for (i = 1; i < order_corrector; i++)
                temp3 *= stepSizeSet_new[i] / (stepSizeSet_new[i] - step_size_h);
            temp3 += 1;
            temp3 =
                -stepSizeSet_new[order_corrector] /
                (step_size_h * temp3 * (l_i[1] + (step_size_h / stepSizeSet_new[order_corrector])) *
                 (Scalar)(order_corrector + 2));

            for (i = 0; i < numOfSpecies; i++)
                temp2[i] = temp3 * (corrector[i] - (corrector_old[i] * temp1)) / error_weights[i];

            normWeightedLocalTruncError = linalg::NRM2(numOfSpecies, temp2, blasONE) / numOfSpecies;

            Scalar mu = ORDER_UP_SAVETY_STEP * (reachableError / normWeightedLocalTruncError);
            Scalar sumTemp = 0.0;
            for (i = 1; i <= order_corrector + 1; i++)
                sumTemp += (1.0 / i);

            eta_q_plus = sumTemp;
            sumTemp = 0.0;
            for (i = 0; i < order_corrector + 1; i++)
                sumTemp += (step_size_h / stepSizeSet_new[i]);
            eta_q_plus /= sumTemp;
            for (i = 2; i <= order_corrector + 1; i++)
                eta_q_plus *= stepSizeSet_new[i - 1] / step_size_h / i;
            eta_q_plus = pow(eta_q_plus, 1.0 / (order_corrector + 2));
            eta_q_plus *= pow(mu, 1.0 / (order_corrector + 2));

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

                    mu = ORDER_UP_SAVETY_STEP *
                         (reachableError_derivatives[z] / normWeightedLocalTruncError);
                    sumTemp = 0.0;
                    for (i = 1; i <= order_corrector + 1; i++)
                        sumTemp += (1.0 / i);

                    eta_q_plus_derivatives = sumTemp;
                    sumTemp = 0.0;
                    for (i = 0; i < order_corrector + 1; i++)
                        sumTemp += (step_size_h / stepSizeSet_new[i]);
                    eta_q_plus_derivatives /= sumTemp;
                    for (i = 2; i <= order_corrector + 1; i++)
                        eta_q_plus_derivatives *= stepSizeSet_new[i - 1] / step_size_h / i;
                    eta_q_plus_derivatives =
                        pow(eta_q_plus_derivatives, 1.0 / (order_corrector + 2));
                    eta_q_plus_derivatives *= pow(mu, 1.0 / (order_corrector + 2));

                    if (eta_q_plus_derivatives < eta_q_plus)
                        eta_q_plus = eta_q_plus_derivatives;
                }
            }
        }

        delete[] temp2;
        Scalar eta_q = minimalEta;

        if (order_corrector - 1 == 1)
            if (eta_q_minus > RESTRICT_ORDER_1_RATIO)
                eta_q_minus = RESTRICT_ORDER_1_RATIO;

        if (order_corrector - 1 == 2)
            if (eta_q_minus > RESTRICT_ORDER_2_RATIO)
                eta_q_minus = RESTRICT_ORDER_2_RATIO;

        if (order_corrector - 1 == 3)
            if (eta_q_minus > RESTRICT_ORDER_3_RATIO)
                eta_q_minus = RESTRICT_ORDER_3_RATIO;

        if (order_corrector - 1 == 4)
            if (eta_q_minus > RESTRICT_ORDER_4_RATIO)
                eta_q_minus = RESTRICT_ORDER_4_RATIO;

        if (order_corrector - 1 == 5)
            if (eta_q_minus > RESTRICT_ORDER_5_RATIO)
                eta_q_minus = RESTRICT_ORDER_5_RATIO;

        if (order_corrector - 1 == 6)
            if (eta_q_minus > RESTRICT_ORDER_6_RATIO)
                eta_q_minus = RESTRICT_ORDER_6_RATIO;

        if (order_corrector + 1 == 1)
            if (eta_q_plus > RESTRICT_ORDER_1_RATIO)
                eta_q_plus = RESTRICT_ORDER_1_RATIO;

        if (order_corrector + 1 == 2)
            if (eta_q_plus > RESTRICT_ORDER_2_RATIO)
                eta_q_plus = RESTRICT_ORDER_2_RATIO;

        if (order_corrector + 1 == 3)
            if (eta_q_plus > RESTRICT_ORDER_3_RATIO)
                eta_q_plus = RESTRICT_ORDER_3_RATIO;

        if (order_corrector + 1 == 4)
            if (eta_q_plus > RESTRICT_ORDER_4_RATIO)
                eta_q_plus = RESTRICT_ORDER_4_RATIO;

        if (order_corrector + 1 == 5)
            if (eta_q_plus > RESTRICT_ORDER_5_RATIO)
                eta_q_plus = RESTRICT_ORDER_5_RATIO;

        if (order_corrector + 1 == 6)
            if (eta_q_plus > RESTRICT_ORDER_6_RATIO)
                eta_q_plus = RESTRICT_ORDER_6_RATIO;

        if (HITSOLUTION)
            if ((eta_q * step_size_h) + t_i[1] > t_ende)
                eta_q = (t_ende - t_i[1]) / step_size_h;

        if (eta_q * step_size_h > max_StepLength)
            eta_q = max_StepLength / step_size_h;

        if (HITSOLUTION)
            if ((eta_q_plus * step_size_h) + t_i[1] > t_ende)
                eta_q_plus = (t_ende - t_i[1]) / step_size_h;

        if (eta_q_plus * step_size_h > max_StepLength)
            eta_q_plus = max_StepLength / step_size_h;

        if (HITSOLUTION)
            if ((eta_q_minus * step_size_h) + t_i[1] > t_ende)
                eta_q_minus = (t_ende - t_i[1]) / step_size_h;

        if (eta_q_minus * step_size_h > max_StepLength)
            eta_q_minus = max_StepLength / step_size_h;

        if (eta_q < eta_q_minus && order_corrector > MINORDER) {
            if (eta_q_minus < eta_q_plus && order_corrector < MAXORDER) {
                for (i = 0; i < numOfSpecies; i++)
                    for (j = order_corrector + 1; j < 7; j++)
                        z_corrector[numOfSpecies * j + i] = 0.0;

                if (numOfDerivatives > 0) {
                    for (z = 0; z < numOfDerivatives; z++)
                        for (i = 0; i < numOfSpecies; i++)
                            for (j = order_corrector + 1; j < 7; j++)
                                z_corrector_derivatives[z][numOfSpecies * j + i] = 0.0;
                }

                order_corrector += 1;
                step_size_h = step_size_h * eta_q_plus;
                if (step_size_h < 4.0 * MACHINEPRECISION * t_i[1]) {
                    YABI_LOG_ERROR("relaxing step size!");

                    step_size_h = 4.0 * MACHINEPRECISION * t_i[1];
                }
                t_i[0] = step_size_h + t_i[1];
                num_of_steps_per_order = 0.0;
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
                for (i = 2; i <= order_corrector; i++)
                    for (j = i; j > 1; j--) {
                        if (j == i) {
                            if (j != 6)
                                d_i[j] = 1.0;
                        } else {
                            // d_i[j]=d_i[j-1]+d_i[j]*stepSizeSet_new[i-3]/step_size_h;
                            y = d_i[j - 1] -
                                (d_kahan[j - 2] * stepSizeSet_new[i - 3] / step_size_h);
                            t = (d_i[j] * stepSizeSet_new[i - 3] / step_size_h) + y;
                            d_kahan[j - 2] = (t - d_i[j]) - y;
                            d_i[j] = t;
                        }
                    }

                Scalar *temp = new Scalar[numOfSpecies];
                for (i = 0; i < numOfSpecies; i++)
                    temp[i] = z_corrector[order_corrector * numOfSpecies + i];

                Scalar blasTEMP = -1.0;
                linalg::GER(numOfSpecies, order_corrector, blasTEMP, temp, blasONE, d_i, blasONE,
                            z_corrector.data(), numOfSpecies);

                if (numOfDerivatives > 0) {
                    for (z = 0; z < numOfDerivatives; z++) {
                        for (i = 0; i < numOfSpecies; i++)
                            temp[i] =
                                z_corrector_derivatives[z][order_corrector * numOfSpecies + i];
                        Scalar blasTEMP = -1.0;
                        linalg::GER(numOfSpecies, order_corrector, blasTEMP, temp, blasONE, d_i,
                                    blasONE, z_corrector_derivatives[z], numOfSpecies);
                    }
                }

                delete[] temp;
                order_corrector -= 1;
                step_size_h = step_size_h * eta_q_minus;
                if (step_size_h < 4.0 * MACHINEPRECISION * t_i[1]) {
                    YABI_LOG_ERROR("relaxing step size!");

                    step_size_h = 4.0 * MACHINEPRECISION * t_i[1];
                }
                t_i[0] = step_size_h + t_i[1];

                delete[] d_i;
                delete[] d_kahan;
                num_of_steps_per_order = 0;
                isOrderDown = true;
            }
        } else if ((eta_q < eta_q_plus || order_corrector < MINORDER) && order_corrector < MAXORDER) {
            for (i = 0; i < numOfSpecies; i++)
                for (j = order_corrector + 1; j < 7; j++)
                    z_corrector[numOfSpecies * j + i] = 0.0;

            if (numOfDerivatives > 0) {
                for (z = 0; z < numOfDerivatives; z++)
                    for (i = 0; i < numOfSpecies; i++)
                        for (j = order_corrector + 1; j < 7; j++)
                            z_corrector_derivatives[z][numOfSpecies * j + i] = 0.0;
            }

            order_corrector += 1;
            step_size_h = step_size_h * eta_q_plus;
            if (step_size_h < 4.0 * MACHINEPRECISION * t_i[1]) {
                YABI_LOG_ERROR("relaxing step size!");

                step_size_h = 4.0 * MACHINEPRECISION * t_i[1];
            }
            t_i[0] = step_size_h + t_i[1];
            num_of_steps_per_order = 0;
            isOrderUp = true;
        } else {
            if (!std::isinf(eta_q)) {
                step_size_h = step_size_h * eta_q;
                if (step_size_h < 4 * MACHINEPRECISION * t_i[1]) {
                    YABI_LOG_ERROR("relaxing step size!");

                    step_size_h = 4.0 * MACHINEPRECISION * t_i[1];
                }
                t_i[0] = step_size_h + t_i[1];
            } else {
                eta_q = max_StepLength / step_size_h;
                step_size_h = step_size_h * eta_q;
                if (step_size_h < 4.0 * MACHINEPRECISION * t_i[1]) {
                    YABI_LOG_ERROR("relaxing step size!");

                    step_size_h = 4.0 * MACHINEPRECISION * t_i[1];
                }
                t_i[0] = step_size_h + t_i[1];
            }
        }
    }
    c_n_old = c_n;
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
            tape->stepSize = step_size_h;
        }
        if (isOrderDown) {
            tape->orderUP = false;
            tape->orderDOWN = true;
            tape->stepSize = step_size_h;
        }
    }
    return true;
}

template <typename Scalar> inline bool BDFIntegrator<Scalar>::calculate_output_vector() {
    unsigned int i, j, z;

    if (internalError)
        return false;

    Scalar *temp = new Scalar[order_corrector + 1];
    for (i = 0; i <= order_corrector; i++) {
        temp[i] = 1.0;
        for (j = 0; j < i; j++)
            temp[i] *= (t_ende - t_i[1]) / step_size_h;
    }

    int blasTEMP = order_corrector + 1;
    linalg::GEMV_NT(numOfSpecies, blasTEMP, blasScalarONE, z_corrector.data(), numOfSpecies, temp, blasONE,
                    blasScalarZERO, solution_Vector.data(), blasONE);

    if (numOfDerivatives > 0)
        for (z = 0; z < numOfDerivatives; z++)
            linalg::GEMV_NT(numOfSpecies, blasTEMP, blasScalarONE, z_corrector_derivatives[z],
                            numOfSpecies, temp, blasONE, blasScalarZERO,
                            solution_Vector_derivatives[z], blasONE);

    delete[] temp;
    if (taping) {
        if (endTAPE == NULL && startTAPE != NULL)
            return false;

        ReplayTape *tape = createReplayTapeNode();
        if (getTAPE == true && intermediateTAPE == NULL)
            intermediateTAPE = endTAPE;

        tape->calculate_output = true;
        tape->stepSize = step_size_h;
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
