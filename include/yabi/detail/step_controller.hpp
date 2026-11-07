/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  Copyright (C) 2010-2012 Dominik Skanda
 *
 *  SPDX-License-Identifier: Apache-2.0
 */
#ifndef YABI_DETAIL_STEP_CONTROLLER_HPP
#define YABI_DETAIL_STEP_CONTROLLER_HPP

#include <cmath>
#include <vector>

#include <yabi/abstract_rhs.hpp>
#include <yabi/detail/integrator_config.hpp>
#include <yabi/detail/nordsieck_state.hpp>
#include <yabi/log.hpp>

namespace yabi {
namespace detail {

// Outcome of one step-size/order decision.
//
// `orderUp` and `orderDown` are mutually exclusive; the replay-tape code treats
// the (unreachable) both-true case as an inconsistency, so it is reported
// explicitly instead of being encoded in the boolean return value.
struct StepDecision {
    bool orderUp = false;
    bool orderDown = false;
};

// Step-size and order selection for the BDF integration.
//
// The controller owns no numeric state: it borrows the Nordsieck history (which
// it resizes through its step/order fields) and the integrator configuration,
// and receives the error weights, thresholds and simultaneous-derivative
// mirrors from the caller.  This keeps step selection independent of the Newton
// and sensitivity machinery while keeping every historical expression intact.
template <typename Scalar> class StepController {
  public:
    StepController(NordsieckState<Scalar> &nordsieck, const IntegratorConfig<Scalar> &config)
        : nordsieck(nordsieck), config(config) {
    }

    // Error weights for the local error estimate.  Mode 0 raises the existing
    // weights (max of weight, |y| and the absolute tolerance); mode 1 replaces
    // them with |y| + atol/rtol.
    static bool calculateErrorWeights(unsigned int numSpecies, int mode, Scalar *errorWeights,
                                      const Scalar *yWeight, const Scalar *absoluteError,
                                      Scalar relativeError) {
        unsigned int i;
        if (mode == 0) {
            for (i = 0; i < numSpecies; i++) {
                if (errorWeights[i] < fabs(yWeight[i]))
                    errorWeights[i] = fabs(yWeight[i]);
                if (errorWeights[i] < absoluteError[i])
                    errorWeights[i] = absoluteError[i];
            }
        }
        if (mode == 1) {
            for (i = 0; i < numSpecies; i++)
                errorWeights[i] = fabs(yWeight[i]) + (absoluteError[i] / relativeError);
        }
        return true;
    }

    // Initial step size from the second derivative of the solution at the
    // initial point, following the dissertation estimate.
    bool estimateStartStepsize(const AbstractRHS<Scalar> &rhs, unsigned int numSpecies,
                               unsigned int rhsType, unsigned int sizeOfParameter,
                               const std::vector<Scalar> &parameter,
                               std::vector<Scalar> &errorWeights,
                               const std::vector<Scalar> &absoluteError, Scalar relativeError,
                               Scalar maxStepLength) {
        unsigned int i, j;

        unsigned int argumentSize = 0;
        if (rhsType == 0)
            argumentSize = numSpecies;
        if (rhsType == 1)
            argumentSize = numSpecies + 1;
        if (rhsType == 2)
            argumentSize = numSpecies + 1 + sizeOfParameter;
        if (argumentSize == 0)
            return false;

        std::vector<Scalar> y_start(argumentSize);
        for (i = 0; i < numSpecies; i++)
            y_start[i] = nordsieck.z_corrector[i];
        if (rhsType >= 1)
            y_start[numSpecies] = nordsieck.t_i[1];
        if (rhsType == 2)
            for (i = 0; i < sizeOfParameter; i++)
                y_start[numSpecies + 1 + i] = parameter[i];

        std::vector<Scalar> f(numSpecies);
        f = rhs.Forward(0, y_start);

        // Jacobian F_y(y_start,t_start,p)
        std::vector<Scalar> J(numSpecies * numSpecies);
        std::vector<Scalar> dx(argumentSize);
        std::vector<Scalar> df(numSpecies);
        for (j = 0; j < argumentSize; j++)
            dx[j] = 0.0;
        for (j = 0; j < numSpecies; j++) {
            dx[j] = 1.0;
            df = rhs.Forward(1, dx);
            dx[j] = 0.0;
            for (i = 0; i < numSpecies; i++)
                J[i * numSpecies + j] = df[i];
        }

        // Second derivative of the solution:
        // y'' = F_y * F + F_t  (F_t is zero for autonomous systems).
        std::vector<Scalar> y_ddot(numSpecies);
        for (i = 0; i < numSpecies; i++) {
            y_ddot[i] = 0.0;
            for (j = 0; j < numSpecies; j++)
                y_ddot[i] += J[i * numSpecies + j] * f[j];
        }
        if (rhsType >= 1) {
            std::vector<Scalar> dx_time(argumentSize);
            for (j = 0; j < argumentSize; j++)
                dx_time[j] = 0.0;
            dx_time[numSpecies] = 1.0;
            rhs.Forward(0, y_start);
            df = rhs.Forward(1, dx_time);
            for (i = 0; i < numSpecies; i++)
                y_ddot[i] += df[i];
        }

        // CALCULATE ERROR WEIGHTS
        calculateErrorWeights(numSpecies, config.ERROR_WEIGHTS_INDEX, errorWeights.data(),
                              y_start.data(), absoluteError.data(), relativeError);

        nordsieck.step_size_h = 0.0;
        Scalar temp = 0.0;
        for (i = 0; i < numSpecies; i++) {
            temp = (y_ddot[i] / errorWeights[i]);
            nordsieck.step_size_h += temp * temp;
        }
        nordsieck.step_size_h = sqrt(nordsieck.step_size_h) / (Scalar)numSpecies;

        // If the second derivative vanishes (e.g. y_start=0 and F(y_start)=0),
        // the estimate is not defined.  Fall back to the maximum step size as
        // described in the dissertation.
        if (!(nordsieck.step_size_h > 0.0) || !std::isfinite(nordsieck.step_size_h))
            nordsieck.step_size_h = maxStepLength;
        else {
            nordsieck.step_size_h =
                sqrt((2.0 * relativeError * config.START_STEP_SIZE_SAFETY) / nordsieck.step_size_h);
            if (!(nordsieck.step_size_h > 0.0) || !std::isfinite(nordsieck.step_size_h))
                nordsieck.step_size_h = maxStepLength;
        }
        if (nordsieck.step_size_h > maxStepLength)
            nordsieck.step_size_h = maxStepLength;
        // For purely autonomous systems the start step heuristic can be overly
        // aggressive for very stiff problems (e.g. Robertson). Be more
        // conservative to resolve fast transients.
        if (rhsType == 0) {
            if (maxStepLength > 0) {
                if (nordsieck.step_size_h > maxStepLength * 1e-4)
                    nordsieck.step_size_h = maxStepLength * 1e-4;
            }
            if (!(nordsieck.step_size_h > 0.0) || !std::isfinite(nordsieck.step_size_h)) {
                if (maxStepLength > 0)
                    nordsieck.step_size_h = maxStepLength * 1e-4;
                else
                    nordsieck.step_size_h = 1e-12;
            }
        }
        return true;
    }

    // Next step size and order from the local truncation error estimates.
    // Updates the Nordsieck step/order/time state and returns whether the order
    // was raised or lowered (or neither).
    StepDecision prepareNextStep(unsigned int numSpecies, unsigned int numDerivatives,
                                 const std::vector<Scalar> &errorWeights, Scalar reachableError,
                                 const std::vector<Scalar> &reachableErrorDerivatives,
                                 std::vector<std::vector<Scalar>> &zCorrectorDerivatives,
                                 std::vector<std::vector<Scalar>> &correctorDerivatives,
                                 std::vector<std::vector<Scalar>> &correctorOldDerivatives,
                                 std::vector<std::vector<Scalar>> &errorWeightsDerivatives,
                                 Scalar minimalEta, Scalar maxStepLength, Scalar tEnd) {
        unsigned int i, j, z;
        bool isOrderUp = false;
        bool isOrderDown = false;

        // c_n
        Scalar c_n = 1.0;
        for (i = 1; i < nordsieck.order_corrector; i++)
            c_n *= nordsieck.stepSizeSet_new[i] /
                   (nordsieck.stepSizeSet_new[i] - nordsieck.step_size_h);
        c_n += 1.0;
        for (i = 1; i <= nordsieck.order_corrector; i++)
            c_n = c_n / (i + 1);

        for (i = 0; i < nordsieck.order_corrector; i++)
            c_n *= nordsieck.stepSizeSet_new[i] / nordsieck.step_size_h;

        if (nordsieck.num_of_steps_per_order < nordsieck.order_corrector + 1) {
            Scalar eta = minimalEta;

            if (config.HITSOLUTION)
                if ((eta * nordsieck.step_size_h) + nordsieck.t_i[1] > tEnd)
                    eta = (tEnd - nordsieck.t_i[1]) / nordsieck.step_size_h;

            if (eta * nordsieck.step_size_h < maxStepLength) {
                nordsieck.step_size_h = nordsieck.step_size_h * eta;
                if (nordsieck.step_size_h < 4.0 * config.MACHINEPRECISION * nordsieck.t_i[1]) {
                    YABI_LOG_ERROR("relaxing step size!");

                    nordsieck.step_size_h = 4.0 * config.MACHINEPRECISION * nordsieck.t_i[1];
                }
                nordsieck.t_i[0] = nordsieck.step_size_h + nordsieck.t_i[1];
            } else {
                eta = maxStepLength / nordsieck.step_size_h;
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
            std::vector<Scalar> temp2(numSpecies);

            if (nordsieck.order_corrector > 1) {
                temp1 = -1.0;
                for (i = 0; i < nordsieck.order_corrector - 1; i++)
                    temp1 *= (nordsieck.stepSizeSet_new[i]) / nordsieck.step_size_h;

                temp1 /=
                    (nordsieck.l_i[1] - (nordsieck.step_size_h /
                                         nordsieck.stepSizeSet_new[nordsieck.order_corrector - 1]));

                for (i = 0; i < numSpecies; i++)
                    temp2[i] = nordsieck.z_corrector[numSpecies * nordsieck.order_corrector + i] *
                               temp1 / errorWeights[i];

                normWeightedLocalTruncError = linalg::NRM2(numSpecies, temp2.data(), 1) / numSpecies;

                Scalar mu =
                    config.ORDER_DOWN_SAVETY_STEP * (reachableError / normWeightedLocalTruncError);
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

                if (numDerivatives > 0) {
                    Scalar eta_q_minus_derivatives;
                    for (z = 0; z < numDerivatives; z++) {
                        for (i = 0; i < numSpecies; i++)
                            temp2[i] =
                                zCorrectorDerivatives[z]
                                                     [numSpecies * nordsieck.order_corrector + i] *
                                temp1 / errorWeightsDerivatives[z][i];

                        normWeightedLocalTruncError =
                            linalg::NRM2(numSpecies, temp2.data(), 1) / numSpecies;

                        mu = config.ORDER_DOWN_SAVETY_STEP *
                             (reachableErrorDerivatives[z] / normWeightedLocalTruncError);
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
                temp1 = (Scalar)pow((nordsieck.step_size_h /
                                     (nordsieck.stepSizeSet_new[1] - nordsieck.step_size_h)),
                                    nordsieck.order_corrector + 1);
                temp1 *= (c_n / nordsieck.c_n_old);

                Scalar temp3 = 1.0;
                for (i = 1; i < nordsieck.order_corrector; i++)
                    temp3 *= nordsieck.stepSizeSet_new[i] /
                             (nordsieck.stepSizeSet_new[i] - nordsieck.step_size_h);
                temp3 += 1;
                temp3 =
                    -nordsieck.stepSizeSet_new[nordsieck.order_corrector] /
                    (nordsieck.step_size_h * temp3 *
                     (nordsieck.l_i[1] + (nordsieck.step_size_h /
                                          nordsieck.stepSizeSet_new[nordsieck.order_corrector])) *
                     (Scalar)(nordsieck.order_corrector + 2));

                for (i = 0; i < numSpecies; i++)
                    temp2[i] = temp3 *
                               (nordsieck.corrector[i] - (nordsieck.corrector_old[i] * temp1)) /
                               errorWeights[i];

                normWeightedLocalTruncError = linalg::NRM2(numSpecies, temp2.data(), 1) / numSpecies;

                Scalar mu =
                    config.ORDER_UP_SAVETY_STEP * (reachableError / normWeightedLocalTruncError);
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

                if (numDerivatives > 0) {
                    Scalar eta_q_plus_derivatives;
                    for (z = 0; z < numDerivatives; z++) {
                        for (i = 0; i < numSpecies; i++)
                            temp2[i] = temp3 *
                                       (correctorDerivatives[z][i] -
                                        (correctorOldDerivatives[z][i] * temp1)) /
                                       errorWeightsDerivatives[z][i];

                        normWeightedLocalTruncError =
                            linalg::NRM2(numSpecies, temp2.data(), 1) / numSpecies;

                        mu = config.ORDER_UP_SAVETY_STEP *
                             (reachableErrorDerivatives[z] / normWeightedLocalTruncError);
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
                if ((eta_q * nordsieck.step_size_h) + nordsieck.t_i[1] > tEnd)
                    eta_q = (tEnd - nordsieck.t_i[1]) / nordsieck.step_size_h;

            if (eta_q * nordsieck.step_size_h > maxStepLength)
                eta_q = maxStepLength / nordsieck.step_size_h;

            if (config.HITSOLUTION)
                if ((eta_q_plus * nordsieck.step_size_h) + nordsieck.t_i[1] > tEnd)
                    eta_q_plus = (tEnd - nordsieck.t_i[1]) / nordsieck.step_size_h;

            if (eta_q_plus * nordsieck.step_size_h > maxStepLength)
                eta_q_plus = maxStepLength / nordsieck.step_size_h;

            if (config.HITSOLUTION)
                if ((eta_q_minus * nordsieck.step_size_h) + nordsieck.t_i[1] > tEnd)
                    eta_q_minus = (tEnd - nordsieck.t_i[1]) / nordsieck.step_size_h;

            if (eta_q_minus * nordsieck.step_size_h > maxStepLength)
                eta_q_minus = maxStepLength / nordsieck.step_size_h;

            if (eta_q < eta_q_minus && nordsieck.order_corrector > config.MINORDER) {
                if (eta_q_minus < eta_q_plus && nordsieck.order_corrector < config.MAXORDER) {
                    for (i = 0; i < numSpecies; i++)
                        for (j = nordsieck.order_corrector + 1; j < 7; j++)
                            nordsieck.z_corrector[numSpecies * j + i] = 0.0;

                    if (numDerivatives > 0) {
                        for (z = 0; z < numDerivatives; z++)
                            for (i = 0; i < numSpecies; i++)
                                for (j = nordsieck.order_corrector + 1; j < 7; j++)
                                    zCorrectorDerivatives[z][numSpecies * j + i] = 0.0;
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
                    std::vector<Scalar> d_kahan(4, (Scalar)0.0);
                    Scalar y;
                    Scalar t;

                    std::vector<Scalar> d_i(6, (Scalar)0.0);
                    d_i[0] = 0.0;
                    d_i[1] = 0.0;
                    for (i = 2; i <= nordsieck.order_corrector; i++)
                        for (j = i; j > 1; j--) {
                            if (j == i) {
                                if (j != 6)
                                    d_i[j] = 1.0;
                            } else {
                                // d_i[j]=d_i[j-1]+d_i[j]*stepSizeSet_new[i-3]/step_size_h;
                                y = d_i[j - 1] -
                                    (d_kahan[j - 2] * nordsieck.stepSizeSet_new[i - 3] /
                                     nordsieck.step_size_h);
                                t = (d_i[j] * nordsieck.stepSizeSet_new[i - 3] /
                                     nordsieck.step_size_h) +
                                    y;
                                d_kahan[j - 2] = (t - d_i[j]) - y;
                                d_i[j] = t;
                            }
                        }

                    std::vector<Scalar> temp(numSpecies);
                    for (i = 0; i < numSpecies; i++)
                        temp[i] = nordsieck.z_corrector[nordsieck.order_corrector * numSpecies + i];

                    Scalar blasTEMP = -1.0;
                    linalg::GER(numSpecies, nordsieck.order_corrector, blasTEMP, temp.data(), 1,
                                d_i.data(), 1, nordsieck.z_corrector.data(), numSpecies);

                    if (numDerivatives > 0) {
                        for (z = 0; z < numDerivatives; z++) {
                            for (i = 0; i < numSpecies; i++)
                                temp[i] = zCorrectorDerivatives[z][nordsieck.order_corrector *
                                                                       numSpecies +
                                                                   i];
                            Scalar blasTEMP = -1.0;
                            linalg::GER(numSpecies, nordsieck.order_corrector, blasTEMP, temp.data(),
                                        1, d_i.data(), 1, zCorrectorDerivatives[z].data(),
                                        numSpecies);
                        }
                    }

                    nordsieck.order_corrector -= 1;
                    nordsieck.step_size_h = nordsieck.step_size_h * eta_q_minus;
                    if (nordsieck.step_size_h < 4.0 * config.MACHINEPRECISION * nordsieck.t_i[1]) {
                        YABI_LOG_ERROR("relaxing step size!");

                        nordsieck.step_size_h = 4.0 * config.MACHINEPRECISION * nordsieck.t_i[1];
                    }
                    nordsieck.t_i[0] = nordsieck.step_size_h + nordsieck.t_i[1];

                    nordsieck.num_of_steps_per_order = 0;
                    isOrderDown = true;
                }
            } else if ((eta_q < eta_q_plus || nordsieck.order_corrector < config.MINORDER) &&
                       nordsieck.order_corrector < config.MAXORDER) {
                for (i = 0; i < numSpecies; i++)
                    for (j = nordsieck.order_corrector + 1; j < 7; j++)
                        nordsieck.z_corrector[numSpecies * j + i] = 0.0;

                if (numDerivatives > 0) {
                    for (z = 0; z < numDerivatives; z++)
                        for (i = 0; i < numSpecies; i++)
                            for (j = nordsieck.order_corrector + 1; j < 7; j++)
                                zCorrectorDerivatives[z][numSpecies * j + i] = 0.0;
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
                    eta_q = maxStepLength / nordsieck.step_size_h;
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

        StepDecision decision;
        decision.orderUp = isOrderUp;
        decision.orderDown = isOrderDown;
        return decision;
    }

  private:
    NordsieckState<Scalar> &nordsieck;
    const IntegratorConfig<Scalar> &config;
};

} // namespace detail
} // namespace yabi

#endif
