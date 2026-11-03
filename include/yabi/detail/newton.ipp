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
// Newton iteration, Jacobian assembly and corrector acceptance.
//
// Generated from the reference implementation in YABI_old without changing any
// expression: the bodies below are the original algorithm verbatim.

#ifndef YABI_DETAIL_NEWTON_IPP
#define YABI_DETAIL_NEWTON_IPP

namespace yabi {

template <typename Scalar> inline bool BDFIntegrator<Scalar>::calculate_newton_scaling() {
    unsigned int i, j;
    for (j = 0; j < numOfSpecies; j++) {
        // logb() is undefined for 0 and inf; casting its result to int
        // is undefined behaviour that used to flush the whole Newton
        // matrix to 0 via scalbn(INT_MIN).  A non-positive weight means
        // atol=0 and y_j=0 (or a broken caller), which must be reported,
        // not silently turned into a wrong answer.
        if (!(error_weights[j] > 0.0) || !std::isfinite(error_weights[j])) {
            YABI_LOG_ERROR("ERROR: non-positive or non-finite error weight for species ", j, " (",
                           error_weights[j], ")!");

            internalError = true;
            return false;
        }
        scaling_D2[j] = (int)floor(logb(error_weights[j]) + 0.5);
    }

    for (i = 0; i < numOfSpecies; i++)
        for (j = 0; j < numOfSpecies; j++)
            P[j * numOfSpecies + i] = (Scalar)scalbn(P[j * numOfSpecies + i], scaling_D2[j]);

    for (i = 0; i < numOfSpecies; i++) {
        Scalar sum = 0.0;
        for (j = 0; j < numOfSpecies; j++)
            sum += fabs(P[j * numOfSpecies + i]);
        if (!(sum > 0.0) || !std::isfinite(sum)) {
            YABI_LOG_ERROR("ERROR: cannot scale column ", i, " of the Newton matrix (column sum ",
                           sum, ")!");

            internalError = true;
            return false;
        }
        scaling_D1[i] = -(int)ceil(logb(sum));
    }

    for (i = 0; i < numOfSpecies; i++)
        for (j = 0; j < numOfSpecies; j++)
            P[j * numOfSpecies + i] = (Scalar)scalbn(P[j * numOfSpecies + i], scaling_D1[i]);

    return true;
}

template <typename Scalar> bool BDFIntegrator<Scalar>::calculate_corrector_vector() {
    Scalar minimalRate = 0.0;
    unsigned int i, j, k, l, z;
    bool wasJacobianSet = false;
    bool wasNewPSet = false;
    bool contructionRateFulfilled = false;

    int sign;

    Scalar error_multiplier = 1.0;
    for (i = 1; i < order_corrector; i++)
        error_multiplier *= stepSizeSet_new[i] / (stepSizeSet_new[i] - step_size_h);
    error_multiplier += 1.0;
    // Local-error multiplier 1/alpha_0 * 1/(1 - 1/alpha_tilde_0)
    // with alpha_0 = -l_i[1]: the magnitude is l_i[1] * (1 + product).
    error_multiplier *= l_i[1];
    error_multiplier = fabs(error_multiplier);

    // Compute Matrix P

    unsigned int cppAD_argument_size = 0;

    if (RHS_type == 0)
        cppAD_argument_size = numOfSpecies;
    if (RHS_type == 1)
        cppAD_argument_size = numOfSpecies + 1;
    if (RHS_type == 2)
        cppAD_argument_size = numOfSpecies + 1 + sizeOfParameter;

    std::vector<Scalar> cppAD_argument(cppAD_argument_size);
    Scalar *root = new Scalar[numOfSpecies];

    for (i = 0; i < numOfSpecies; i++)
        root[i] = z_predictor[i];

    calculate_error_weights(error_weights, root, absolut_error, relative_error);

    Scalar *b = new Scalar[numOfSpecies];
    Scalar *temp = new Scalar[numOfSpecies];

    Scalar **b_derivatives = NULL;
    Scalar **root_derivatives = NULL;
    Scalar **convergenceResidual_derivatives = NULL;
    if (numOfDerivatives > 0 && !direct) {
        root_derivatives = new Scalar *[numOfDerivatives];
        b_derivatives = new Scalar *[numOfDerivatives];
        convergenceResidual_derivatives = new Scalar *[numOfDerivatives];
        for (i = 0; i < numOfDerivatives; i++) {
            convergenceResidual_derivatives[i] = new Scalar[2];
            convergenceResidual_derivatives[i][0] = 0.0;
            convergenceResidual_derivatives[i][1] = 0.0;
            root_derivatives[i] = new Scalar[numOfSpecies];
            b_derivatives[i] = new Scalar[numOfSpecies];
            for (j = 0; j < numOfSpecies; j++) {
                root_derivatives[i][j] = z_predictor_derivatives[i][j];
                corrector_derivatives[i][j] = 0.0;
            }
            calculate_error_weights(error_weights_derivatives[i], z_predictor_derivatives[i],
                                    absolut_error_derivatives[i], relative_error_derivatives[i]);
            // Same guard as the main path (T09): a zero/non-finite weight makes the later
            // logb(error_weights_derivatives[z][i]) undefined (flushes the sensitivity Newton
            // matrix via scalbn(INT_MIN)). Report it instead of silently returning a wrong answer.
            for (j = 0; j < numOfSpecies; j++)
                if (!(error_weights_derivatives[i][j] > 0.0) ||
                    !std::isfinite(error_weights_derivatives[i][j])) {
                    YABI_LOG_ERROR("ERROR: non-positive or non-finite derivative error weight for direction ",
                                   i, ", species ", j, " (", error_weights_derivatives[i][j], ")!");

                    internalError = true;
                    return false;
                }
        }
    }

    std::vector<Scalar> rhs(numOfSpecies);

    if (!newtonIteration_Started) {
        setJacobian = true;
        newtonIteration_Started = true;
    }

    unsigned int newtonIterate = 1;

    Scalar convergenceResidual[2];
    convergenceResidual[0] = 0.0;
    convergenceResidual[1] = 0.0;

    for (i = 0; i < numOfSpecies; i++)
        corrector[i] = 0.0;

    Scalar *tempMACHINEprecision = new Scalar[numOfSpecies];

    // Release this function's workspace on a hard internal failure.  The
    // caller aborts the integration if any of the guards below fires; the
    // alternative (continuing with a singular Newton matrix) used to return
    // a wrong answer as success.
    auto releaseWorkspace = [&]() {
        delete[] root;
        delete[] b;
        delete[] temp;
        delete[] tempMACHINEprecision;
        if (numOfDerivatives > 0 && !direct) {
            for (l = 0; l < numOfDerivatives; l++) {
                delete[] root_derivatives[l];
                delete[] b_derivatives[l];
                delete[] convergenceResidual_derivatives[l];
            }
            delete[] b_derivatives;
            delete[] root_derivatives;
            delete[] convergenceResidual_derivatives;
        }
    };

    while (newtonIterate <= MAXNEWTONITERATION) {
        newtonIterations++;
        if (setJacobian && !wasJacobianSet) {
            updateJacobian++;
            wasJacobianSet = true;

            for (i = 0; i < numOfSpecies; i++)
                cppAD_argument[i] = root[i];
            if (RHS_type >= 1)
                cppAD_argument[numOfSpecies] = t_i[0];
            if (RHS_type == 2)
                for (i = 0; i < sizeOfParameter; i++)
                    cppAD_argument[numOfSpecies + 1 + i] = parameter[i];
            RHS->Forward(0, cppAD_argument);

            std::vector<Scalar> dx(cppAD_argument_size);
            std::vector<Scalar> df(numOfSpecies);
            for (j = 0; j < cppAD_argument_size; j++)
                dx[j] = 0.0;
            for (j = 0; j < numOfSpecies; j++) {
                dx[j] = 1.0;
                df = RHS->Forward(1, dx);
                for (i = 0; i < numOfSpecies; i++) {
                    if (std::isnan(df[i]) || std::isinf(df[i])) {
                        YABI_LOG_ERROR("element of jacobian is nan or inf!");

                        internalError = true;
                        releaseWorkspace();
                        return false;
                    }
                    J[i * numOfSpecies + j] = df[i];
                }
                dx[j] = 0.0;
            }

            for (j = 0; j < numOfSpecies; j++)
                for (k = 0; k < numOfSpecies; k++)
                    if (j == k)
                        P[k * numOfSpecies + j] =
                            1.0 - ((step_size_h / l_i[1]) * J[j * numOfSpecies + k]);
                    else
                        P[k * numOfSpecies + j] = -(step_size_h / l_i[1]) * J[j * numOfSpecies + k];

            // LU factor (and overwrite) the matrix P and ip and jp via pluggable solver
            if (!calculate_newton_scaling()) {
                YABI_LOG_ERROR("Newton scaling failed (invalid error weight)!");

                internalError = true;
                releaseWorkspace();
                return false;
            }
            if (linearSolver) {
                if (!linearSolver->initialize(numOfSpecies)) {
                    internalError = true;
                    releaseWorkspace();
                    return false;
                }
                if (!linearSolver->factorize(P, ip, scaling_D1, scaling_D2, numOfSpecies)) {
                    YABI_LOG_ERROR("ERROR: LuFactor Failed!");
                    internalError = true;
                    releaseWorkspace();
                    return false;
                }
            } else {
                sign = linalg::GETRF(numOfSpecies, numOfSpecies, P, numOfSpecies, ip);
                if (sign != 0) {
                    YABI_LOG_ERROR("ERROR: LuFactor Failed! (dgetrf info=", sign, ")");
                    internalError = true;
                    releaseWorkspace();
                    return false;
                }
            }
            // Sensitivity Generation

            if (taping) {
                if (!direct) {
                    if (endTAPE->P == NULL)
                        endTAPE->P = new Scalar[numOfSpecies * numOfSpecies];
                    if (endTAPE->ip == NULL)
                        endTAPE->ip = new int[numOfSpecies];
                    if (endTAPE->scaling_D1 == NULL)
                        endTAPE->scaling_D1 = new long[numOfSpecies];
                    if (endTAPE->scaling_D2 == NULL)
                        endTAPE->scaling_D2 = new long[numOfSpecies];

                    for (i = 0; i < numOfSpecies; i++) {
                        endTAPE->scaling_D1[i] = scaling_D1[i];
                        endTAPE->scaling_D2[i] = scaling_D2[i];
                        endTAPE->ip[i] = ip[i];
                        for (j = 0; j < numOfSpecies; j++)
                            endTAPE->P[i * numOfSpecies + j] = P[i * numOfSpecies + j];
                    }
                }
            }
        }
        if (setNewP == true && setJacobian == false && !wasNewPSet) {
            updateP++;
            wasNewPSet = true;
            for (j = 0; j < numOfSpecies; j++)
                for (k = 0; k < numOfSpecies; k++)
                    if (j == k)
                        P[k * numOfSpecies + j] =
                            1.0 - ((step_size_h / l_i[1]) * J[j * numOfSpecies + k]);
                    else
                        P[k * numOfSpecies + j] = -(step_size_h / l_i[1]) * J[j * numOfSpecies + k];

            // LU factor (and overwrite) the matrix P via pluggable solver
            if (!calculate_newton_scaling()) {
                YABI_LOG_ERROR("Newton scaling failed (invalid error weight)!");

                internalError = true;
                releaseWorkspace();
                return false;
            }
            if (linearSolver) {
                if (!linearSolver->initialize(numOfSpecies)) {
                    internalError = true;
                    releaseWorkspace();
                    return false;
                }
                if (!linearSolver->factorize(P, ip, scaling_D1, scaling_D2, numOfSpecies)) {
                    YABI_LOG_ERROR("ERROR: LuFactor Failed!");
                    internalError = true;
                    releaseWorkspace();
                    return false;
                }
            } else {
                sign = linalg::GETRF(numOfSpecies, numOfSpecies, P, numOfSpecies, ip);
                if (sign != 0) {
                    YABI_LOG_ERROR("ERROR: LuFactor Failed! (dgetrf info=", sign, ")");
                    internalError = true;
                    releaseWorkspace();
                    return false;
                }
            }
            // Sensitivity Generation

            if (taping) {
                if (!direct) {
                    if (endTAPE->P == NULL)
                        endTAPE->P = new Scalar[numOfSpecies * numOfSpecies];
                    if (endTAPE->ip == NULL)
                        endTAPE->ip = new int[numOfSpecies];
                    if (endTAPE->scaling_D1 == NULL)
                        endTAPE->scaling_D1 = new long[numOfSpecies];
                    if (endTAPE->scaling_D2 == NULL)
                        endTAPE->scaling_D2 = new long[numOfSpecies];

                    for (i = 0; i < numOfSpecies; i++) {
                        endTAPE->scaling_D1[i] = scaling_D1[i];
                        endTAPE->scaling_D2[i] = scaling_D2[i];
                        endTAPE->ip[i] = ip[i];
                        for (j = 0; j < numOfSpecies; j++)
                            endTAPE->P[i * numOfSpecies + j] = P[i * numOfSpecies + j];
                    }
                }
            }
        }

        for (j = 0; j < numOfSpecies; j++)
            cppAD_argument[j] = root[j];
        if (RHS_type >= 1)
            cppAD_argument[numOfSpecies] = t_i[0];
        if (RHS_type == 2)
            for (j = 0; j < sizeOfParameter; j++)
                cppAD_argument[numOfSpecies + 1 + j] = parameter[j];

        if (taping && !direct) {
            if (endTAPE->CppAD_tape[newtonIterate - 1] == NULL) {
                if (intermediateCppADFunIterator == cppADFunList.end()) {
                    std::unique_ptr<AbstractRHS<Scalar>> tempADFun(RHS->clone());
    tempADFun->capacity_order(MINTAYLOR);
    cppADFunList.push_back(std::move(tempADFun));
                    intermediateCppADFunIterator--;
                }
                endTAPE->CppAD_tape[newtonIterate - 1] = intermediateCppADFunIterator->get();
                intermediateCppADFunIterator++;
            }
            rhs = endTAPE->CppAD_tape[newtonIterate - 1]->Forward(0, cppAD_argument);
        } else
            rhs = RHS->Forward(0, cppAD_argument);

        for (j = 0; j < numOfSpecies; j++) {
            temp[j] = rhs[j];
            b[j] = z_predictor[numOfSpecies + j];
        }
        Scalar blasTEMP1 = 1.0 / l_i[1];
        Scalar blasTEMP2 = -step_size_h / l_i[1];

        linalg::SCAL(numOfSpecies, blasTEMP1, b, blasONE);
        linalg::AXPY(numOfSpecies, blasTEMP2, temp, blasONE, b, blasONE);
        linalg::AXPY(numOfSpecies, blasScalarONE, root, blasONE, b, blasONE);
        linalg::AXPY(numOfSpecies, blasScalarMinusONE, z_predictor, blasONE, b, blasONE);
        for (j = 0; j < numOfSpecies; j++)
            b[j] = -(Scalar)scalbn(b[j], scaling_D1[j]);

        if (linearSolver) {
            sign = linearSolver->solve(P, ip, scaling_D1, scaling_D2, b, numOfSpecies, 1);
        } else {
            if (linearSolver) {
                sign = linearSolver->solve(P, ip, scaling_D1, scaling_D2, b, numOfSpecies, 1);
            } else {
                sign = linalg::GETRS_NT(numOfSpecies, blasONE, P, numOfSpecies, ip, b, numOfSpecies);
            }
        }

        for (j = 0; j < numOfSpecies; j++)
            temp[j] = (Scalar)scalbn(b[j], scaling_D2[j]);

        linalg::AXPY(numOfSpecies, blasScalarONE, temp, blasONE, root, blasONE);
        linalg::AXPY(numOfSpecies, blasScalarONE, temp, blasONE, corrector, blasONE);

        if (numOfDerivatives > 0 && !direct) {
            for (z = 0; z < numOfDerivatives; z++) {
                for (j = 0; j < numOfSpecies; j++)
                    cppAD_argument[j] = root_derivatives[z][j];
                if (RHS_type >= 1)
                    cppAD_argument[numOfSpecies] = 0.0;
                if (RHS_type == 2)
                    for (j = 0; j < sizeOfParameter; j++)
                        cppAD_argument[numOfSpecies + 1 + j] = p_seed_derivatives[z][j];

                if (taping && !direct)
                    rhs = endTAPE->CppAD_tape[newtonIterate - 1]->Forward(1, cppAD_argument);
                else
                    rhs = RHS->Forward(1, cppAD_argument);

                for (j = 0; j < numOfSpecies; j++) {
                    temp[j] = rhs[j];
                    b_derivatives[z][j] = z_predictor_derivatives[z][numOfSpecies + j];
                }

                linalg::SCAL(numOfSpecies, blasTEMP1, b_derivatives[z], blasONE);
                linalg::AXPY(numOfSpecies, blasTEMP2, temp, blasONE, b_derivatives[z], blasONE);
                linalg::AXPY(numOfSpecies, blasScalarONE, root_derivatives[z], blasONE,
                             b_derivatives[z], blasONE);
                linalg::AXPY(numOfSpecies, blasScalarMinusONE, z_predictor_derivatives[z], blasONE,
                             b_derivatives[z], blasONE);
                for (j = 0; j < numOfSpecies; j++)
                    b_derivatives[z][j] = -(Scalar)scalbn(b_derivatives[z][j], scaling_D1[j]);

                sign = linearSolver->solve(P, ip, scaling_D1, scaling_D2, b_derivatives[z], numOfSpecies, 1);

                for (j = 0; j < numOfSpecies; j++)
                    temp[j] = (Scalar)scalbn(b_derivatives[z][j], scaling_D2[j]);

                linalg::AXPY(numOfSpecies, blasScalarONE, temp, blasONE, root_derivatives[z],
                             blasONE);
                linalg::AXPY(numOfSpecies, blasScalarONE, temp, blasONE, corrector_derivatives[z],
                             blasONE);
            }
        }

        if (newtonIterate == 1) {
            convergenceResidual[0] = linalg::NRM2(numOfSpecies, b, blasONE) / numOfSpecies;

            minimalRate = (Scalar)pow((error_multiplier * NEWTON_CONVERGED * relative_error) /
                                          convergenceResidual[0],
                                      1.0 / (MAXNEWTONITERATION - 1));

            if (numOfDerivatives > 0 && !direct) {
                for (z = 0; z < numOfDerivatives; z++) {
                    for (i = 0; i < numOfSpecies; i++)
                        temp[i] = (Scalar)scalbn(
                            scalbn(b_derivatives[z][i], scaling_D2[i]),
                            -(int)floor(logb(error_weights_derivatives[z][i]) + 0.5));

                    convergenceResidual_derivatives[z][0] =
                        linalg::NRM2(numOfSpecies, temp, blasONE) / numOfSpecies;

                    Scalar tempRATE = 0;
                    tempRATE = (Scalar)pow(
                        (error_multiplier * NEWTON_CONVERGED * relative_error_derivatives[z]) /
                            convergenceResidual_derivatives[z][0],
                        1.0 / (MAXNEWTONITERATION - 1));
                    if (tempRATE < minimalRate)
                        minimalRate = tempRATE;
                }
            }
            if (NEWTON_SAFETY > 0)
                minimalRate = NEWTON_SAFETY;
            bool pass = true;
            if (numOfDerivatives > 0 && !direct)
                for (z = 0; z < numOfDerivatives; z++) {
                    for (j = 0; j < numOfSpecies; j++)
                        tempMACHINEprecision[j] = (Scalar)scalbn(
                            root_derivatives[z][j],
                            -(int)floor(logb(error_weights_derivatives[z][j]) + 0.5));

                    Scalar tempMACHINEprecisionNorm =
                        linalg::NRM2(numOfSpecies, tempMACHINEprecision, blasONE) / numOfSpecies;

                    if (!(convergenceResidual_derivatives[z][0] <
                              error_multiplier * NEWTON_CONVERGED * relative_error_derivatives[z] ||
                          convergenceResidual_derivatives[z][0] <
                              2.0 * MACHINEPRECISION * tempMACHINEprecisionNorm))
                        pass = false;
                }

            for (j = 0; j < numOfSpecies; j++)
                tempMACHINEprecision[j] = (Scalar)scalbn(root[j], -scaling_D2[j]);

            Scalar tempMACHINEprecisionNorm =
                linalg::NRM2(numOfSpecies, tempMACHINEprecision, blasONE) / numOfSpecies;

            if (pass == true)
                if (convergenceResidual[0] < error_multiplier * NEWTON_CONVERGED * relative_error ||
                    convergenceResidual[0] < 2.0 * MACHINEPRECISION * tempMACHINEprecisionNorm) {
                    convergenceIndicator = (Scalar)0.1;
                    if (taping) {
                        if (direct) {
                            if (endTAPE->endSolution == NULL)
                                endTAPE->endSolution = new Scalar[numOfSpecies];
                            for (i = 0; i < numOfSpecies; i++)
                                endTAPE->endSolution[i] = root[i];

                            endTAPE->timePoint = t_i[0];
                        } else
                            endTAPE->NewtonIterates = newtonIterate;
                    }
                    if (numOfDerivatives > 0 && !direct) {
                        for (l = 0; l < numOfDerivatives; l++) {
                            delete[] root_derivatives[l];
                            delete[] b_derivatives[l];
                            delete[] convergenceResidual_derivatives[l];
                        }
                        delete[] b_derivatives;
                        delete[] root_derivatives;
                        delete[] convergenceResidual_derivatives;
                    }
                    // DIRECT SIMULTANEOUS MODE

                    if (numOfDerivatives > 0 && direct) {
                        std::vector<Scalar> df(numOfSpecies);
                        Scalar *P_ = new Scalar[numOfSpecies * numOfSpecies];
                        int *ip_ = new int[numOfSpecies];

                        long *scaling_D1_ = new long[numOfSpecies];
                        long *scaling_D2_ = new long[numOfSpecies];

                        // Free only this block's matrices; releaseWorkspace() handles the
                        // rest of the function's workspace on a hard failure.
                        auto releaseDirect = [&]() {
                            delete[] P_;
                            delete[] ip_;
                            delete[] scaling_D1_;
                            delete[] scaling_D2_;
                        };

                        // CALCULATION OF JACOBIAN

                        for (j = 0; j < numOfSpecies; j++)
                            cppAD_argument[j] = root[j];
                        if (RHS_type >= 1)
                            cppAD_argument[numOfSpecies] = t_i[0];
                        if (RHS_type == 2)
                            for (j = 0; j < sizeOfParameter; j++)
                                cppAD_argument[numOfSpecies + 1 + j] = parameter[j];

                        std::vector<Scalar> J_(numOfSpecies * numOfSpecies);

                        if (taping) {
                            if (endTAPE->direct_fun == NULL) {
                                if (intermediateCppADFunIterator == cppADFunList.end()) {
                                    std::unique_ptr<AbstractRHS<Scalar>> tempADFun(RHS->clone());
    tempADFun->capacity_order(MINTAYLOR);
    cppADFunList.push_back(std::move(tempADFun));
                                    intermediateCppADFunIterator--;
                                }
                                endTAPE->direct_fun = intermediateCppADFunIterator->get();
                                intermediateCppADFunIterator++;
                            }

                            endTAPE->direct_fun->Forward(0, cppAD_argument);
                            for (j = 0; j < cppAD_argument_size; j++)
                                cppAD_argument[j] = 0.0;

                            for (j = 0; j < numOfSpecies; j++) {
                                cppAD_argument[j] = 1.0;
                                df = endTAPE->direct_fun->Forward(1, cppAD_argument);
                                for (i = 0; i < numOfSpecies; i++) {
                                    if (std::isnan(df[i]) || std::isinf(df[i])) {
                                        YABI_LOG_ERROR("element of jacobian is nan or inf! (direct "
                                                       "sensitivities simultaneous1)");

                                        internalError = true;
                                        releaseDirect();
                                        releaseWorkspace();
                                        return false;
                                    }
                                    J_[i * numOfSpecies + j] = df[i];
                                }
                                cppAD_argument[j] = 0.0;
                            }
                        } else {
                            RHS->Forward(0, cppAD_argument);
                            for (j = 0; j < cppAD_argument_size; j++)
                                cppAD_argument[j] = 0.0;

                            for (j = 0; j < numOfSpecies; j++) {
                                cppAD_argument[j] = 1.0;
                                df = RHS->Forward(1, cppAD_argument);
                                for (i = 0; i < numOfSpecies; i++) {
                                    if (std::isnan(df[i]) || std::isinf(df[i])) {
                                        YABI_LOG_ERROR("element of jacobian is nan or inf! (direct "
                                                       "sensitivities)");

                                        internalError = true;
                                        releaseDirect();
                                        releaseWorkspace();
                                        return false;
                                    }
                                    J_[i * numOfSpecies + j] = df[i];
                                }
                                cppAD_argument[j] = 0.0;
                            }
                        }
                        for (j = 0; j < numOfSpecies; j++)
                            for (k = 0; k < numOfSpecies; k++) {
                                if (j == k)
                                    P_[k * numOfSpecies + j] =
                                        1 - ((step_size_h / l_i[1]) * J_[j * numOfSpecies + k]);
                                else
                                    P_[k * numOfSpecies + j] =
                                        -(step_size_h / l_i[1]) * J_[j * numOfSpecies + k];
                            }

                        for (j = 0; j < numOfSpecies; j++)
                            scaling_D2_[j] = 0;

                        if (taping) {
                            if (endTAPE->scaling_D1 == NULL)
                                endTAPE->scaling_D1 = new long[numOfSpecies];
                            if (endTAPE->scaling_D2 == NULL)
                                endTAPE->scaling_D2 = new long[numOfSpecies];

                            for (j = 0; j < numOfSpecies; j++)
                                endTAPE->scaling_D2[j] = scaling_D2_[j];
                        }

                        for (i = 0; i < numOfSpecies; i++) {
                            Scalar sum = 0.0;
                            for (j = 0; j < numOfSpecies; j++)
                                sum += fabs(P_[j * numOfSpecies + i]);
                            if (!(sum > 0.0) || !std::isfinite(sum)) {
                                YABI_LOG_ERROR(
                                    "ERROR: cannot scale column ", i,
                                    " of the Newton matrix (direct sensitivities, column sum ", sum,
                                    ")!");

                                internalError = true;
                                releaseDirect();
                                releaseWorkspace();
                                return false;
                            }
                            scaling_D1_[i] = (int)ceil(logb(sum));
                        }

                        if (taping) {
                            for (j = 0; j < numOfSpecies; j++)
                                endTAPE->scaling_D1[j] = scaling_D1_[j];
                        }

                        for (i = 0; i < numOfSpecies; i++)
                            for (j = 0; j < numOfSpecies; j++)
                                P_[j * numOfSpecies + i] =
                                    (Scalar)scalbn(P_[j * numOfSpecies + i], scaling_D1_[i]);

                        // LU factor (and overwrite) the matrix P and ip and jp
                        int info;
                        info = linalg::GETRF(numOfSpecies, numOfSpecies, P_, numOfSpecies, ip_);
                        if (info != 0) {
                            YABI_LOG_ERROR("ERROR: LuFactor Failed! (dgetrf info=", info, ")");
                            internalError = true;
                            releaseDirect();
                            releaseWorkspace();
                            return false;
                        }

                        if (taping) {
                            if (endTAPE->P == NULL)
                                endTAPE->P = new Scalar[numOfSpecies * numOfSpecies];
                            if (endTAPE->ip == NULL)
                                endTAPE->ip = new int[numOfSpecies];

                            for (i = 0; i < numOfSpecies; i++) {
                                endTAPE->ip[i] = ip_[i];
                                for (j = 0; j < numOfSpecies; j++)
                                    endTAPE->P[i * numOfSpecies + j] = P_[i * numOfSpecies + j];
                            }
                        }

                        for (z = 0; z < numOfDerivatives; z++) {
                            for (j = 0; j < numOfSpecies; j++)
                                cppAD_argument[j] = 0.0;

                            if (RHS_type >= 1)
                                cppAD_argument[numOfSpecies] = 0.0;

                            if (RHS_type == 2) {
                                for (j = 0; j < sizeOfParameter; j++)
                                    cppAD_argument[numOfSpecies + 1 + j] = p_seed_derivatives[z][j];
                            }

                            if (taping)
                                df = endTAPE->direct_fun->Forward(1, cppAD_argument);
                            else
                                df = RHS->Forward(1, cppAD_argument);

                            for (j = 0; j < numOfSpecies; j++)
                                b[j] = -(Scalar)scalbn(
                                    ((-z_predictor_derivatives[z][j]) -
                                     (((step_size_h / l_i[1]) * df[j]) -
                                      (z_predictor_derivatives[z][numOfSpecies + j] / l_i[1]))),
                                    scaling_D1_[j]);

                            linalg::GETRS_NT(numOfSpecies, blasONE, P_, numOfSpecies, ip_, b,
                                             numOfSpecies);

                            for (i = 0; i < numOfSpecies; i++)
                                corrector_derivatives[z][i] = (Scalar)scalbn(b[i], scaling_D2_[i]) -
                                                              z_predictor_derivatives[z][i];
                        }

                        delete[] P_;
                        delete[] ip_;

                        delete[] scaling_D1_;
                        delete[] scaling_D2_;
                    }

                    delete[] root;
                    delete[] b;
                    delete[] temp;
                    delete[] tempMACHINEprecision;
                    return true;
                }
        } else {
            convergenceResidual[1] = convergenceResidual[0];
            convergenceResidual[0] = linalg::NRM2(numOfSpecies, b, blasONE) / numOfSpecies;

            if (numOfDerivatives > 0 && !direct) {
                for (z = 0; z < numOfDerivatives; z++) {
                    convergenceResidual_derivatives[z][1] = convergenceResidual_derivatives[z][0];

                    for (i = 0; i < numOfSpecies; i++)
                        temp[i] = (Scalar)scalbn(
                            scalbn(b_derivatives[z][i], scaling_D2[i]),
                            -(int)floor(logb(error_weights_derivatives[z][i]) + 0.5));
                    convergenceResidual_derivatives[z][0] =
                        linalg::NRM2(numOfSpecies, temp, blasONE) / numOfSpecies;
                }
            }
        }
        if (newtonIterate > 1) {
            Scalar ratio = convergenceResidual[0] / convergenceResidual[1];

            if (numOfDerivatives > 0 && !direct)
                for (z = 0; z < numOfDerivatives; z++)
                    if (ratio < convergenceResidual_derivatives[z][0] /
                                    convergenceResidual_derivatives[z][1])
                        ratio = convergenceResidual_derivatives[z][0] /
                                convergenceResidual_derivatives[z][1];

            bool pass = true;
            if (numOfDerivatives > 0 && !direct)
                for (z = 0; z < numOfDerivatives; z++) {
                    for (j = 0; j < numOfSpecies; j++)
                        tempMACHINEprecision[j] = (Scalar)scalbn(
                            root_derivatives[z][j],
                            -(int)floor(logb(error_weights_derivatives[z][j]) + 0.5));

                    Scalar tempMACHINEprecisionNorm =
                        linalg::NRM2(numOfSpecies, tempMACHINEprecision, blasONE) / numOfSpecies;

                    if (!(ratio * convergenceResidual_derivatives[z][0] <
                              error_multiplier * NEWTON_CONVERGED * relative_error_derivatives[z] ||
                          convergenceResidual_derivatives[z][0] <
                              error_multiplier * NEWTON_CONVERGED * relative_error_derivatives[z] ||
                          (ratio < ACCEPT_NEWTON_CONVERGENCE_RATE && contructionRateFulfilled) ||
                          convergenceResidual_derivatives[z][0] <
                              2.0 * MACHINEPRECISION * tempMACHINEprecisionNorm))
                        pass = false;
                }

            for (j = 0; j < numOfSpecies; j++)
                tempMACHINEprecision[j] = (Scalar)scalbn(root[j], -scaling_D2[j]);
            Scalar tempMACHINEprecisionNorm =
                linalg::NRM2(numOfSpecies, tempMACHINEprecision, blasONE) / numOfSpecies;

            if (pass == true)
                if (ratio * convergenceResidual[0] <
                        error_multiplier * NEWTON_CONVERGED * relative_error ||
                    convergenceResidual[0] < error_multiplier * NEWTON_CONVERGED * relative_error ||
                    (ratio < ACCEPT_NEWTON_CONVERGENCE_RATE && contructionRateFulfilled) ||
                    convergenceResidual[0] < 2.0 * MACHINEPRECISION * tempMACHINEprecisionNorm) {
                    convergenceIndicator = (Scalar)0.1;
                    if (taping) {
                        if (direct) {
                            if (endTAPE->endSolution == NULL)
                                endTAPE->endSolution = new Scalar[numOfSpecies];
                            for (i = 0; i < numOfSpecies; i++)
                                endTAPE->endSolution[i] = root[i];

                            endTAPE->timePoint = t_i[0];
                        } else
                            endTAPE->NewtonIterates = newtonIterate;
                    }

                    if (numOfDerivatives > 0 && !direct) {
                        for (l = 0; l < numOfDerivatives; l++) {
                            delete[] root_derivatives[l];
                            delete[] b_derivatives[l];
                            delete[] convergenceResidual_derivatives[l];
                        }
                        delete[] b_derivatives;
                        delete[] root_derivatives;
                        delete[] convergenceResidual_derivatives;
                    }

                    if (numOfDerivatives > 0 && direct) {
                        std::vector<Scalar> df(numOfSpecies);
                        Scalar *P_ = new Scalar[numOfSpecies * numOfSpecies];
                        int *ip_ = new int[numOfSpecies];

                        long *scaling_D1_ = new long[numOfSpecies];
                        long *scaling_D2_ = new long[numOfSpecies];

                        // Free only this block's matrices; releaseWorkspace() handles the
                        // rest of the function's workspace on a hard failure.
                        auto releaseDirect = [&]() {
                            delete[] P_;
                            delete[] ip_;
                            delete[] scaling_D1_;
                            delete[] scaling_D2_;
                        };

                        // CALCULATION OF JACOBIAN

                        for (j = 0; j < numOfSpecies; j++)
                            cppAD_argument[j] = root[j];
                        if (RHS_type >= 1)
                            cppAD_argument[numOfSpecies] = t_i[0];
                        if (RHS_type == 2)
                            for (j = 0; j < sizeOfParameter; j++)
                                cppAD_argument[numOfSpecies + 1 + j] = parameter[j];

                        std::vector<Scalar> J_(numOfSpecies * numOfSpecies);

                        if (taping) {
                            if (endTAPE->direct_fun == NULL) {
                                if (intermediateCppADFunIterator == cppADFunList.end()) {
                                    std::unique_ptr<AbstractRHS<Scalar>> tempADFun(RHS->clone());
    tempADFun->capacity_order(MINTAYLOR);
    cppADFunList.push_back(std::move(tempADFun));
                                    intermediateCppADFunIterator--;
                                }
                                endTAPE->direct_fun = intermediateCppADFunIterator->get();
                                intermediateCppADFunIterator++;
                            }

                            endTAPE->direct_fun->Forward(0, cppAD_argument);
                            for (j = 0; j < cppAD_argument_size; j++)
                                cppAD_argument[j] = 0.0;

                            for (j = 0; j < numOfSpecies; j++) {
                                cppAD_argument[j] = 1.0;
                                df = endTAPE->direct_fun->Forward(1, cppAD_argument);
                                for (i = 0; i < numOfSpecies; i++) {
                                    if (std::isnan(df[i]) || std::isinf(df[i])) {
                                        YABI_LOG_ERROR("element of jacobian is nan or inf! (direct "
                                                       "sensitivities simultaneous2)");

                                        internalError = true;
                                        releaseDirect();
                                        releaseWorkspace();
                                        return false;
                                    }
                                    J_[i * numOfSpecies + j] = df[i];
                                }
                                cppAD_argument[j] = 0.0;
                            }
                        } else {
                            RHS->Forward(0, cppAD_argument);
                            for (j = 0; j < cppAD_argument_size; j++)
                                cppAD_argument[j] = 0.0;

                            for (j = 0; j < numOfSpecies; j++) {
                                cppAD_argument[j] = 1.;
                                df = RHS->Forward(1, cppAD_argument);
                                for (i = 0; i < numOfSpecies; i++) {
                                    if (std::isnan(df[i]) || std::isinf(df[i])) {
                                        YABI_LOG_ERROR("element of jacobian is nan or inf! (direct "
                                                       "sensitivities)");

                                        internalError = true;
                                        releaseDirect();
                                        releaseWorkspace();
                                        return false;
                                    }
                                    J_[i * numOfSpecies + j] = df[i];
                                }
                                cppAD_argument[j] = 0.0;
                            }
                        }
                        for (j = 0; j < numOfSpecies; j++)
                            for (k = 0; k < numOfSpecies; k++) {
                                if (j == k)
                                    P_[k * numOfSpecies + j] =
                                        1.0 - ((step_size_h / l_i[1]) * J_[j * numOfSpecies + k]);
                                else
                                    P_[k * numOfSpecies + j] =
                                        -(step_size_h / l_i[1]) * J_[j * numOfSpecies + k];
                            }

                        for (j = 0; j < numOfSpecies; j++)
                            scaling_D2_[j] = 0;

                        if (taping) {
                            if (endTAPE->scaling_D1 == NULL)
                                endTAPE->scaling_D1 = new long[numOfSpecies];
                            if (endTAPE->scaling_D2 == NULL)
                                endTAPE->scaling_D2 = new long[numOfSpecies];

                            for (j = 0; j < numOfSpecies; j++)
                                endTAPE->scaling_D2[j] = scaling_D2_[j];
                        }

                        for (i = 0; i < numOfSpecies; i++) {
                            Scalar sum = 0.0;
                            for (j = 0; j < numOfSpecies; j++)
                                sum += fabs(P_[j * numOfSpecies + i]);
                            if (!(sum > 0.0) || !std::isfinite(sum)) {
                                YABI_LOG_ERROR(
                                    "ERROR: cannot scale column ", i,
                                    " of the Newton matrix (direct sensitivities, column sum ", sum,
                                    ")!");

                                internalError = true;
                                releaseDirect();
                                releaseWorkspace();
                                return false;
                            }
                            scaling_D1_[i] = (int)ceil(logb(sum));
                        }

                        if (taping) {
                            for (j = 0; j < numOfSpecies; j++)
                                endTAPE->scaling_D1[j] = scaling_D1_[j];
                        }

                        for (i = 0; i < numOfSpecies; i++)
                            for (j = 0; j < numOfSpecies; j++)
                                P_[j * numOfSpecies + i] =
                                    (Scalar)scalbn(P_[j * numOfSpecies + i], scaling_D1_[i]);

                        // LU factor (and overwrite) the matrix P and ip and jp
                        int info;
                        info = linalg::GETRF(numOfSpecies, numOfSpecies, P_, numOfSpecies, ip_);
                        if (info != 0) {
                            YABI_LOG_ERROR("ERROR: LuFactor Failed! (dgetrf info=", info, ")");
                            internalError = true;
                            releaseDirect();
                            releaseWorkspace();
                            return false;
                        }

                        if (taping) {
                            if (endTAPE->P == NULL)
                                endTAPE->P = new Scalar[numOfSpecies * numOfSpecies];
                            if (endTAPE->ip == NULL)
                                endTAPE->ip = new int[numOfSpecies];

                            for (i = 0; i < numOfSpecies; i++) {
                                endTAPE->ip[i] = ip_[i];
                                for (j = 0; j < numOfSpecies; j++)
                                    endTAPE->P[i * numOfSpecies + j] = P_[i * numOfSpecies + j];
                            }
                        }

                        for (z = 0; z < numOfDerivatives; z++) {
                            for (j = 0; j < numOfSpecies; j++)
                                cppAD_argument[j] = 0.0;

                            if (RHS_type >= 1)
                                cppAD_argument[numOfSpecies] = 0.0;

                            if (RHS_type == 2) {
                                for (j = 0; j < sizeOfParameter; j++)
                                    cppAD_argument[numOfSpecies + 1 + j] = p_seed_derivatives[z][j];
                            }

                            if (taping)
                                df = endTAPE->direct_fun->Forward(1, cppAD_argument);
                            else
                                df = RHS->Forward(1, cppAD_argument);

                            for (j = 0; j < numOfSpecies; j++)
                                b[j] = -(Scalar)scalbn(
                                    ((-z_predictor_derivatives[z][j]) -
                                     (((step_size_h / l_i[1]) * df[j]) -
                                      (z_predictor_derivatives[z][numOfSpecies + j] / l_i[1]))),
                                    scaling_D1_[j]);

                            linalg::GETRS_NT(numOfSpecies, blasONE, P_, numOfSpecies, ip_, b,
                                             numOfSpecies);

                            for (i = 0; i < numOfSpecies; i++)
                                corrector_derivatives[z][i] = (Scalar)scalbn(b[i], scaling_D2_[i]) -
                                                              z_predictor_derivatives[z][i];
                        }

                        delete[] P_;
                        delete[] ip_;

                        delete[] scaling_D1_;
                        delete[] scaling_D2_;
                    }

                    delete[] root;
                    delete[] b;
                    delete[] temp;
                    delete[] tempMACHINEprecision;

                    return true;
                }
            if (ratio > minimalRate && ratio > ACCEPT_NEWTON_CONVERGENCE_RATE) {
                if (setJacobian == true) {
                    convergenceIndicator = -1.0;
                    if (numOfDerivatives > 0 && !direct) {
                        for (l = 0; l < numOfDerivatives; l++) {
                            delete[] root_derivatives[l];
                            delete[] b_derivatives[l];
                            delete[] convergenceResidual_derivatives[l];
                        }
                        delete[] b_derivatives;
                        delete[] root_derivatives;
                        delete[] convergenceResidual_derivatives;
                    }
                    delete[] root;
                    delete[] b;
                    delete[] temp;
                    delete[] tempMACHINEprecision;

                    return true;
                }
                if (setNewP == true && setJacobian == false) {
                    newtonIterate = 1;
                    contructionRateFulfilled = false;
                    setJacobian = true;
                    for (i = 0; i < numOfSpecies; i++) {
                        corrector[i] = 0.0;
                        root[i] = z_predictor[i];
                    }
                    if (numOfDerivatives > 0 && !direct)
                        for (z = 0; z < numOfDerivatives; z++) {
                            for (i = 0; i < numOfSpecies; i++) {
                                root_derivatives[z][i] = z_predictor_derivatives[z][i];
                                corrector_derivatives[z][i] = 0.0;
                            }
                        }
                    continue;
                }

                if (setNewP == false && setJacobian == false) {
                    newtonIterate = 1;
                    contructionRateFulfilled = false;
                    setNewP = true;
                    for (i = 0; i < numOfSpecies; i++) {
                        root[i] = z_predictor[i];
                        corrector[i] = 0.0;
                    }
                    if (numOfDerivatives > 0 && !direct)
                        for (z = 0; z < numOfDerivatives; z++) {
                            for (i = 0; i < numOfSpecies; i++) {
                                root_derivatives[z][i] = z_predictor_derivatives[z][i];
                                corrector_derivatives[z][i] = 0.0;
                            }
                        }
                    continue;
                }
            }
            if (ratio < ACCEPT_NEWTON_CONVERGENCE_RATE) {
                contructionRateFulfilled = true;
            }
        }
        newtonIterate += 1;
    }

    convergenceIndicator = -1;

    if (numOfDerivatives > 0 && !direct) {
        for (l = 0; l < numOfDerivatives; l++) {
            delete[] root_derivatives[l];
            delete[] b_derivatives[l];
            delete[] convergenceResidual_derivatives[l];
        }
        delete[] b_derivatives;
        delete[] root_derivatives;
        delete[] convergenceResidual_derivatives;
    }
    delete[] root;
    delete[] b;
    delete[] temp;
    delete[] tempMACHINEprecision;

    return true;
}

template <typename Scalar> inline bool BDFIntegrator<Scalar>::calculate_next_corrector() {
    unsigned int i, j, z;

    for (i = 0; i < numOfSpecies; i++)
        for (j = 0; j <= order_corrector; j++)
            z_corrector[numOfSpecies * j + i] = z_predictor[numOfSpecies * j + i];

    int blasTEMP = order_corrector + 1;
    Scalar blasTEMP2 = 1.0;
    linalg::GER(numOfSpecies, blasTEMP, blasTEMP2, corrector, blasONE, l_i, blasONE, z_corrector,
                numOfSpecies);

    if (numOfDerivatives > 0)
        for (z = 0; z < numOfDerivatives; z++) {
            for (i = 0; i < numOfSpecies; i++)
                for (j = 0; j <= order_corrector; j++)
                    z_corrector_derivatives[z][numOfSpecies * j + i] =
                        z_predictor_derivatives[z][numOfSpecies * j + i];
            blasTEMP = order_corrector + 1;
            blasTEMP2 = 1.0;
            linalg::GER(numOfSpecies, blasTEMP, blasTEMP2, corrector_derivatives[z], blasONE, l_i,
                        blasONE, z_corrector_derivatives[z], numOfSpecies);
        }

    return true;
}

template <typename Scalar> unsigned int BDFIntegrator<Scalar>::accept_corrector_vector() {
    if (convergenceIndicator < 0 && !(step_size_h < 4.0 * MACHINEPRECISION * t_i[1])) {
        num_of_steps_per_order = 0;

        if (newtonFAIL > allowedNEWTON)
            return 3;
        newtonFAIL += 1;
        rejectedNewtonSteps++;
        Scalar eta = NEWTON_STEP_REDUCTION;
        step_size_h = step_size_h * eta;

        if (step_size_h < 4.0 * MACHINEPRECISION * t_i[1]) {
            YABI_LOG_ERROR("relaxing step size!");

            step_size_h = 4.0 * MACHINEPRECISION * t_i[1];
        }
        t_i[0] = step_size_h + t_i[1];
        return 0; // repeat step
    }
    if (convergenceIndicator < 0 && (step_size_h < 4.0 * MACHINEPRECISION * t_i[1])) {
        YABI_LOG_ERROR("WARNING: accept newton iteration due to minimal step size");
    }
    unsigned int i, z;
    bool pass = true;

    Scalar temp1;
    temp1 = 1.0;
    for (i = 1; i < order_corrector; i++)
        temp1 *= stepSizeSet_new[i] / (stepSizeSet_new[i] - step_size_h);
    temp1 += 1.0;
    // |LE| = |e| / (l_i[1] * (1 + product)); see the local-error
    // estimator in the dissertation (alpha_0 = -l_i[1]).
    temp1 *= l_i[1];

    Scalar *temp2 = new Scalar[numOfSpecies];

    for (i = 0; i < numOfSpecies; i++)
        temp2[i] = z_predictor[i];
    linalg::AXPY(numOfSpecies, blasScalarONE, corrector, blasONE, temp2, blasONE);
    calculate_error_weights(error_weights, temp2, absolut_error, relative_error);
    for (i = 0; i < numOfSpecies; i++)
        temp2[i] /= error_weights[i];
    reachableError =
        2 * MACHINEPRECISION * linalg::NRM2(numOfSpecies, temp2, blasONE) / numOfSpecies;
    if (reachableError < relative_error)
        reachableError = relative_error;

    else
        YABI_LOG_ERROR("WARNING: relax error threshold due to limiting machine precision");

    for (i = 0; i < numOfSpecies; i++)
        temp2[i] = corrector[i] / temp1 / error_weights[i];

    Scalar normWeightedLocalTruncError = linalg::NRM2(numOfSpecies, temp2, blasONE) / numOfSpecies;
    Scalar *normWeightedLocalTruncError_derivatives = NULL;

    if (normWeightedLocalTruncError > reachableError)
        pass = false;

    if (numOfDerivatives > 0) {
        normWeightedLocalTruncError_derivatives = new Scalar[numOfDerivatives];
        for (z = 0; z < numOfDerivatives; z++) {
            for (i = 0; i < numOfSpecies; i++)
                temp2[i] = z_predictor_derivatives[z][i];
            linalg::AXPY(numOfSpecies, blasScalarONE, corrector_derivatives[z], blasONE, temp2,
                         blasONE);
            calculate_error_weights(error_weights_derivatives[z], temp2,
                                    absolut_error_derivatives[z], relative_error_derivatives[z]);
            for (i = 0; i < numOfSpecies; i++)
                temp2[i] /= error_weights_derivatives[z][i];
            reachableError_derivatives[z] =
                2 * MACHINEPRECISION * linalg::NRM2(numOfSpecies, temp2, blasONE) / numOfSpecies;
            if (reachableError_derivatives[z] < relative_error_derivatives[z])
                reachableError_derivatives[z] = relative_error_derivatives[z];

            else
                YABI_LOG_ERROR("WARNING: relax error threshold due to limiting machine precision");

            for (i = 0; i < numOfSpecies; i++)
                temp2[i] = corrector_derivatives[z][i] / temp1 / error_weights_derivatives[z][i];

            normWeightedLocalTruncError_derivatives[z] =
                linalg::NRM2(numOfSpecies, temp2, blasONE) / numOfSpecies;

            if (normWeightedLocalTruncError_derivatives[z] > reachableError_derivatives[z])
                pass = false;
        }
    }
    delete[] temp2;
    if (step_size_h < 4.0 * MACHINEPRECISION * t_i[1]) {
        pass = true;

        std::cout << "WARNING: accept step due to limiting minimal step size";
    }
    if (pass) {
        acceptedSteps++;

        Scalar mu = ORDER_STAY_SAVETY_STEP * (reachableError / normWeightedLocalTruncError);
        Scalar sumTemp = 0.0;
        for (i = 1; i <= order_corrector; i++)
            sumTemp += (1.0 / i);

        minimalEta = sumTemp;
        sumTemp = 0.0;
        for (i = 0; i < order_corrector; i++)
            sumTemp += (step_size_h / stepSizeSet_new[i]);
        minimalEta /= sumTemp;
        for (i = 2; i <= order_corrector; i++)
            minimalEta *= stepSizeSet_new[i - 1] / step_size_h / i;
        minimalEta = pow(minimalEta, 1.0 / (order_corrector + 1));
        minimalEta *= pow(mu, 1.0 / (order_corrector + 1));

        if (numOfDerivatives > 0) {
            Scalar minimalEta_derivatives = 0.0;
            for (z = 0; z < numOfDerivatives; z++) {
                mu = ORDER_STAY_SAVETY_STEP *
                     (reachableError_derivatives[z] / normWeightedLocalTruncError_derivatives[z]);
                sumTemp = 0.0;
                for (i = 1; i <= order_corrector; i++)
                    sumTemp += (1.0 / i);
                minimalEta_derivatives = sumTemp;
                sumTemp = 0;
                for (i = 0; i < order_corrector; i++)
                    sumTemp += (step_size_h / stepSizeSet_new[i]);
                minimalEta_derivatives /= sumTemp;
                for (i = 2; i <= order_corrector; i++)
                    minimalEta_derivatives *= stepSizeSet_new[i - 1] / step_size_h / i;
                minimalEta_derivatives = pow(minimalEta_derivatives, 1.0 / (order_corrector + 1));
                minimalEta_derivatives *= pow(mu, 1.0 / (order_corrector + 1));
                if (minimalEta > minimalEta_derivatives)
                    minimalEta = minimalEta_derivatives;
            }
            delete[] normWeightedLocalTruncError_derivatives;
        }

        if (order_corrector == 1)
            if (minimalEta > RESTRICT_ORDER_1_RATIO)
                minimalEta = RESTRICT_ORDER_1_RATIO;

        if (order_corrector == 2)
            if (minimalEta > RESTRICT_ORDER_2_RATIO)
                minimalEta = RESTRICT_ORDER_2_RATIO;

        if (order_corrector == 3)
            if (minimalEta > RESTRICT_ORDER_3_RATIO)
                minimalEta = RESTRICT_ORDER_3_RATIO;

        if (order_corrector == 4)
            if (minimalEta > RESTRICT_ORDER_4_RATIO)
                minimalEta = RESTRICT_ORDER_4_RATIO;

        if (order_corrector == 5)
            if (minimalEta > RESTRICT_ORDER_5_RATIO)
                minimalEta = RESTRICT_ORDER_5_RATIO;

        if (order_corrector == 6)
            if (minimalEta > RESTRICT_ORDER_6_RATIO)
                minimalEta = RESTRICT_ORDER_6_RATIO;

        if (taping) {
            endTAPE->timePoint = t_i[0];
            endTAPE->stepSize = step_size_h;
        }

        // ACCEPTING STEP
        num_of_steps_per_order += 1;
        t_i[1] = t_i[0];
        t_i[0] = t_i[1] + step_size_h;
        if (t_i[1] >= t_ende - 2 * MACHINEPRECISION * t_ende)
            return 2; // calculate OutPUT-Vector
        else
            return 1; // next Step
    } else {
        // NOT ACCEPTING STEP

        num_of_steps_per_order = 0;

        if (stepFAIL > allowedSTEPS)
            return 4;

        rejectedSteps++;

        Scalar mu = STEP_NOT_ACCEPTED_SAVETY_STEP * (reachableError / normWeightedLocalTruncError);
        Scalar eta = 2.0;
        if (mu >= 0.05 && mu <= 1.0) {
            eta = sqrt(pow(mu, Scalar(0.5 + (1.0 / (order_corrector + 1)))));
        }
        if (mu < 0.05) {
            Scalar nu = 1;
            for (i = 1; i < order_corrector; i++)
                nu *= stepSizeSet_new[i] / step_size_h;
            for (i = 1; i < order_corrector; i++)
                nu /= (stepSizeSet_new[i] / step_size_h) - 1;
            nu /= l_i[1];
            nu = sqrt(nu);
            nu -= 1;
            nu *= pow(mu, Scalar(order_corrector - 1) / Scalar(2 * order_corrector + 2));
            if (nu > 1)
                nu = 1;
            eta = nu * pow(mu, 1.0 / Scalar(order_corrector + 1)) + (1 - nu) * pow(mu, 0.5);
        }
        if (mu > 1.0)
            eta = 2.0;

        if (numOfDerivatives > 0) {
            Scalar eta_derivatives = 0.0;
            for (z = 0; z < numOfDerivatives; z++) {
                mu = STEP_NOT_ACCEPTED_SAVETY_STEP *
                     (reachableError_derivatives[z] / normWeightedLocalTruncError_derivatives[z]);
                if (mu >= 0.05 && mu <= 1.0) {
                    eta_derivatives = sqrt(pow(mu, Scalar(0.5 + (1.0 / (order_corrector + 1)))));
                }
                if (mu < 0.05) {
                    Scalar nu = 1;
                    for (i = 1; i < order_corrector; i++)
                        nu *= stepSizeSet_new[i] / step_size_h;
                    for (i = 1; i < order_corrector; i++)
                        nu /= (stepSizeSet_new[i] / step_size_h) - 1;
                    nu /= l_i[1];
                    nu = sqrt(nu);
                    nu -= 1.0;
                    nu *= pow(mu, Scalar(order_corrector - 1) / Scalar(2 * order_corrector + 2));
                    if (nu > 1.0)
                        nu = 1.0;
                    eta_derivatives =
                        nu * pow(mu, 1.0 / Scalar(order_corrector + 1)) + (1.0 - nu) * pow(mu, 0.5);
                }
                if (mu > 1.0)
                    eta_derivatives = 2.0;
                if (eta_derivatives < eta)
                    eta = eta_derivatives;
            }
        }

        if (eta < 0.0)
            YABI_LOG_ERROR("step size reduction is negative check code!");

        stepFAIL += 1;

        // DASSL/LSODE-style order reduction (Gear 1971; Byrne & Hindmarsh 1975; Petzold 1982;
        // Hindmarsh/LSODE): on a rejected step, estimate the local truncation error at order-1
        // from the highest Nordsieck term and, if dropping the order gives a smaller (or already
        // acceptable) error, reduce the order instead of only shrinking the step. The reduction is
        // recorded in the tape (orderDOWN marker below) so the sensitivity replay reconstructs it.
        if (order_corrector > MINORDER) {
            Scalar c_down = -1.0;
            for (i = 0; i < order_corrector - 1; i++)
                c_down *= stepSizeSet_new[i] / step_size_h;
            c_down /= (l_i[1] - (step_size_h / stepSizeSet_new[order_corrector - 1]));
            Scalar *err_down = new Scalar[numOfSpecies];
            for (i = 0; i < numOfSpecies; i++)
                err_down[i] = z_corrector[numOfSpecies * order_corrector + i] * c_down / error_weights[i];
            Scalar norm_err_down = linalg::NRM2(numOfSpecies, err_down, blasONE) / numOfSpecies;
            delete[] err_down;
            if (norm_err_down < normWeightedLocalTruncError || norm_err_down <= reachableError) {
                order_corrector -= 1;
                stepFAIL = 0;
                YABI_LOG_WARN("reducing BDF order to ", order_corrector,
                              " (order-1 error estimate)");

                // Record the mid-step order reduction in the sensitivity tape so the replay
                // reconstructs the correct order. Insert an orderDOWN marker entry immediately
                // before the current step entry (mirrors prepare_next_step's order-change marker).
                if (taping && endTAPE != NULL) {
                    bdf_TAPE *marker = new bdf_TAPE;
                    marker->P = NULL;
                    marker->endSolution = NULL;
                    marker->direct_fun = NULL;
                    marker->old_parameter_seed = NULL;
                    marker->old_y_seed = NULL;
                    marker->updatedTaylorFunction = false;
                    marker->lastTaylorOrder = 0;
                    marker->CppAD_tape = new AbstractRHS<Scalar> *[MAXNEWTONITERATION];
                    for (i = 0; i < MAXNEWTONITERATION; i++)
                        marker->CppAD_tape[i] = NULL;
                    marker->ip = NULL;
                    marker->scaling_D1 = NULL;
                    marker->scaling_D2 = NULL;
                    marker->calculate_output = false;
                    marker->orderUP = false;
                    marker->orderDOWN = true;
                    marker->stepSize = step_size_h;
                    marker->timePoint = t_i[1];
                    marker->forwardLINK = endTAPE;
                    marker->backLINK = endTAPE->backLINK;
                    if (endTAPE->backLINK != NULL)
                        endTAPE->backLINK->forwardLINK = marker;
                    endTAPE->backLINK = marker;
                }
            }
        }

        if (eta > 1.0) {
            eta = STEP_NOT_ACCEPTED_REDUCTION;

            YABI_LOG_ERROR("force step size reduction!");
        }
        step_size_h = step_size_h * eta;
        if (step_size_h < 4.0 * MACHINEPRECISION * t_i[1]) {
            YABI_LOG_ERROR("relaxing step size!");

            step_size_h = 4.0 * MACHINEPRECISION * t_i[1];
        }
        t_i[0] = step_size_h + t_i[1];

        if (numOfDerivatives > 0)
            delete[] normWeightedLocalTruncError_derivatives;
        return 0; // repeat step
    }
    return true;
}

} // namespace yabi

#endif
