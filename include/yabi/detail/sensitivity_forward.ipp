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

template <typename Scalar> inline bool BDFIntegrator<Scalar>::calculate_l_vector_forward() {
    unsigned int i, j, k;

    Scalar *l_kahan = new Scalar[6];
    for (i = 0; i < 6; i++)
        l_kahan[i] = 0.0;
    Scalar y;
    Scalar t;

    l_i_tape[0] = 1.0;
    for (i = 1; i <= order_corrector_tape; i++)
        for (j = i; j > 0; j--) {
            if (j == i) {
                l_i_tape[j] = 1.0;
                for (k = 0; k < i; k++)
                    l_i_tape[j] *= (step_size_h_tape / stepSizeSet_tape[k]);
            } else {
                // l_i_tape[j]=l_i_tape[j]+(l_i_tape[j-1]*(step_size_h_tape/stepSizeSet_tape[i-1]));
                y = (l_i_tape[j - 1] * (step_size_h_tape / stepSizeSet_tape[i - 1])) -
                    l_kahan[j - 1];
                t = l_i_tape[j] + y;
                l_kahan[j - 1] = (t - l_i_tape[j]) - y;
                l_i_tape[j] = t;
            }
        }
    delete[] l_kahan;

    return true;
}

template <typename Scalar> inline bool BDFIntegrator<Scalar>::calculate_next_predictor_forward() {
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

    int blasTEMP;
    switch (order_corrector_tape) {
    case 1:
        blasTEMP = 2;
        linalg::DGEMM_NT_T(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE, z_corrector_tape,
                           numOfSpecies, A1, blasTEMP, blasScalarZERO, z_predictor_tape,
                           numOfSpecies);
        break;
    case 2:
        blasTEMP = 3;
        linalg::DGEMM_NT_T(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE, z_corrector_tape,
                           numOfSpecies, A2, blasTEMP, blasScalarZERO, z_predictor_tape,
                           numOfSpecies);
        break;
    case 3:
        blasTEMP = 4;
        linalg::DGEMM_NT_T(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE, z_corrector_tape,
                           numOfSpecies, A3, blasTEMP, blasScalarZERO, z_predictor_tape,
                           numOfSpecies);
        break;
    case 4:
        blasTEMP = 5;
        linalg::DGEMM_NT_T(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE, z_corrector_tape,
                           numOfSpecies, A4, blasTEMP, blasScalarZERO, z_predictor_tape,
                           numOfSpecies);
        break;
    case 5:
        blasTEMP = 6;
        linalg::DGEMM_NT_T(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE, z_corrector_tape,
                           numOfSpecies, A5, blasTEMP, blasScalarZERO, z_predictor_tape,
                           numOfSpecies);
        break;
    case 6:
        blasTEMP = 7;
        linalg::DGEMM_NT_T(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE, z_corrector_tape,
                           numOfSpecies, A6, blasTEMP, blasScalarZERO, z_predictor_tape,
                           numOfSpecies);
        break;
    }

    return true;
}

template <typename Scalar> inline bool BDFIntegrator<Scalar>::calculate_next_corrector_forward() {
    unsigned int i, j;
    for (i = 0; i < numOfSpecies; i++)
        for (j = 0; j <= order_corrector_tape; j++)
            z_corrector_tape[j * numOfSpecies + i] = z_predictor_tape[j * numOfSpecies + i];
    int blasTEMP = order_corrector_tape + 1;
    Scalar blasTEMP2 = 1.0;
    linalg::GER(numOfSpecies, blasTEMP, blasTEMP2, correctorVector_tape, blasONE, l_i_tape, blasONE,
                z_corrector_tape, numOfSpecies);

    return true;
}

template <typename Scalar> inline bool BDFIntegrator<Scalar>::calculate_output_vector_forward() {
    unsigned int i, j;
    Scalar *temp = new Scalar[order_corrector_tape + 1];
    for (i = 0; i <= order_corrector_tape; i++) {
        temp[i] = 1.0;
        for (j = 0; j < i; j++)
            temp[i] *= (intermediateTAPE->timePoint - time_tape) / step_size_h_tape;
    }

    int blasTEMP = order_corrector_tape + 1;
    linalg::GEMV_NT(numOfSpecies, blasTEMP, blasScalarONE, z_corrector_tape, numOfSpecies, temp,
                    blasONE, blasScalarZERO, forward_solution_Vector, blasONE);

    delete[] temp;
    return true;
}

template <typename Scalar>
bool BDFIntegrator<Scalar>::forward(unsigned int order, Scalar *y_seed, Scalar *parameter_seed,
                                    bool reset) {
    if (internalError) {
        YABI_LOG_ERROR("internal error please reset integration!");

        return false;
    }
    unsigned int i, j;
    if (startTAPE == NULL) {
        YABI_LOG_ERROR("no tape recorded!");

        return false;
    }

    if (reset || order != lastForwardOrder)
        getTAPE = false;
    lastForwardOrder = order;

    if (getTAPE == false) {
        getTAPE = true;
        if (z_corrector_tape == NULL)
            z_corrector_tape = new Scalar[numOfSpecies * 7];
        if (z_predictor_tape == NULL) {
            z_predictor_tape = new Scalar[numOfSpecies * 7];
            for (i = 0; i < numOfSpecies; i++)
                for (j = 0; j < 7; j++)
                    z_predictor_tape[j * numOfSpecies + i] = 0;
        }

        for (i = 0; i < numOfSpecies; i++)
            z_corrector_tape[i] = y_seed[i];

        for (i = 0; i < 6; i++)
            stepSizeSet_tape[i] = 0.0;

        unsigned int cppAD_argument_size = 0;

        if (RHS_type == 0)
            cppAD_argument_size = numOfSpecies;
        if (RHS_type == 1)
            cppAD_argument_size = numOfSpecies + 1;
        if (RHS_type == 2)
            cppAD_argument_size = numOfSpecies + 1 + sizeOfParameter;

        CppAD::vector<Scalar> cppAD_argument(cppAD_argument_size);

        time_tape = start_integration_time;
        for (i = 0; i < numOfSpecies; i++)
            cppAD_argument[i] = y_seed[i];
        if (RHS_type > 0) {
            if (order == 0)
                cppAD_argument[numOfSpecies] = time_tape;
            else
                cppAD_argument[numOfSpecies] = 0.0;
        }
        if (RHS_type > 1)
            for (i = 0; i < sizeOfParameter; i++)
                cppAD_argument[numOfSpecies + 1 + i] = parameter_seed[i];

        CppAD::vector<Scalar> df(numOfSpecies);
        if (order > 1 && start_CppAD_tape->size_order() < order) {
            YABI_LOG_ERROR("forward(order=", order, ") requested, but the replay tape only stores ",
                           start_CppAD_tape->size_order(),
                           " Taylor order(s). CppAD Forward(q) is incremental: call forward(1), forward(2), ... "
                           "in ascending order first (e.g. forward(1, ...) before forward(2, ...)).");
            return false;
        }
        df = start_CppAD_tape->Forward(order, cppAD_argument);

        for (i = 0; i < numOfSpecies; i++) {
            z_corrector_tape[numOfSpecies + i] = startTape_stepSize * df[i];
        }
        for (i = 0; i < numOfSpecies; i++)
            for (j = 2; j < 7; j++)
                z_corrector_tape[j * numOfSpecies + i] = 0.0;

        intermediateTAPE = startTAPE;
        order_corrector_tape = 1;
        step_size_h_tape = startTape_stepSize;

        if (!direct) {
            if (P_tape == NULL)
                P_tape = new Scalar[numOfSpecies * numOfSpecies];
            if (ip_tape == NULL)
                ip_tape = new int[numOfSpecies];
            if (scaling_D1_tape == NULL)
                scaling_D1_tape = new long[numOfSpecies];
            if (scaling_D2_tape == NULL)
                scaling_D2_tape = new long[numOfSpecies];
        }
        adjointTapePointer = NULL;
    }

    while (intermediateTAPE != NULL) {
        adjointTapePointer = NULL;
        if (intermediateTAPE->orderUP) {
            order_corrector_tape += 1;
            for (i = 0; i < numOfSpecies; i++)
                for (j = order_corrector_tape; j <= 6; j++)
                    z_corrector_tape[j * numOfSpecies + i] = 0.0;

            intermediateTAPE = intermediateTAPE->forwardLINK;
            continue;
        }
        if (intermediateTAPE->orderDOWN) {
            Scalar *d_kahan = new Scalar[4];
            for (i = 0; i < 4; i++)
                d_kahan[i] = 0.0;
            Scalar y;
            Scalar t;

            Scalar *d_i = new Scalar[6];
            d_i[0] = 0.0;
            d_i[1] = 0.0;
            for (i = 2; i <= order_corrector_tape; i++)
                for (j = i; j > 1; j--) {
                    if (j == i) {
                        if (j != 6)
                            d_i[j] = 1.0;
                    } else {
                        // d_i[j]=d_i[j-1]+d_i[j]*stepSizeSet_tape[i-3]/step_size_h_tape;
                        y = d_i[j - 1] -
                            (d_kahan[j - 2] * stepSizeSet_tape[i - 3] / step_size_h_tape);
                        t = (d_i[j] * stepSizeSet_tape[i - 3] / step_size_h_tape) + y;
                        d_kahan[j - 2] = (t - d_i[j]) - y;
                        d_i[j] = t;
                    }
                }
            Scalar *temp = new Scalar[numOfSpecies];
            for (i = 0; i < numOfSpecies; i++)
                temp[i] = z_corrector_tape[order_corrector_tape * numOfSpecies + i];
            Scalar blasTEMP = -1.0;
            linalg::GER(numOfSpecies, order_corrector_tape, blasTEMP, temp, blasONE, d_i, blasONE,
                        z_corrector_tape, numOfSpecies);

            order_corrector_tape -= 1;
            delete[] d_kahan;
            delete[] d_i;
            delete[] temp;
            intermediateTAPE = intermediateTAPE->forwardLINK;
            continue;
        }

        if (intermediateTAPE->calculate_output == true) {
            calculate_output_vector_forward();
            // T26 guard: the taping replay is unstable for high-order BDF over long stiff
            // intervals (HIRES t=321.812 blows up ~1e+100 vs the correct ~1e-3). Fail loudly
            // instead of returning a value ~100 orders of magnitude off — the simultaneous
            // (variational) sensitivity is the stable path.
            for (i = 0; i < numOfSpecies; i++)
                if (!std::isfinite(forward_solution_Vector[i]) ||
                    std::fabs(forward_solution_Vector[i]) > 1e50) {
                    YABI_LOG_ERROR("ERROR: forward sensitivity diverged (component ", i, " = ",
                                   forward_solution_Vector[i],
                                   "). The taping replay is unstable for this problem; use the simultaneous "
                                   "(variational) sensitivity instead.");
                    return false;
                }
            adjointTapePointer = intermediateTAPE;
            possible_adjoint_order = order + 1;
            start_adjoint_from_forward_mode = true;
            intermediateTAPE = intermediateTAPE->forwardLINK;
            return true;
        }

        if (intermediateTAPE->stepSize != step_size_h_tape) {
            Scalar temp;
            for (i = 0; i < numOfSpecies; i++)
                for (j = 0; j <= order_corrector_tape; j++) {
                    temp = intermediateTAPE->stepSize / step_size_h_tape;
                    z_corrector_tape[j * numOfSpecies + i] =
                        z_corrector_tape[j * numOfSpecies + i] * (Scalar)pow(temp, j);
                }
        }
        step_size_h_tape = intermediateTAPE->stepSize;
        time_tape = intermediateTAPE->timePoint;
        for (j = 5; j > 0; j--)
            stepSizeSet_tape[j] = stepSizeSet_tape[j - 1] + step_size_h_tape;
        stepSizeSet_tape[0] = step_size_h_tape;
        calculate_next_predictor_forward();
        calculate_l_vector_forward();
        calculate_corrector_vector_forward(order, y_seed, parameter_seed);
        if (internalError) {
            YABI_LOG_ERROR("internal error please reset integration!");

            return false;
        }
        calculate_next_corrector_forward();

        intermediateTAPE = intermediateTAPE->forwardLINK;
    }

    return true;
}

template <typename Scalar>
bool BDFIntegrator<Scalar>::calculate_corrector_vector_forward(unsigned int order, Scalar *y_seed,
                                                               Scalar *parameter_seed) {
    (void)y_seed;
    unsigned int i, j, k;
    if (intermediateTAPE->P != NULL && !direct) {
        for (i = 0; i < numOfSpecies; i++) {
            scaling_D1_tape[i] = intermediateTAPE->scaling_D1[i];
            scaling_D2_tape[i] = intermediateTAPE->scaling_D2[i];
            ip_tape[i] = intermediateTAPE->ip[i];
            for (j = 0; j < numOfSpecies; j++)
                P_tape[i * numOfSpecies + j] = intermediateTAPE->P[i * numOfSpecies + j];
        }
    }

    for (i = 0; i < numOfSpecies; i++)
        correctorVector_tape[i] = z_predictor_tape[i];

    unsigned int cppAD_argument_size = 0;

    if (RHS_type == 0)
        cppAD_argument_size = numOfSpecies;
    if (RHS_type == 1)
        cppAD_argument_size = numOfSpecies + 1;
    if (RHS_type == 2)
        cppAD_argument_size = numOfSpecies + 1 + sizeOfParameter;

    CppAD::vector<Scalar> cppAD_argument(cppAD_argument_size);
    Scalar *b = new Scalar[numOfSpecies];
    Scalar *temp2 = new Scalar[numOfSpecies];

    if (!direct) {
        Scalar *temp = new Scalar[numOfSpecies];
        for (i = 0; i < numOfSpecies; i++)
            temp[i] = 0;
        for (i = 0; i < intermediateTAPE->NewtonIterates; i++) {
            CppAD::vector<Scalar> rhs(numOfSpecies);
            for (j = 0; j < numOfSpecies; j++)
                cppAD_argument[j] = correctorVector_tape[j];

            if (RHS_type >= 1) {
                if (order == 0)
                    cppAD_argument[numOfSpecies] = time_tape;
                else
                    cppAD_argument[numOfSpecies] = 0;
            }
            if (RHS_type == 2)
                for (j = 0; j < sizeOfParameter; j++)
                    cppAD_argument[numOfSpecies + 1 + j] = parameter_seed[j];

            rhs = intermediateTAPE->CppAD_tape[i]->Forward(order, cppAD_argument);

            for (j = 0; j < numOfSpecies; j++) {
                temp2[j] = rhs[j];
                b[j] = z_predictor_tape[numOfSpecies + j];
            }

            Scalar blasTEMP1 = 1.0 / l_i_tape[1];
            Scalar blasTEMP2 = -step_size_h_tape / l_i_tape[1];

            linalg::SCAL(numOfSpecies, blasTEMP1, b, blasONE);
            linalg::AXPY(numOfSpecies, blasTEMP2, temp2, blasONE, b, blasONE);
            linalg::AXPY(numOfSpecies, blasScalarONE, correctorVector_tape, blasONE, b, blasONE);
            linalg::AXPY(numOfSpecies, blasScalarMinusONE, z_predictor_tape, blasONE, b, blasONE);
            for (j = 0; j < numOfSpecies; j++)
                b[j] = -(Scalar)scalbn(b[j], scaling_D1_tape[j]);

            linalg::GETRS_NT(numOfSpecies, blasONE, P_tape, numOfSpecies, ip_tape, b, numOfSpecies);

            for (j = 0; j < numOfSpecies; j++)
                temp2[j] = (Scalar)scalbn(b[j], scaling_D2_tape[j]);

            linalg::AXPY(numOfSpecies, blasScalarONE, temp2, blasONE, correctorVector_tape,
                         blasONE);
            linalg::AXPY(numOfSpecies, blasScalarONE, temp2, blasONE, temp, blasONE);
        }
        for (i = 0; i < numOfSpecies; i++)
            correctorVector_tape[i] = temp[i];
        delete[] temp;
    } else {
        if (order != 0) {
            CppAD::vector<Scalar> df(numOfSpecies);
            Scalar *P_ = new Scalar[numOfSpecies * numOfSpecies];
            int *ip_ = new int[numOfSpecies];

            // Free this block's factorisation on a hard failure.
            auto releaseDirectForward = [&]() {
                delete[] P_;
                delete[] ip_;
            };
            if (intermediateTAPE->direct_fun == NULL) {
                for (j = 0; j < numOfSpecies; j++)
                    cppAD_argument[j] = intermediateTAPE->endSolution[j];

                if (RHS_type >= 1)
                    cppAD_argument[numOfSpecies] = time_tape;
                if (RHS_type == 2)
                    for (j = 0; j < sizeOfParameter; j++)
                        cppAD_argument[numOfSpecies + 1 + j] = parameter[j];

                CppAD::vector<Scalar> J_(numOfSpecies * numOfSpecies);

                if (intermediateCppADFunIterator == cppADFunList.end()) {
                    CppAD::ADFun<Scalar> *tempADFun = new CppAD::ADFun<Scalar>();
                    (*tempADFun) = (*RHS);
                    tempADFun->capacity_order(MINTAYLOR);
                    cppADFunList.push_back(tempADFun);
                    intermediateCppADFunIterator--;
                }
                intermediateTAPE->direct_fun = (*intermediateCppADFunIterator);
                intermediateCppADFunIterator++;

                intermediateTAPE->direct_fun->Forward(0, cppAD_argument);
                for (j = 0; j < cppAD_argument_size; j++)
                    cppAD_argument[j] = 0.0;

                for (j = 0; j < numOfSpecies; j++) {
                    cppAD_argument[j] = 1.0;
                    df = intermediateTAPE->direct_fun->Forward(1, cppAD_argument);
                    for (i = 0; i < numOfSpecies; i++) {
                        if (std::isnan(df[i]) || std::isinf(df[i])) {
                            YABI_LOG_ERROR(
                                "element of jacobian is nan or inf! (direct sensitivities)");

                            internalError = true;
                            releaseDirectForward();
                            return false;
                        }
                        J_[i * numOfSpecies + j] = df[i];
                    }
                    cppAD_argument[j] = 0.0;
                }
                for (j = 0; j < numOfSpecies; j++)
                    for (k = 0; k < numOfSpecies; k++) {
                        if (j == k)
                            P_[k * numOfSpecies + j] =
                                1.0 - ((step_size_h_tape / l_i_tape[1]) * J_[j * numOfSpecies + k]);
                        else
                            P_[k * numOfSpecies + j] =
                                -(step_size_h_tape / l_i_tape[1]) * J_[j * numOfSpecies + k];
                    }

                if (intermediateTAPE->scaling_D1 == NULL)
                    intermediateTAPE->scaling_D1 = new long[numOfSpecies];
                if (intermediateTAPE->scaling_D2 == NULL)
                    intermediateTAPE->scaling_D2 = new long[numOfSpecies];

                for (j = 0; j < numOfSpecies; j++)
                    intermediateTAPE->scaling_D2[j] = 0;

                for (i = 0; i < numOfSpecies; i++)
                    for (j = 0; j < numOfSpecies; j++)
                        P_[j * numOfSpecies + i] = (Scalar)scalbn(P_[j * numOfSpecies + i],
                                                                  intermediateTAPE->scaling_D2[j]);

                for (i = 0; i < numOfSpecies; i++) {
                    Scalar sum = 0.0;
                    for (j = 0; j < numOfSpecies; j++)
                        sum += fabs(P_[j * numOfSpecies + i]);
                    if (!(sum > 0.0) || !std::isfinite(sum)) {
                        YABI_LOG_ERROR("ERROR: cannot scale column ", i,
                                       " of the Newton matrix (forward sensitivities, column sum ",
                                       sum, ")!");

                        internalError = true;
                        releaseDirectForward();
                        return false;
                    }
                    intermediateTAPE->scaling_D1[i] = (int)ceil(logb(sum));
                }

                for (i = 0; i < numOfSpecies; i++)
                    for (j = 0; j < numOfSpecies; j++)
                        P_[j * numOfSpecies + i] = (Scalar)scalbn(P_[j * numOfSpecies + i],
                                                                  intermediateTAPE->scaling_D1[i]);

                // LU factor (and overwrite) the matrix P and ip and jp
                int info;
                info = linalg::GETRF(numOfSpecies, numOfSpecies, P_, numOfSpecies, ip_);
                if (info != 0) {
                    YABI_LOG_ERROR("ERROR: LuFactor Failed! (dgetrf info=", info, ")");
                    internalError = true;
                    releaseDirectForward();
                    return false;
                }

                if (intermediateTAPE->P == NULL)
                    intermediateTAPE->P = new Scalar[numOfSpecies * numOfSpecies];
                if (intermediateTAPE->ip == NULL)
                    intermediateTAPE->ip = new int[numOfSpecies];

                for (i = 0; i < numOfSpecies; i++) {
                    intermediateTAPE->ip[i] = ip_[i];
                    for (j = 0; j < numOfSpecies; j++)
                        intermediateTAPE->P[i * numOfSpecies + j] = P_[i * numOfSpecies + j];
                }
            } else {
                for (i = 0; i < numOfSpecies; i++) {
                    ip_[i] = intermediateTAPE->ip[i];
                    for (j = 0; j < numOfSpecies; j++)
                        P_[i * numOfSpecies + j] = intermediateTAPE->P[i * numOfSpecies + j];
                }
            }

            if (intermediateTAPE->old_parameter_seed == NULL)
                intermediateTAPE->old_parameter_seed = new Scalar[sizeOfParameter];
            if (intermediateTAPE->old_y_seed == NULL)
                intermediateTAPE->old_y_seed = new Scalar[numOfSpecies];

            if (order == 1) {
                for (j = 0; j < numOfSpecies; j++)
                    cppAD_argument[j] = 0.0;

                if (RHS_type >= 1)
                    cppAD_argument[numOfSpecies] = 0.0;

                if (RHS_type == 2) {
                    for (j = 0; j < sizeOfParameter; j++)
                        cppAD_argument[numOfSpecies + 1 + j] = parameter_seed[j];
                }

                df = intermediateTAPE->direct_fun->Forward(order, cppAD_argument);

                for (j = 0; j < sizeOfParameter; j++)
                    intermediateTAPE->old_parameter_seed[j] = parameter_seed[j];
            } else {
                for (j = 0; j < numOfSpecies; j++)
                    cppAD_argument[j] = intermediateTAPE->old_y_seed[j];

                if (RHS_type >= 1)
                    cppAD_argument[numOfSpecies] = 0.0;

                if (RHS_type == 2) {
                    for (j = 0; j < sizeOfParameter; j++)
                        cppAD_argument[numOfSpecies + 1 + j] =
                            intermediateTAPE->old_parameter_seed[j];
                }

                intermediateTAPE->direct_fun->Forward(order - 1, cppAD_argument);

                for (j = 0; j < numOfSpecies; j++)
                    cppAD_argument[j] = 0.0;

                if (RHS_type >= 1)
                    cppAD_argument[numOfSpecies] = 0.0;

                if (RHS_type == 2) {
                    for (j = 0; j < sizeOfParameter; j++)
                        cppAD_argument[numOfSpecies + 1 + j] = parameter_seed[j];
                }

                df = intermediateTAPE->direct_fun->Forward(order, cppAD_argument);

                for (j = 0; j < sizeOfParameter; j++)
                    intermediateTAPE->old_parameter_seed[j] = parameter_seed[j];
            }

            for (j = 0; j < numOfSpecies; j++)
                b[j] = -(Scalar)scalbn(
                    ((-z_predictor_tape[j]) - (((step_size_h_tape / l_i_tape[1]) * df[j]) -
                                               (z_predictor_tape[numOfSpecies + j] / l_i_tape[1]))),
                    intermediateTAPE->scaling_D1[j]);

            linalg::GETRS_NT(numOfSpecies, blasONE, P_, numOfSpecies, ip_, b, numOfSpecies);

            for (i = 0; i < numOfSpecies; i++)
                correctorVector_tape[i] =
                    (Scalar)scalbn(b[i], intermediateTAPE->scaling_D2[i]) - z_predictor_tape[i];

            for (j = 0; j < numOfSpecies; j++)
                intermediateTAPE->old_y_seed[j] =
                    (Scalar)scalbn(b[j], intermediateTAPE->scaling_D2[j]);

            intermediateTAPE->updatedTaylorFunction = false;
            intermediateTAPE->lastTaylorOrder = order;

            delete[] P_;
            delete[] ip_;

        } else
            for (i = 0; i < numOfSpecies; i++) {
                correctorVector_tape[i] = intermediateTAPE->endSolution[i] - z_predictor_tape[i];
            }
    }

    delete[] b;
    delete[] temp2;

    return true;
}

// Reverse Mode

} // namespace yabi

#endif
