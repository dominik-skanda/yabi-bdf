/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  Copyright (C) 2010-2012 Dominik Skanda
 *
 *  SPDX-License-Identifier: Apache-2.0
 */
#ifndef YABI_DETAIL_ADJOINT_ENGINE_HPP
#define YABI_DETAIL_ADJOINT_ENGINE_HPP

#include <array>
#include <cmath>
#include <iostream>
#include <list>
#include <memory>
#include <vector>

#include <yabi/abstract_rhs.hpp>
#include <yabi/detail/integrator_config.hpp>
#include <yabi/detail/newton_solver.hpp>
#include <yabi/detail/nordsieck_state.hpp>
#include <yabi/detail/replay_tape.hpp>
#include <yabi/detail/sensitivity_engine.hpp>
#include <yabi/linear_algebra.hpp>
#include <yabi/log.hpp>

namespace yabi {
namespace detail {

// Reverse / discrete-adjoint sensitivity.
//
// The engine owns the adjoint result arrays and the Taylor order they were
// computed for.  The replay data itself belongs to the forward engine (it is
// produced while the integration is taped) and the Newton matrix/scaling state
// belongs to the Newton solver, so both are borrowed.  The algorithm below is
// the historical replay body unchanged.
template <typename Scalar> class AdjointEngine {
  public:
    using AdFunList = std::list<std::unique_ptr<AbstractRHS<Scalar>>>;

    AdjointEngine(SensitivityEngine<Scalar> &sensitivity, NordsieckState<Scalar> &nordsieck,
                  NewtonSolver<Scalar> &newtonSolver, std::unique_ptr<AbstractRHS<Scalar>> &RHS,
                  const unsigned int &numOfSpecies, const unsigned int &RHS_type,
                  const unsigned int &sizeOfParameter, std::vector<Scalar> &parameter,
                  const IntegratorConfig<Scalar> &config, bool &direct, bool &internalError,
                  const Scalar &startTape_stepSize, AbstractRHS<Scalar> *&start_CppAD_tape,
                  AdFunList &cppADFunList,
                  typename AdFunList::iterator &intermediateCppADFunIterator, int &blasONE,
                  Scalar &blasScalarONE, Scalar &blasScalarMinusONE, Scalar &blasScalarZERO)
        : sensitivity(sensitivity), nordsieck(nordsieck), newtonSolver(newtonSolver), RHS(RHS),
          numOfSpecies(numOfSpecies), RHS_type(RHS_type), sizeOfParameter(sizeOfParameter),
          parameter(parameter), config(config), direct(direct), internalError(internalError),
          startTape_stepSize(startTape_stepSize), start_CppAD_tape(start_CppAD_tape),
          cppADFunList(cppADFunList), intermediateCppADFunIterator(intermediateCppADFunIterator),
          blasONE(blasONE), blasScalarONE(blasScalarONE), blasScalarMinusONE(blasScalarMinusONE),
          blasScalarZERO(blasScalarZERO) {
    }

    ~AdjointEngine() {
        release();
    }

    AdjointEngine(const AdjointEngine &) = delete;
    AdjointEngine &operator=(const AdjointEngine &) = delete;

    // Release the adjoint result arrays and forget their Taylor order.
    void release() {
        solution_adjoint_species.clear();
        solution_adjoint_parameter.clear();
        stored_adjoint_order = 0;
    }

    // Discrete-adjoint sweep over the recorded tape, producing the sensitivity
    // of the seeded functional for Taylor orders 1..order.
    bool reverse(unsigned int order, Scalar *seed_weights) {
        unsigned int i, j, k, l;

        if (sensitivity.adjointTapePointer == NULL) {
            std::cout << "Error: wrong use of reverse mode!" << std::endl;

            return false;
        }
        if (order <= 0 || order > sensitivity.possible_adjoint_order) {
            std::cout << "Error: calculation of reverse run of order!" << order
                      << "  is not possible!" << std::endl;

            return false;
        }

        if (direct && order >= 3) {
            std::cout << "Error: not implemented yes!" << std::endl;
        }

        Scalar step_size_h_adjoint = 0.0;
        Scalar step_size_h_adjoint_old = 0.0;
        std::array<Scalar, 6> stepSizeSet_adjoint{};
        std::array<Scalar, 7> l_i_adjoint{};
        unsigned int order_corrector_adjoint;

        std::vector<int> scaling_D1_adjoint;
        std::vector<int> scaling_D2_adjoint;
        std::vector<int> ip_adjoint;
        std::vector<Scalar> P_adjoint;

        if (!direct) {
            scaling_D1_adjoint.resize(numOfSpecies);
            scaling_D2_adjoint.resize(numOfSpecies);
            ip_adjoint.resize(numOfSpecies);
            P_adjoint.resize(numOfSpecies * numOfSpecies);
        }

        Scalar time_tape_adjoint;
        if (sensitivity.start_adjoint_from_forward_mode) {
            step_size_h_adjoint = sensitivity.step_size_h_tape;
            order_corrector_adjoint = sensitivity.order_corrector_tape;
            for (i = 0; i < 6; i++)
                stepSizeSet_adjoint[i] = sensitivity.stepSizeSet_tape[i];
            time_tape_adjoint = sensitivity.time_tape;
            for (i = 0; i < 7; i++)
                l_i_adjoint[i] = sensitivity.l_i_tape[i];

            if (!direct) {
                for (i = 0; i < numOfSpecies; i++) {
                    scaling_D1_adjoint[i] = sensitivity.scaling_D1_tape[i];
                    scaling_D2_adjoint[i] = sensitivity.scaling_D2_tape[i];
                    ip_adjoint[i] = sensitivity.ip_tape[i];
                    for (j = 0; j < numOfSpecies; j++)
                        P_adjoint[i * numOfSpecies + j] = sensitivity.P_tape[i * numOfSpecies + j];
                }
            }
        } else {
            step_size_h_adjoint = nordsieck.step_size_h_old;
            order_corrector_adjoint = nordsieck.order_corrector;
            for (i = 0; i < 6; i++)
                stepSizeSet_adjoint[i] = nordsieck.stepSizeSet_new[i];
            time_tape_adjoint = nordsieck.t_i[1];
            for (i = 0; i < 7; i++)
                l_i_adjoint[i] = nordsieck.l_i[i];

            if (!direct) {
                for (i = 0; i < numOfSpecies; i++) {
                    scaling_D1_adjoint[i] = newtonSolver.scaling_D1[i];
                    scaling_D2_adjoint[i] = newtonSolver.scaling_D2[i];
                    ip_adjoint[i] = newtonSolver.ip[i];
                    for (j = 0; j < numOfSpecies; j++)
                        P_adjoint[i * numOfSpecies + j] = newtonSolver.P[i * numOfSpecies + j];
                }
            }
        }

        ReplayTape<Scalar> *adjointIntermediatePointer;
        ReplayTape<Scalar> *adjointStepSizePointer;
        ReplayTape<Scalar> *adjointIterationMatrixPointer;

        std::vector<std::vector<Scalar>> z_adjoint(order,
                                                    std::vector<Scalar>(numOfSpecies * 7, (Scalar)0.0));
        std::vector<std::vector<Scalar>> z_adjoint_store(
            order, std::vector<Scalar>(numOfSpecies * 7, (Scalar)0.0));
        std::vector<std::vector<Scalar>> correctorVector_adjoint(
            order, std::vector<Scalar>(numOfSpecies, (Scalar)0.0));
        std::vector<std::vector<Scalar>> temp_adjoint(order,
                                                      std::vector<Scalar>(numOfSpecies, (Scalar)0.0));
        std::vector<std::vector<Scalar>> b_adjoint(order,
                                                   std::vector<Scalar>(numOfSpecies, (Scalar)0.0));
        std::vector<Scalar> temp(numOfSpecies);

        for (k = 0; k < order; k++) {
            for (i = 0; i < numOfSpecies; i++) {
                for (j = 0; j < 7; j++) {
                    if (k == 0 && j <= order_corrector_adjoint) {
                        z_adjoint[k][numOfSpecies * j + i] =
                            seed_weights[i] *
                            (Scalar)pow(
                                (sensitivity.adjointTapePointer->timePoint - time_tape_adjoint) /
                                    step_size_h_adjoint,
                                j);
                    }
                }
            }
        }

        adjointIntermediatePointer = sensitivity.adjointTapePointer;
        adjointStepSizePointer = NULL;
        adjointIterationMatrixPointer = NULL;

        solution_adjoint_species.assign(numOfSpecies, std::vector<Scalar>(order, (Scalar)0.0));
        solution_adjoint_parameter.assign(sizeOfParameter, std::vector<Scalar>(order, (Scalar)0.0));

        std::vector<Scalar> w(numOfSpecies * order);
        unsigned int cppAD_argument_size = 0;

        if (RHS_type == 0)
            cppAD_argument_size = numOfSpecies;
        if (RHS_type == 1)
            cppAD_argument_size = numOfSpecies + 1;
        if (RHS_type == 2)
            cppAD_argument_size = numOfSpecies + 1 + sizeOfParameter;
        std::vector<Scalar> dw((cppAD_argument_size)*order);

        // perform reverse mode
        while (adjointIntermediatePointer != NULL) {
            if (adjointIntermediatePointer->calculate_output == true) {
                adjointIntermediatePointer = adjointIntermediatePointer->backLINK;
                continue;
            }

            if (adjointIntermediatePointer->orderUP == true) {
                for (i = 0; i < numOfSpecies; i++)
                    for (j = order_corrector_adjoint; j < 7; j++)
                        for (k = 0; k < order; k++)
                            z_adjoint[k][numOfSpecies * j + i] = 0.0;

                order_corrector_adjoint -= 1;
                adjointIntermediatePointer = adjointIntermediatePointer->backLINK;
                continue;
            }

            if (adjointIntermediatePointer->orderDOWN == true) {
                std::vector<Scalar> d_kahan(4, (Scalar)0.0);
                Scalar y;
                Scalar t;

                order_corrector_adjoint += 1;
                std::vector<Scalar> d_i(6, (Scalar)0.0);

                d_i[0] = 0.0;
                d_i[1] = 0.0;
                for (i = 2; i <= order_corrector_adjoint; i++)
                    for (j = i; j > 1; j--) {
                        if (j == i) {
                            if (j != 6)
                                d_i[j] = 1.0;
                        } else {
                            // d_i[j]=d_i[j-1]+d_i[j]*stepSizeSet_adjoint[i-3]/step_size_h_adjoint;
                            y = d_i[j - 1] -
                                (d_kahan[j - 2] * stepSizeSet_adjoint[i - 3] / step_size_h_adjoint);
                            t = (d_i[j] * stepSizeSet_adjoint[i - 3] / step_size_h_adjoint) + y;
                            d_kahan[j - 2] = (t - d_i[j]) - y;
                            d_i[j] = t;
                        }
                    }

                for (k = 0; k < order; k++) {
                    for (i = 0; i < numOfSpecies; i++)
                        temp[i] = 0;

                    linalg::GEMV_NT(numOfSpecies, order_corrector_adjoint, blasScalarMinusONE,
                                    z_adjoint[k].data(), numOfSpecies, d_i.data(), blasONE,
                                    blasScalarZERO, temp.data(), blasONE);

                    for (i = 0; i < numOfSpecies; i++)
                        z_adjoint[k][numOfSpecies * order_corrector_adjoint + i] = temp[i];
                }
                adjointIntermediatePointer = adjointIntermediatePointer->backLINK;
                continue;
            }

            // next_l_vector

            std::vector<Scalar> l_kahan(6, (Scalar)0.0);
            Scalar y;
            Scalar t;

            l_i_adjoint[0] = 1.0;
            for (i = 1; i <= order_corrector_adjoint; i++)
                for (j = i; j > 0; j--) {
                    if (j == i) {
                        l_i_adjoint[j] = 1.0;
                        for (k = 0; k < i; k++)
                            l_i_adjoint[j] *= (step_size_h_adjoint / stepSizeSet_adjoint[k]);
                    } else {
                        // l_i_adjoint[j]=l_i_adjoint[j]+(l_i_adjoint[j-1]*(step_size_h_adjoint/stepSizeSet_adjoint[i-1]));
                        y = (l_i_adjoint[j - 1] *
                             (step_size_h_adjoint / stepSizeSet_adjoint[i - 1])) -
                            l_kahan[j - 1];
                        t = l_i_adjoint[j] + y;
                        l_kahan[j - 1] = (t - l_i_adjoint[j]) - y;
                        l_i_adjoint[j] = t;
                    }
                }

            // BEGIN ITERATIVE MODE

            if (!direct) {
                // next_corrector!

                for (k = 0; k < order; k++) {
                    for (i = 0; i < numOfSpecies; i++) {
                        temp[i] = 0.0;
                        correctorVector_adjoint[k][i] = 0.0;
                    }

                    int blasTEMP = order_corrector_adjoint + 1;
                    linalg::GEMV_NT(numOfSpecies, blasTEMP, blasScalarONE, z_adjoint[k].data(),
                                    numOfSpecies, l_i_adjoint.data(), blasONE, blasScalarZERO, temp.data(),
                                    blasONE);

                    for (i = 0; i < numOfSpecies; i++)
                        temp_adjoint[k][i] = temp[i];
                }

                for (i = 0; i < adjointIntermediatePointer->NewtonIterates; i++) {
                    for (k = 0; k < order; k++) {
                        for (j = 0; j < numOfSpecies; j++) {
                            b_adjoint[k][j] =
                                (Scalar)scalbn(temp_adjoint[k][j], scaling_D2_adjoint[j]);
                            temp[j] = (Scalar)scalbn(correctorVector_adjoint[k][j],
                                                     scaling_D2_adjoint[j]);
                        }

                        linalg::AXPY(numOfSpecies, blasScalarONE, temp.data(), blasONE, b_adjoint[k].data(),
                                     blasONE);
                        linalg::GETRS_T(numOfSpecies, blasONE, P_adjoint.data(), numOfSpecies, ip_adjoint.data(),
                                        b_adjoint[k].data(), numOfSpecies);

                        for (j = 0; j < numOfSpecies; j++) {
                            b_adjoint[k][j] =
                                -(Scalar)scalbn(b_adjoint[k][j], scaling_D1_adjoint[j]);
                            w[j * order + order - k - 1] =
                                (-b_adjoint[k][j] * step_size_h_adjoint) / l_i_adjoint[1];
                        }
                        Scalar blasTEMP = (Scalar)1.0 / l_i_adjoint[1];
                        linalg::AXPY(numOfSpecies, blasScalarONE, b_adjoint[k].data(), blasONE,
                                     correctorVector_adjoint[k].data(), blasONE);
                        linalg::AXPY(numOfSpecies, blasScalarMinusONE, b_adjoint[k].data(), blasONE,
                                     z_adjoint[k].data(), blasONE);
                        linalg::AXPY(numOfSpecies, blasTEMP, b_adjoint[k].data(), blasONE,
                                     &z_adjoint[k][numOfSpecies], blasONE);
                    }
                    dw = adjointIntermediatePointer
                             ->CppAD_tape[adjointIntermediatePointer->NewtonIterates - i - 1]
                             ->Reverse(order, w);

                    for (k = 0; k < order; k++) {
                        for (j = 0; j < numOfSpecies; j++)
                            temp[j] = dw[j * order + order - k - 1];
                        linalg::AXPY(numOfSpecies, blasScalarONE, temp.data(), blasONE,
                                     correctorVector_adjoint[k].data(), blasONE);
                    }

                    // SAVE PARAMETER VALUES

                    if (RHS_type == 2) {
                        for (j = 0; j < sizeOfParameter; j++)
                            for (k = 0; k < order; k++)
                                solution_adjoint_parameter[j][k] +=
                                    dw[(numOfSpecies + 1 + j) * order + order - k - 1];
                    }
                }

                for (k = 0; k < order; k++)
                    linalg::AXPY(numOfSpecies, blasScalarONE, correctorVector_adjoint[k].data(), blasONE,
                                 z_adjoint[k].data(), blasONE);

                if (!adjointIntermediatePointer->P.empty()) {
                    adjointIterationMatrixPointer = adjointIntermediatePointer->backLINK;
                    if (adjointIterationMatrixPointer != NULL)
                        while (adjointIterationMatrixPointer->P.empty()) {
                            adjointIterationMatrixPointer = adjointIterationMatrixPointer->backLINK;
                            if (adjointIterationMatrixPointer == NULL)
                                break;
                        }
                    if (adjointIterationMatrixPointer != NULL)
                        if (!adjointIterationMatrixPointer->P.empty()) {
                            for (i = 0; i < numOfSpecies; i++) {
                                scaling_D1_adjoint[i] =
                                    adjointIterationMatrixPointer->scaling_D1[i];
                                scaling_D2_adjoint[i] =
                                    adjointIterationMatrixPointer->scaling_D2[i];
                                ip_adjoint[i] = adjointIterationMatrixPointer->ip[i];
                                for (j = 0; j < numOfSpecies; j++)
                                    P_adjoint[i * numOfSpecies + j] =
                                        adjointIterationMatrixPointer->P[i * numOfSpecies + j];
                            }
                        }
                }
            }
            // END ITERATIVE MODE
            // BEGIN DIRECT MODE
            else {
                // IF NOT EXISTING CALCULATE LU FACTORIZATION

                if (adjointIntermediatePointer->direct_fun == NULL) {
                    if (adjointIntermediatePointer->P.empty())
                        adjointIntermediatePointer->P.resize(numOfSpecies * numOfSpecies);

                    unsigned int cppAD_argument_size = 0;

                    if (RHS_type == 0)
                        cppAD_argument_size = numOfSpecies;
                    if (RHS_type == 1)
                        cppAD_argument_size = numOfSpecies + 1;
                    if (RHS_type == 2)
                        cppAD_argument_size = numOfSpecies + 1 + sizeOfParameter;

                    std::vector<Scalar> cppAD_argument(cppAD_argument_size);

                    for (j = 0; j < numOfSpecies; j++)
                        cppAD_argument[j] = adjointIntermediatePointer->endSolution[j];

                    if (RHS_type >= 1)
                        cppAD_argument[numOfSpecies] = adjointIntermediatePointer->timePoint;
                    if (RHS_type == 2)
                        for (j = 0; j < sizeOfParameter; j++)
                            cppAD_argument[numOfSpecies + 1 + j] = parameter[j];

                    if (intermediateCppADFunIterator == cppADFunList.end()) {
                        std::unique_ptr<AbstractRHS<Scalar>> tempADFun(RHS->clone());
                        tempADFun->capacity_order(config.MINTAYLOR);
                        cppADFunList.push_back(std::move(tempADFun));
                        intermediateCppADFunIterator--;
                    }
                    adjointIntermediatePointer->direct_fun = intermediateCppADFunIterator->get();
                    intermediateCppADFunIterator++;

                    adjointIntermediatePointer->direct_fun->Forward(0, cppAD_argument);
                    for (j = 0; j < cppAD_argument_size; j++)
                        cppAD_argument[j] = 0.0;
                    std::vector<Scalar> df(numOfSpecies);
                    for (j = 0; j < numOfSpecies; j++) {
                        cppAD_argument[j] = 1.;
                        df = adjointIntermediatePointer->direct_fun->Forward(1, cppAD_argument);
                        for (i = 0; i < numOfSpecies; i++) {
                            if (std::isnan(df[i]) || std::isinf(df[i])) {
                                YABI_LOG_ERROR(
                                    "element of jacobian is nan or inf! (direct sensitivities)");

                                internalError = true;
                                return false;
                            }
                            adjointIntermediatePointer->P[j * numOfSpecies + i] = df[i];
                        }
                        cppAD_argument[j] = 0.0;
                    }
                    for (j = 0; j < numOfSpecies; j++)
                        for (k = 0; k < numOfSpecies; k++) {
                            if (j == k)
                                adjointIntermediatePointer->P[k * numOfSpecies + j] =
                                    1.0 - ((step_size_h_adjoint / l_i_adjoint[1]) *
                                           adjointIntermediatePointer->P[k * numOfSpecies + j]);
                            else
                                adjointIntermediatePointer->P[k * numOfSpecies + j] =
                                    -(step_size_h_adjoint / l_i_adjoint[1]) *
                                    adjointIntermediatePointer->P[k * numOfSpecies + j];
                        }

                    if (adjointIntermediatePointer->scaling_D1.empty())
                        adjointIntermediatePointer->scaling_D1.resize(numOfSpecies);
                    if (adjointIntermediatePointer->scaling_D2.empty())
                        adjointIntermediatePointer->scaling_D2.resize(numOfSpecies);

                    for (j = 0; j < numOfSpecies; j++)
                        adjointIntermediatePointer->scaling_D2[j] = 0;

                    for (i = 0; i < numOfSpecies; i++)
                        for (j = 0; j < numOfSpecies; j++)
                            adjointIntermediatePointer->P[j * numOfSpecies + i] =
                                (Scalar)scalbn(adjointIntermediatePointer->P[j * numOfSpecies + i],
                                               adjointIntermediatePointer->scaling_D2[j]);

                    for (i = 0; i < numOfSpecies; i++) {
                        Scalar sum = 0.0;
                        for (j = 0; j < numOfSpecies; j++)
                            sum += fabs(adjointIntermediatePointer->P[j * numOfSpecies + i]);
                        if (!(sum > 0.0) || !std::isfinite(sum)) {
                            YABI_LOG_ERROR(
                                "ERROR: cannot scale column ", i,
                                " of the Newton matrix (reverse sensitivities, column sum ", sum,
                                ")!");

                            internalError = true;
                            return false;
                        }
                        adjointIntermediatePointer->scaling_D1[i] = (int)ceil(logb(sum));
                    }

                    for (i = 0; i < numOfSpecies; i++)
                        for (j = 0; j < numOfSpecies; j++)
                            adjointIntermediatePointer->P[j * numOfSpecies + i] =
                                (Scalar)scalbn(adjointIntermediatePointer->P[j * numOfSpecies + i],
                                               adjointIntermediatePointer->scaling_D1[i]);

                    if (adjointIntermediatePointer->ip.empty())
                        adjointIntermediatePointer->ip.resize(numOfSpecies);

                    // LU factor (and overwrite) the matrix P and ip
                    int info;
                    info = linalg::GETRF(numOfSpecies, numOfSpecies,
                                         adjointIntermediatePointer->P.data(), numOfSpecies,
                                         adjointIntermediatePointer->ip.data());
                    if (info != 0) {
                        YABI_LOG_ERROR("ERROR: LuFactor Failed! (dgetrf info=", info, ")");
                        internalError = true;
                        return false;
                    }
                }

                // IF TAYLOR COEFFICIENTS ARE NOT UP TO DATE

                if ((adjointIntermediatePointer->updatedTaylorFunction == false) && (order > 1) &&
                    (adjointIntermediatePointer->lastTaylorOrder + 1 == order)) {
                    unsigned int cppAD_argument_size = 0;

                    if (RHS_type == 0)
                        cppAD_argument_size = numOfSpecies;
                    if (RHS_type == 1)
                        cppAD_argument_size = numOfSpecies + 1;
                    if (RHS_type == 2)
                        cppAD_argument_size = numOfSpecies + 1 + sizeOfParameter;

                    std::vector<Scalar> cppAD_argument(cppAD_argument_size);

                    for (j = 0; j < numOfSpecies; j++)
                        cppAD_argument[j] = adjointIntermediatePointer->old_y_seed[j];

                    if (RHS_type >= 1)
                        cppAD_argument[numOfSpecies] = 0.0;

                    if (RHS_type == 2) {
                        for (j = 0; j < sizeOfParameter; j++)
                            cppAD_argument[numOfSpecies + 1 + j] =
                                adjointIntermediatePointer->old_parameter_seed[j];
                    }

                    adjointIntermediatePointer->direct_fun->Forward(order - 1, cppAD_argument);

                    adjointIntermediatePointer->updatedTaylorFunction = true;
                }

                // CALCULATION OF CORRECTOR SEED VECTOR

                std::vector<Scalar> saveOldSeed(numOfSpecies);

                for (k = 0; k < order; k++) {
                    for (i = 0; i < numOfSpecies; i++)
                        correctorVector_adjoint[k][i] = 0.0;

                    int blasTEMP = order_corrector_adjoint + 1;
                    linalg::GEMV_NT(numOfSpecies, blasTEMP, blasScalarONE, z_adjoint[k].data(),
                                    numOfSpecies, l_i_adjoint.data(), blasONE, blasScalarZERO,
                                    correctorVector_adjoint[k].data(), blasONE);

                    linalg::AXPY(numOfSpecies, blasScalarMinusONE, correctorVector_adjoint[k].data(),
                                 blasONE, z_adjoint[k].data(), blasONE);

                    // CALCULATION OF THE RHS SEED INPUT MATRIX

                    for (i = 0; i < numOfSpecies; i++) {
                        if (k == 1) {
                            saveOldSeed[i] = correctorVector_adjoint[k][i];
                        }
                        correctorVector_adjoint[k][i] =
                            (Scalar)scalbn(correctorVector_adjoint[k][i],
                                           adjointIntermediatePointer->scaling_D2[i]);
                    }
                    linalg::GETRS_T(numOfSpecies, blasONE, adjointIntermediatePointer->P.data(),
                                    numOfSpecies, adjointIntermediatePointer->ip.data(),
                                    correctorVector_adjoint[k].data(), numOfSpecies);

                    for (i = 0; i < numOfSpecies; i++)
                        correctorVector_adjoint[k][i] =
                            (Scalar)scalbn(correctorVector_adjoint[k][i],
                                           adjointIntermediatePointer->scaling_D1[i]);

                    for (j = 0; j < numOfSpecies; j++)
                        w[j * order + order - k - 1] = -correctorVector_adjoint[k][j];
                }

                // CALCULATION OF THE RHS RESULT SEEDS
                dw = adjointIntermediatePointer->direct_fun->Reverse(order, w);

                // 0-th Taylor Coefficients

                if (RHS_type == 2)
                    for (j = 0; j < sizeOfParameter; j++) {
                        solution_adjoint_parameter[j][0] -=
                            (step_size_h_adjoint / l_i_adjoint[1]) *
                            dw[(numOfSpecies + 1 + j) * order + order - 0 - 1];
                    }

                linalg::AXPY(numOfSpecies, blasScalarONE, correctorVector_adjoint[0].data(), blasONE,
                             z_adjoint[0].data(), blasONE);
                Scalar blasTEMP = -(1 / l_i_adjoint[1]);
                linalg::AXPY(numOfSpecies, blasTEMP, correctorVector_adjoint[0].data(), blasONE,
                             &z_adjoint[0][numOfSpecies], blasONE);

                // 1-th Taylor Coefficients

                if (order > 1) {
                    linalg::AXPY(numOfSpecies, blasScalarONE, correctorVector_adjoint[1].data(), blasONE,
                                 z_adjoint[1].data(), blasONE);
                    linalg::AXPY(numOfSpecies, blasTEMP, correctorVector_adjoint[1].data(), blasONE,
                                 &z_adjoint[1][numOfSpecies], blasONE);

                    if (RHS_type == 2)
                        for (j = 0; j < sizeOfParameter; j++) {
                            solution_adjoint_parameter[j][1] -=
                                (step_size_h_adjoint / l_i_adjoint[1]) *
                                dw[(numOfSpecies + 1 + j) * order + order - 1 - 1];
                        }

                    for (j = 0; j < numOfSpecies; j++)
                        correctorVector_adjoint[0][j] = -(step_size_h_adjoint / l_i_adjoint[1]) *
                                                            dw[j * order + order - 1 - 1] -
                                                        correctorVector_adjoint[1][j] +
                                                        saveOldSeed[j];

                    for (i = 0; i < numOfSpecies; i++)
                        correctorVector_adjoint[0][i] =
                            (Scalar)scalbn(correctorVector_adjoint[0][i],
                                           adjointIntermediatePointer->scaling_D2[i]);

                    linalg::GETRS_T(numOfSpecies, blasONE, adjointIntermediatePointer->P.data(),
                                    numOfSpecies, adjointIntermediatePointer->ip.data(),
                                    correctorVector_adjoint[0].data(), numOfSpecies);

                    for (i = 0; i < numOfSpecies; i++)
                        correctorVector_adjoint[0][i] =
                            (Scalar)scalbn(correctorVector_adjoint[0][i],
                                           adjointIntermediatePointer->scaling_D1[i]);

                    for (k = 0; k < order; k++)
                        for (j = 0; j < numOfSpecies; j++)
                            w[j * order + order - k - 1] = 0.0;

                    for (j = 0; j < numOfSpecies; j++)
                        w[j] = -correctorVector_adjoint[0][j];

                    dw = adjointIntermediatePointer->direct_fun->Reverse(1, w);

                    if (RHS_type == 2)
                        for (j = 0; j < sizeOfParameter; j++)
                            solution_adjoint_parameter[j][1] -=
                                (step_size_h_adjoint / l_i_adjoint[1]) * dw[numOfSpecies + 1 + j];

                    linalg::AXPY(numOfSpecies, blasScalarONE, correctorVector_adjoint[0].data(), blasONE,
                                 z_adjoint[1].data(), blasONE);
                    linalg::AXPY(numOfSpecies, blasTEMP, correctorVector_adjoint[0].data(), blasONE,
                                 &z_adjoint[1][numOfSpecies], blasONE);
                }
            }

            // END DIRECT MODE
            // next _predictor!

            static Scalar A1[4] = {1., 0., 1., 1.};
            static Scalar A2[9] = {1., 0., 0., 1., 1., 0., 1., 2., 1.};
            static Scalar A3[16] = {1., 0., 0., 0., 1., 1., 0., 0., 1., 2., 1., 0., 1., 3., 3., 1.};
            static Scalar A4[25] = {1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 2., 1.,
                                    0., 0., 1., 3., 3., 1., 0., 1., 4., 6., 4., 1.};
            static Scalar A5[36] = {1., 0., 0., 0., 0., 0., 1., 1., 0.,  0.,  0., 0.,
                                    1., 2., 1., 0., 0., 0., 1., 3., 3.,  1.,  0., 0.,
                                    1., 4., 6., 4., 1., 0., 1., 5., 10., 10., 5., 1.};
            static Scalar A6[49] = {1., 0., 0., 0., 0., 0.,  0.,  1.,  1., 0., 0., 0.,  0.,
                                    0., 1., 2., 1., 0., 0.,  0.,  0.,  1., 3., 3., 1.,  0.,
                                    0., 0., 1., 4., 6., 4.,  1.,  0.,  0., 1., 5., 10., 10.,
                                    5., 1., 0., 1., 6., 15., 20., 15., 6., 1.};

            for (k = 0; k < order; k++) {
                for (i = 0; i < numOfSpecies; i++)
                    for (j = 0; j <= order_corrector_adjoint; j++)
                        z_adjoint_store[k][numOfSpecies * j + i] =
                            z_adjoint[k][numOfSpecies * j + i];

                int blasTEMP;
                switch (order_corrector_adjoint) {
                case 1:
                    blasTEMP = 2;
                    linalg::DGEMM_NT_NT(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE,
                                        z_adjoint_store[k].data(), numOfSpecies, A1, blasTEMP,
                                        blasScalarZERO, z_adjoint[k].data(), numOfSpecies);
                    break;
                case 2:
                    blasTEMP = 3;
                    linalg::DGEMM_NT_NT(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE,
                                        z_adjoint_store[k].data(), numOfSpecies, A2, blasTEMP,
                                        blasScalarZERO, z_adjoint[k].data(), numOfSpecies);
                    break;
                case 3:
                    blasTEMP = 4;
                    linalg::DGEMM_NT_NT(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE,
                                        z_adjoint_store[k].data(), numOfSpecies, A3, blasTEMP,
                                        blasScalarZERO, z_adjoint[k].data(), numOfSpecies);
                    break;
                case 4:
                    blasTEMP = 5;
                    linalg::DGEMM_NT_NT(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE,
                                        z_adjoint_store[k].data(), numOfSpecies, A4, blasTEMP,
                                        blasScalarZERO, z_adjoint[k].data(), numOfSpecies);
                    break;
                case 5:
                    blasTEMP = 6;
                    linalg::DGEMM_NT_NT(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE,
                                        z_adjoint_store[k].data(), numOfSpecies, A5, blasTEMP,
                                        blasScalarZERO, z_adjoint[k].data(), numOfSpecies);
                    break;
                case 6:
                    blasTEMP = 7;
                    linalg::DGEMM_NT_NT(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE,
                                        z_adjoint_store[k].data(), numOfSpecies, A6, blasTEMP,
                                        blasScalarZERO, z_adjoint[k].data(), numOfSpecies);
                    break;
                }
            }

            // change step_size nordsiek array
            step_size_h_adjoint_old = step_size_h_adjoint;

            adjointStepSizePointer = adjointIntermediatePointer->backLINK;
            if (adjointStepSizePointer != NULL)
                while (adjointStepSizePointer->calculate_output == true ||
                       adjointStepSizePointer->orderDOWN == true ||
                       adjointStepSizePointer->orderUP == true) {
                    adjointStepSizePointer = adjointStepSizePointer->backLINK;
                    if (adjointStepSizePointer == NULL)
                        break;
                }

            if (adjointStepSizePointer != NULL)
                step_size_h_adjoint = adjointStepSizePointer->stepSize;
            else
                step_size_h_adjoint = startTape_stepSize;

            if (step_size_h_adjoint != step_size_h_adjoint_old)
                for (i = 0; i < numOfSpecies; i++)
                    for (j = 0; j <= order_corrector_adjoint; j++)
                        for (k = 0; k < order; k++) {
                            Scalar tempPOW = step_size_h_adjoint_old / step_size_h_adjoint;
                            for (l = 0; l < j; l++)
                                z_adjoint[k][numOfSpecies * j + i] *= tempPOW;
                        }

            stepSizeSet_adjoint[0] = step_size_h_adjoint;

            i = 1;
            while (i < 6 && adjointStepSizePointer != NULL) {
                adjointStepSizePointer = adjointStepSizePointer->backLINK;
                if (adjointStepSizePointer != NULL)
                    while (adjointStepSizePointer->calculate_output == true ||
                           adjointStepSizePointer->orderDOWN == true ||
                           adjointStepSizePointer->orderUP == true) {
                        adjointStepSizePointer = adjointStepSizePointer->backLINK;
                        if (adjointStepSizePointer == NULL)
                            break;
                    }
                if (adjointStepSizePointer != NULL) {
                    stepSizeSet_adjoint[i] = adjointStepSizePointer->stepSize;
                    i++;
                }
            }

            for (i = i; i < 6; i++)
                stepSizeSet_adjoint[i] = 0;
            for (i = 4; i > 0; i--)
                for (j = i + 1; j < 6; j++)
                    stepSizeSet_adjoint[j] += stepSizeSet_adjoint[i];
            for (j = 1; j < 6; j++)
                stepSizeSet_adjoint[j] += stepSizeSet_adjoint[0];
            adjointIntermediatePointer = adjointIntermediatePointer->backLINK;
        }

        for (i = 0; i < numOfSpecies; i++)
            for (k = 0; k < order; k++)
                solution_adjoint_species[i][k] = z_adjoint[k][i];

        for (i = 0; i < numOfSpecies; i++)
            for (k = 0; k < order; k++)
                w[i * order + order - k - 1] = z_adjoint[k][numOfSpecies + i] * startTape_stepSize;

        dw = start_CppAD_tape->Reverse(order, w);

        for (i = 0; i < numOfSpecies; i++)
            for (k = 0; k < order; k++)
                solution_adjoint_species[i][k] += dw[i * order + order - k - 1];

        for (i = 0; i < sizeOfParameter; i++)
            for (k = 0; k < order; k++)
                solution_adjoint_parameter[i][k] +=
                    dw[(numOfSpecies + 1 + i) * order + order - k - 1];

        // CppAD's Reverse weights the k-th Taylor order by 1/k! (see the
        // Reverse definition: W = sum_k w^(k) / k! * d^k/dt^k Y).  The replay
        // therefore yields the adjoint of the k-th Taylor *coefficient*.
        // forward(order) returns order! times that coefficient (the pure
        // directional derivative), so rescale the adjoint of order k+1 by k!
        // to keep both APIs on the same pure-derivative convention.
        Scalar adjoint_factorial = 1.0;
        for (k = 0; k < order; k++) {
            if (k > 0)
                adjoint_factorial *= (Scalar)k;
            for (i = 0; i < numOfSpecies; i++)
                solution_adjoint_species[i][k] *= adjoint_factorial;
            for (i = 0; i < sizeOfParameter; i++)
                solution_adjoint_parameter[i][k] *= adjoint_factorial;
        }

        stored_adjoint_order = order;

        // (RAII: all workspace vectors above clean themselves up on return.)

        // T24 guard: same as forward — the taping replay is unstable for high-order BDF
        // over long stiff intervals. Fail loudly instead of returning a diverged value.
        for (i = 0; i < numOfSpecies; i++)
            for (j = 0; j < stored_adjoint_order; j++)
                if (!std::isfinite(solution_adjoint_species[i][j]) ||
                    std::fabs(solution_adjoint_species[i][j]) > 1e50) {
                    YABI_LOG_ERROR(
                        "ERROR: reverse sensitivity diverged. The taping replay is unstable for "
                        "this "
                        "problem; use the simultaneous (variational) sensitivity instead.");
                    return false;
                }

        return true;
    }

    // Adjoint results for Taylor orders 0..stored_adjoint_order-1.
    std::vector<std::vector<Scalar>> solution_adjoint_species;
    std::vector<std::vector<Scalar>> solution_adjoint_parameter;
    unsigned int stored_adjoint_order = 0;

  private:
    SensitivityEngine<Scalar> &sensitivity;
    NordsieckState<Scalar> &nordsieck;
    NewtonSolver<Scalar> &newtonSolver;
    std::unique_ptr<AbstractRHS<Scalar>> &RHS;

    const unsigned int &numOfSpecies;
    const unsigned int &RHS_type;
    const unsigned int &sizeOfParameter;
    std::vector<Scalar> &parameter;
    const IntegratorConfig<Scalar> &config;
    bool &direct;
    bool &internalError;

    const Scalar &startTape_stepSize;
    AbstractRHS<Scalar> *&start_CppAD_tape;
    AdFunList &cppADFunList;
    typename AdFunList::iterator &intermediateCppADFunIterator;

    int &blasONE;
    Scalar &blasScalarONE;
    Scalar &blasScalarMinusONE;
    Scalar &blasScalarZERO;
};

} // namespace detail
} // namespace yabi

#endif
