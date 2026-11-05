/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  Copyright (C) 2010-2012 Dominik Skanda
 *
 *  SPDX-License-Identifier: Apache-2.0
 */
#ifndef YABI_DETAIL_SENSITIVITY_ENGINE_HPP
#define YABI_DETAIL_SENSITIVITY_ENGINE_HPP

#include <cmath>
#include <list>
#include <memory>
#include <vector>

#include <yabi/abstract_rhs.hpp>
#include <yabi/detail/integrator_config.hpp>
#include <yabi/detail/nordsieck_state.hpp>
#include <yabi/detail/replay_tape.hpp>
#include <yabi/linear_algebra.hpp>
#include <yabi/log.hpp>

namespace yabi {
namespace detail {

// Forward and simultaneous (variational) sensitivity.
//
// The engine owns every sensitivity array: the simultaneous-derivative
// mirrors fed into the Newton/step-control code, the replay buffers used by
// forward mode, and the cursor that tells the adjoint engine where a reverse
// sweep may start.  Tape records, tape cursors and the CppAD function pool are
// shared with the Newton iteration, so they stay in the integrator façade and
// are borrowed here by reference.  This keeps the historical expressions intact
// while giving forward/simultaneous sensitivity a single owner.
template <typename Scalar> class SensitivityEngine {
  public:
    using AdFunList = std::list<std::unique_ptr<AbstractRHS<Scalar>>>;

    SensitivityEngine(NordsieckState<Scalar> &nordsieck, const IntegratorConfig<Scalar> &config,
                      std::unique_ptr<AbstractRHS<Scalar>> &RHS, const unsigned int &numOfSpecies,
                      const unsigned int &RHS_type, const unsigned int &sizeOfParameter,
                      std::vector<Scalar> &parameter, const Scalar &start_integration_time,
                      bool &direct, bool &internalError, ReplayTape<Scalar> *&startTAPE,
                      ReplayTape<Scalar> *&intermediateTAPE, AbstractRHS<Scalar> *&start_CppAD_tape,
                      const Scalar &startTape_stepSize, AdFunList &cppADFunList,
                      typename AdFunList::iterator &intermediateCppADFunIterator, int &blasONE,
                      Scalar &blasScalarONE, Scalar &blasScalarMinusONE, Scalar &blasScalarZERO)
        : nordsieck(nordsieck), config(config), RHS(RHS), numOfSpecies(numOfSpecies),
          RHS_type(RHS_type), sizeOfParameter(sizeOfParameter), parameter(parameter),
          start_integration_time(start_integration_time), direct(direct),
          internalError(internalError), startTAPE(startTAPE), intermediateTAPE(intermediateTAPE),
          start_CppAD_tape(start_CppAD_tape), startTape_stepSize(startTape_stepSize),
          cppADFunList(cppADFunList), intermediateCppADFunIterator(intermediateCppADFunIterator),
          blasONE(blasONE), blasScalarONE(blasScalarONE), blasScalarMinusONE(blasScalarMinusONE),
          blasScalarZERO(blasScalarZERO) {
    }

    // Size the forward output vector once the integrator knows the system size.
    void initialize(unsigned int numSpecies) {
        forward_solution_Vector.assign(numSpecies, (Scalar)0.0);
        correctorVector_tape.resize(numSpecies);
    }

    // Start a fresh tape: the next forward() must rebuild its replay state.
    void beginNewTape() {
        getTAPE = false;
        lastForwardOrder = 0;
    }

    void zeroForwardSolution() {
        for (unsigned int i = 0; i < forward_solution_Vector.size(); i++)
            forward_solution_Vector[i] = 0.0;
    }

    // Release the per-tape replay buffers without touching tape lifetime.
    void resetForwardBuffers() {
        z_corrector_tape.clear();
        z_predictor_tape.clear();
        P_tape.clear();
        ip_tape.clear();
        scaling_D1_tape.clear();
        scaling_D2_tape.clear();
        adjointTapePointer = nullptr;
    }

    // Allocate the simultaneous-derivative storage and seed the Taylor history
    // from the user's directional derivative.
    bool setSimultaneousDerivatives(Scalar **y_seed_derivatives_, Scalar **p_seed_derivatives_,
                                    Scalar *relative_error_derivatives_,
                                    Scalar **absolut_error_derivatives_,
                                    unsigned int numOfDerivatives_) {
        unsigned int i, j, k;
        numOfDerivatives = numOfDerivatives_;
        if (numOfDerivatives > 0 && y_seed_derivatives_ != NULL) {
            relative_error_derivatives = relative_error_derivatives_;
            absolut_error_derivatives = absolut_error_derivatives_;
            y_seed_derivatives = y_seed_derivatives_;
            p_seed_derivatives = p_seed_derivatives_;
            reachableError_derivatives.resize(numOfDerivatives);
            z_predictor_derivatives = new Scalar *[numOfDerivatives];
            z_corrector_derivatives = new Scalar *[numOfDerivatives];
            z_corrector_derivatives_old = new Scalar *[numOfDerivatives];
            corrector_derivatives = new Scalar *[numOfDerivatives];
            corrector_old_derivatives = new Scalar *[numOfDerivatives];
            solution_Vector_derivatives = new Scalar *[numOfDerivatives];
            error_weights_derivatives = new Scalar *[numOfDerivatives];
            for (i = 0; i < numOfDerivatives; i++) {
                z_predictor_derivatives[i] = new Scalar[numOfSpecies * 7];
                z_corrector_derivatives[i] = new Scalar[numOfSpecies * 7];
                z_corrector_derivatives_old[i] = new Scalar[numOfSpecies * 7];
                corrector_derivatives[i] = new Scalar[numOfSpecies];
                corrector_old_derivatives[i] = new Scalar[numOfSpecies];
                solution_Vector_derivatives[i] = new Scalar[numOfSpecies];
                error_weights_derivatives[i] = new Scalar[numOfSpecies];
                for (j = 0; j < numOfSpecies; j++) {
                    error_weights_derivatives[i][j] = 0.0;
                    for (k = 0; k < 7; k++)
                        z_predictor_derivatives[i][numOfSpecies * k + j] = 0.0;
                    z_corrector_derivatives[i][j] = y_seed_derivatives[i][j];
                    z_corrector_derivatives_old[i][j] = y_seed_derivatives[i][j];
                }

                unsigned int cppAD_argument_size = 0;
                if (RHS_type == 0)
                    cppAD_argument_size = numOfSpecies;
                if (RHS_type == 1)
                    cppAD_argument_size = numOfSpecies + 1;
                if (RHS_type == 2)
                    cppAD_argument_size = numOfSpecies + 1 + sizeOfParameter;

                std::vector<Scalar> cppAD_argument(cppAD_argument_size);
                for (j = 0; j < numOfSpecies; j++)
                    cppAD_argument[j] = y_seed_derivatives[i][j];

                if (RHS_type > 0)
                    cppAD_argument[numOfSpecies] = 0.0;

                if (RHS_type > 1)
                    for (j = 0; j < sizeOfParameter; j++)
                        cppAD_argument[numOfSpecies + 1 + j] = p_seed_derivatives[i][j];

                std::vector<Scalar> df(numOfSpecies);
                df = RHS->Forward(1, cppAD_argument);

                for (j = 0; j < numOfSpecies; j++) {
                    z_corrector_derivatives[i][numOfSpecies + j] = nordsieck.step_size_h * df[j];
                    z_corrector_derivatives_old[i][numOfSpecies + j] =
                        nordsieck.step_size_h * df[j];
                }
                for (j = 0; j < numOfSpecies; j++)
                    for (k = 2; k < 7; k++) {
                        z_corrector_derivatives[i][numOfSpecies * k + j] = 0.0;
                        z_corrector_derivatives_old[i][numOfSpecies * k + j] = 0.0;
                    }
            }
        } else
            numOfDerivatives = 0;
        return true;
    }

    // Release all simultaneous-derivative storage.
    bool freeSimultaneousDerivatives() {
        unsigned int i;

        if (error_weights_derivatives != NULL) {
            for (i = 0; i < numOfDerivatives; i++)
                delete[] error_weights_derivatives[i];
            delete[] error_weights_derivatives;
        }
        error_weights_derivatives = NULL;

        if (z_predictor_derivatives != NULL) {
            for (i = 0; i < numOfDerivatives; i++)
                delete[] z_predictor_derivatives[i];
            delete[] z_predictor_derivatives;
        }
        z_predictor_derivatives = NULL;
        if (z_corrector_derivatives != NULL) {
            for (i = 0; i < numOfDerivatives; i++)
                delete[] z_corrector_derivatives[i];
            delete[] z_corrector_derivatives;
        }
        z_corrector_derivatives = NULL;
        if (corrector_derivatives != NULL) {
            for (i = 0; i < numOfDerivatives; i++)
                delete[] corrector_derivatives[i];
            delete[] corrector_derivatives;
        }
        corrector_derivatives = NULL;
        if (corrector_old_derivatives != NULL) {
            for (i = 0; i < numOfDerivatives; i++)
                delete[] corrector_old_derivatives[i];
            delete[] corrector_old_derivatives;
        }
        corrector_old_derivatives = NULL;
        if (solution_Vector_derivatives != NULL) {
            for (i = 0; i < numOfDerivatives; i++)
                delete[] solution_Vector_derivatives[i];
            delete[] solution_Vector_derivatives;
        }
        solution_Vector_derivatives = NULL;

        if (z_corrector_derivatives_old != NULL) {
            for (i = 0; i < numOfDerivatives; i++)
                delete[] z_corrector_derivatives_old[i];
            delete[] z_corrector_derivatives_old;
        }
        z_corrector_derivatives_old = NULL;

        if (!reachableError_derivatives.empty())

            reachableError_derivatives.clear();

        numOfDerivatives = 0;
        return true;
    }

    // Replay algebra: l_i, predictor, corrector and output for the tape state.
    inline bool calculate_l_vector_forward() {
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

    inline bool calculate_next_predictor_forward() {
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

        int blasTEMP;
        switch (order_corrector_tape) {
        case 1:
            blasTEMP = 2;
            linalg::DGEMM_NT_T(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE,
                               z_corrector_tape.data(), numOfSpecies, A1, blasTEMP, blasScalarZERO,
                               z_predictor_tape.data(), numOfSpecies);
            break;
        case 2:
            blasTEMP = 3;
            linalg::DGEMM_NT_T(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE,
                               z_corrector_tape.data(), numOfSpecies, A2, blasTEMP, blasScalarZERO,
                               z_predictor_tape.data(), numOfSpecies);
            break;
        case 3:
            blasTEMP = 4;
            linalg::DGEMM_NT_T(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE,
                               z_corrector_tape.data(), numOfSpecies, A3, blasTEMP, blasScalarZERO,
                               z_predictor_tape.data(), numOfSpecies);
            break;
        case 4:
            blasTEMP = 5;
            linalg::DGEMM_NT_T(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE,
                               z_corrector_tape.data(), numOfSpecies, A4, blasTEMP, blasScalarZERO,
                               z_predictor_tape.data(), numOfSpecies);
            break;
        case 5:
            blasTEMP = 6;
            linalg::DGEMM_NT_T(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE,
                               z_corrector_tape.data(), numOfSpecies, A5, blasTEMP, blasScalarZERO,
                               z_predictor_tape.data(), numOfSpecies);
            break;
        case 6:
            blasTEMP = 7;
            linalg::DGEMM_NT_T(numOfSpecies, blasTEMP, blasTEMP, blasScalarONE,
                               z_corrector_tape.data(), numOfSpecies, A6, blasTEMP, blasScalarZERO,
                               z_predictor_tape.data(), numOfSpecies);
            break;
        }

        return true;
    }

    inline bool calculate_next_corrector_forward() {
        unsigned int i, j;
        for (i = 0; i < numOfSpecies; i++)
            for (j = 0; j <= order_corrector_tape; j++)
                z_corrector_tape[j * numOfSpecies + i] = z_predictor_tape[j * numOfSpecies + i];
        int blasTEMP = order_corrector_tape + 1;
        Scalar blasTEMP2 = 1.0;
        linalg::GER(numOfSpecies, blasTEMP, blasTEMP2, correctorVector_tape.data(), blasONE,
                    l_i_tape, blasONE, z_corrector_tape.data(), numOfSpecies);

        return true;
    }

    inline bool calculate_output_vector_forward() {
        unsigned int i, j;
        Scalar *temp = new Scalar[order_corrector_tape + 1];
        for (i = 0; i <= order_corrector_tape; i++) {
            temp[i] = 1.0;
            for (j = 0; j < i; j++)
                temp[i] *= (intermediateTAPE->timePoint - time_tape) / step_size_h_tape;
        }

        int blasTEMP = order_corrector_tape + 1;
        linalg::GEMV_NT(numOfSpecies, blasTEMP, blasScalarONE, z_corrector_tape.data(),
                        numOfSpecies, temp, blasONE, blasScalarZERO, forward_solution_Vector.data(),
                        blasONE);

        delete[] temp;
        return true;
    }

    // Forward-mode replay of the recorded tape, yielding the directional
    // derivative for the requested Taylor order.
    bool forward(unsigned int order, Scalar *y_seed, Scalar *parameter_seed, bool reset) {
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
            z_corrector_tape.resize(numOfSpecies * 7);
            if (z_predictor_tape.empty()) {
                z_predictor_tape.resize(numOfSpecies * 7);
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

            std::vector<Scalar> cppAD_argument(cppAD_argument_size);

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

            std::vector<Scalar> df(numOfSpecies);
            if (order > 1 && start_CppAD_tape->size_order() < order) {
                YABI_LOG_ERROR(
                    "forward(order=", order, ") requested, but the replay tape only stores ",
                    start_CppAD_tape->size_order(),
                    " Taylor order(s). CppAD Forward(q) is incremental: call forward(1), "
                    "forward(2), ... "
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
                P_tape.resize(numOfSpecies * numOfSpecies);
                ip_tape.resize(numOfSpecies);
                scaling_D1_tape.resize(numOfSpecies);
                scaling_D2_tape.resize(numOfSpecies);
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
                linalg::GER(numOfSpecies, order_corrector_tape, blasTEMP, temp, blasONE, d_i,
                            blasONE, z_corrector_tape.data(), numOfSpecies);

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
                                       "). The taping replay is unstable for this problem; use the "
                                       "simultaneous "
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

    // One replay step: assemble the sensitivity RHS, factor/reuse the tape's
    // Newton matrix and update the corrector vector.
    bool calculate_corrector_vector_forward(unsigned int order, Scalar *y_seed,
                                            Scalar *parameter_seed) {
        (void)y_seed;
        unsigned int i, j, k;
        if (!intermediateTAPE->P.empty() && !direct) {
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

        std::vector<Scalar> cppAD_argument(cppAD_argument_size);
        Scalar *b = new Scalar[numOfSpecies];
        Scalar *temp2 = new Scalar[numOfSpecies];

        if (!direct) {
            Scalar *temp = new Scalar[numOfSpecies];
            for (i = 0; i < numOfSpecies; i++)
                temp[i] = 0;
            for (i = 0; i < intermediateTAPE->NewtonIterates; i++) {
                std::vector<Scalar> rhs(numOfSpecies);
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
                linalg::AXPY(numOfSpecies, blasScalarONE, correctorVector_tape.data(), blasONE, b,
                             blasONE);
                linalg::AXPY(numOfSpecies, blasScalarMinusONE, z_predictor_tape.data(), blasONE, b,
                             blasONE);
                for (j = 0; j < numOfSpecies; j++)
                    b[j] = -(Scalar)scalbn(b[j], scaling_D1_tape[j]);

                linalg::GETRS_NT(numOfSpecies, blasONE, P_tape.data(), numOfSpecies, ip_tape.data(),
                                 b, numOfSpecies);

                for (j = 0; j < numOfSpecies; j++)
                    temp2[j] = (Scalar)scalbn(b[j], scaling_D2_tape[j]);

                linalg::AXPY(numOfSpecies, blasScalarONE, temp2, blasONE,
                             correctorVector_tape.data(), blasONE);
                linalg::AXPY(numOfSpecies, blasScalarONE, temp2, blasONE, temp, blasONE);
            }
            for (i = 0; i < numOfSpecies; i++)
                correctorVector_tape[i] = temp[i];
            delete[] temp;
        } else {
            if (order != 0) {
                std::vector<Scalar> df(numOfSpecies);
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

                    std::vector<Scalar> J_(numOfSpecies * numOfSpecies);

                    if (intermediateCppADFunIterator == cppADFunList.end()) {
                        std::unique_ptr<AbstractRHS<Scalar>> tempADFun(RHS->clone());
                        tempADFun->capacity_order(config.MINTAYLOR);
                        cppADFunList.push_back(std::move(tempADFun));
                        intermediateCppADFunIterator--;
                    }
                    intermediateTAPE->direct_fun = intermediateCppADFunIterator->get();
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
                                P_[k * numOfSpecies + j] = 1.0 - ((step_size_h_tape / l_i_tape[1]) *
                                                                  J_[j * numOfSpecies + k]);
                            else
                                P_[k * numOfSpecies + j] =
                                    -(step_size_h_tape / l_i_tape[1]) * J_[j * numOfSpecies + k];
                        }

                    if (intermediateTAPE->scaling_D1.empty())
                        intermediateTAPE->scaling_D1.resize(numOfSpecies);
                    if (intermediateTAPE->scaling_D2.empty())
                        intermediateTAPE->scaling_D2.resize(numOfSpecies);

                    for (j = 0; j < numOfSpecies; j++)
                        intermediateTAPE->scaling_D2[j] = 0;

                    for (i = 0; i < numOfSpecies; i++)
                        for (j = 0; j < numOfSpecies; j++)
                            P_[j * numOfSpecies + i] = (Scalar)scalbn(
                                P_[j * numOfSpecies + i], intermediateTAPE->scaling_D2[j]);

                    for (i = 0; i < numOfSpecies; i++) {
                        Scalar sum = 0.0;
                        for (j = 0; j < numOfSpecies; j++)
                            sum += fabs(P_[j * numOfSpecies + i]);
                        if (!(sum > 0.0) || !std::isfinite(sum)) {
                            YABI_LOG_ERROR(
                                "ERROR: cannot scale column ", i,
                                " of the Newton matrix (forward sensitivities, column sum ", sum,
                                ")!");

                            internalError = true;
                            releaseDirectForward();
                            return false;
                        }
                        intermediateTAPE->scaling_D1[i] = (int)ceil(logb(sum));
                    }

                    for (i = 0; i < numOfSpecies; i++)
                        for (j = 0; j < numOfSpecies; j++)
                            P_[j * numOfSpecies + i] = (Scalar)scalbn(
                                P_[j * numOfSpecies + i], intermediateTAPE->scaling_D1[i]);

                    // LU factor (and overwrite) the matrix P.data() and ip.data() and jp
                    int info;
                    info = linalg::GETRF(numOfSpecies, numOfSpecies, P_, numOfSpecies, ip_);
                    if (info != 0) {
                        YABI_LOG_ERROR("ERROR: LuFactor Failed! (dgetrf info=", info, ")");
                        internalError = true;
                        releaseDirectForward();
                        return false;
                    }

                    if (intermediateTAPE->P.empty())
                        intermediateTAPE->P.resize(numOfSpecies * numOfSpecies);
                    if (intermediateTAPE->ip.empty())
                        intermediateTAPE->ip.resize(numOfSpecies);

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

                if (intermediateTAPE->old_parameter_seed.empty())
                    intermediateTAPE->old_parameter_seed.resize(sizeOfParameter);
                if (intermediateTAPE->old_y_seed.empty())
                    intermediateTAPE->old_y_seed.resize(numOfSpecies);

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
                    b[j] = -(Scalar)scalbn(((-z_predictor_tape[j]) -
                                            (((step_size_h_tape / l_i_tape[1]) * df[j]) -
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
                    correctorVector_tape[i] =
                        intermediateTAPE->endSolution[i] - z_predictor_tape[i];
                }
        }

        delete[] b;
        delete[] temp2;

        return true;
    }

    // --- simultaneous-derivative state (also consumed by Newton and step control) ---
    Scalar **z_predictor_derivatives = nullptr;
    Scalar **z_corrector_derivatives = nullptr;
    Scalar **z_corrector_derivatives_old = nullptr;
    Scalar **y_seed_derivatives = nullptr;
    Scalar **p_seed_derivatives = nullptr;
    unsigned int numOfDerivatives = 0;
    Scalar **corrector_derivatives = nullptr;
    Scalar **corrector_old_derivatives = nullptr;
    Scalar **error_weights_derivatives = nullptr;
    Scalar **absolut_error_derivatives = nullptr;
    Scalar *relative_error_derivatives = nullptr;
    Scalar **solution_Vector_derivatives = nullptr;
    std::vector<Scalar> reachableError_derivatives;

    // --- forward replay state ---
    Scalar l_i_tape[7] = {};
    Scalar time_tape = 0.0;
    Scalar stepSizeSet_tape[6] = {};
    std::vector<Scalar> z_predictor_tape;
    std::vector<Scalar> z_corrector_tape;
    std::vector<Scalar> correctorVector_tape;
    Scalar step_size_h_tape = 0.0;
    std::vector<Scalar> forward_solution_Vector;
    unsigned int order_corrector_tape = 0;
    bool getTAPE = false;
    unsigned int lastForwardOrder = 0;
    std::vector<Scalar> P_tape;
    std::vector<int> ip_tape;
    std::vector<long> scaling_D1_tape;
    std::vector<long> scaling_D2_tape;

    // --- hand-off cursor for the adjoint engine ---
    ReplayTape<Scalar> *adjointTapePointer = nullptr;
    unsigned int possible_adjoint_order = 0;
    bool start_adjoint_from_forward_mode = true;

  private:
    NordsieckState<Scalar> &nordsieck;
    const IntegratorConfig<Scalar> &config;
    std::unique_ptr<AbstractRHS<Scalar>> &RHS;

    const unsigned int &numOfSpecies;
    const unsigned int &RHS_type;
    const unsigned int &sizeOfParameter;
    std::vector<Scalar> &parameter;
    const Scalar &start_integration_time;
    bool &direct;
    bool &internalError;

    ReplayTape<Scalar> *&startTAPE;
    ReplayTape<Scalar> *&intermediateTAPE;
    AbstractRHS<Scalar> *&start_CppAD_tape;
    const Scalar &startTape_stepSize;
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
