/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  Copyright (C) 2010-2012 Dominik Skanda
 *
 *  SPDX-License-Identifier: Apache-2.0
 */
#ifndef YABI_DETAIL_SENSITIVITY_ENGINE_HPP
#define YABI_DETAIL_SENSITIVITY_ENGINE_HPP

#include <algorithm>
#include <array>
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
        // The Nordsieck array has just been initialised with the nominal
        // initial state (resetIntegration writes it before createNewTape()).
        // Higher-order Taylor replays start from that value, not from the
        // directional seed.
        initial_state.assign(numOfSpecies, (Scalar)0.0);
        for (unsigned int i = 0; i < numOfSpecies; i++)
            initial_state[i] = nordsieck.z_corrector[i];
        initial_time = start_integration_time;
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
            z_predictor_derivatives.assign(numOfDerivatives,
                                           std::vector<Scalar>(numOfSpecies * 7, (Scalar)0.0));
            z_corrector_derivatives.assign(numOfDerivatives,
                                           std::vector<Scalar>(numOfSpecies * 7, (Scalar)0.0));
            z_corrector_derivatives_old.assign(numOfDerivatives,
                                               std::vector<Scalar>(numOfSpecies * 7, (Scalar)0.0));
            corrector_derivatives.assign(numOfDerivatives,
                                         std::vector<Scalar>(numOfSpecies, (Scalar)0.0));
            corrector_old_derivatives.assign(numOfDerivatives,
                                             std::vector<Scalar>(numOfSpecies, (Scalar)0.0));
            solution_Vector_derivatives.assign(numOfDerivatives,
                                               std::vector<Scalar>(numOfSpecies, (Scalar)0.0));
            error_weights_derivatives.assign(numOfDerivatives,
                                             std::vector<Scalar>(numOfSpecies, (Scalar)0.0));
            for (i = 0; i < numOfDerivatives; i++) {
                for (j = 0; j < numOfSpecies; j++) {
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
        error_weights_derivatives.clear();
        z_predictor_derivatives.clear();
        z_corrector_derivatives.clear();
        corrector_derivatives.clear();
        corrector_old_derivatives.clear();
        solution_Vector_derivatives.clear();
        z_corrector_derivatives_old.clear();
        reachableError_derivatives.clear();

        numOfDerivatives = 0;
        return true;
    }

    // Replay algebra: l_i, predictor, corrector and output for the tape state.
    inline bool calculate_l_vector_forward() {
        unsigned int i, j, k;

        std::vector<Scalar> l_kahan(6, (Scalar)0.0);
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
                    l_i_tape.data(), blasONE, z_corrector_tape.data(), numOfSpecies);

        return true;
    }

    inline bool calculate_output_vector_forward() {
        unsigned int i, j;
        std::vector<Scalar> temp(order_corrector_tape + 1);
        for (i = 0; i <= order_corrector_tape; i++) {
            temp[i] = 1.0;
            for (j = 0; j < i; j++)
                temp[i] *= (intermediateTAPE->timePoint - time_tape) / step_size_h_tape;
        }

        int blasTEMP = order_corrector_tape + 1;
        linalg::GEMV_NT(numOfSpecies, blasTEMP, blasScalarONE, z_corrector_tape.data(),
                        numOfSpecies, temp.data(), blasONE, blasScalarZERO,
                        forward_solution_Vector.data(), blasONE);

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

        // Orders >= 2 need the full Taylor polynomial of every intermediate
        // value.  The legacy single-coefficient replay below cannot provide
        // that, so route to the polynomial replay.  Direct mode has no full
        // Taylor state of the implicit solve and stays limited to order 1.
        if (order >= 2) {
            if (direct) {
                YABI_LOG_ERROR("forward(order=", order,
                               ") is not supported in direct sensitivity mode; use direct=false.");
                return false;
            }
            if (!start_CppAD_tape->supports_taylor()) {
                YABI_LOG_ERROR("forward(order=", order,
                               ") needs a right-hand side with full Taylor support (CppAD); "
                               "the current RHS backend only provides single-coefficient Forward.");
                return false;
            }
            return forwardFull(order, y_seed, parameter_seed);
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
                std::vector<Scalar> d_kahan(4, (Scalar)0.0);
                Scalar y;
                Scalar t;

                std::vector<Scalar> d_i(6, (Scalar)0.0);
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
                std::vector<Scalar> temp(numOfSpecies);
                for (i = 0; i < numOfSpecies; i++)
                    temp[i] = z_corrector_tape[order_corrector_tape * numOfSpecies + i];
                Scalar blasTEMP = -1.0;
                linalg::GER(numOfSpecies, order_corrector_tape, blasTEMP, temp.data(), blasONE,
                            d_i.data(), blasONE, z_corrector_tape.data(), numOfSpecies);

                order_corrector_tape -= 1;
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

    // Full Taylor-polynomial forward replay for orders >= 2.
    //
    // Every intermediate quantity of the frozen BDF algorithm is a truncated
    // Taylor polynomial in the sensitivity parameter tau, with the initial
    // state y0 + tau*y_seed (and p0 + tau*parameter_seed).  The nonlinear RHS
    // is evaluated with CppAD's full-Taylor Forward(q, xq) so that all
    // coefficients 0..order are obtained at once.  The returned value is the
    // order-th derivative d^order/d tau^order of the output (order! times the
    // order-th Taylor coefficient), reproducing the documented forward API.
    bool forwardFull(unsigned int order, Scalar *y_seed, Scalar *parameter_seed) {
        const unsigned int n = numOfSpecies;
        const std::size_t P = (std::size_t)order + 1;
        if (initial_state.size() != n) {
            YABI_LOG_ERROR("forward(order=", order, "): initial state was not recorded!");
            return false;
        }

        auto blank = [&]() { return std::vector<Scalar>(P * n, (Scalar)0); };
        std::vector<std::vector<Scalar>> zc(7), zp(7);
        for (unsigned int j = 0; j < 7; j++) {
            zc[j] = blank();
            zp[j] = blank();
        }

        // Initial Nordsieck array: z0 = (y_I(tau), h_init * F(y_I(tau)), 0, ...)
        for (unsigned int i = 0; i < n; i++) {
            zc[0][i] = initial_state[i];
            zc[0][n + i] = (order >= 1 && y_seed != NULL) ? y_seed[i] : (Scalar)0;
        }

        const std::size_t domain = start_CppAD_tape->Domain();
        std::vector<Scalar> xpoly(domain * P, (Scalar)0);
        std::vector<Scalar> ypoly;

        for (unsigned int i = 0; i < n; i++)
            for (std::size_t q = 0; q < P; q++)
                xpoly[P * i + q] = zc[0][q * n + i];
        if (RHS_type >= 1)
            xpoly[P * n] = initial_time;
        if (RHS_type == 2)
            for (unsigned int j = 0; j < sizeOfParameter; j++) {
                xpoly[P * (n + 1 + j)] = parameter[j];
                if (parameter_seed != NULL)
                    xpoly[P * (n + 1 + j) + 1] = parameter_seed[j];
            }

        if (!start_CppAD_tape->taylorForward(order, xpoly, ypoly)) {
            YABI_LOG_ERROR("forward(order=", order,
                           "): the RHS tape failed to produce full Taylor coefficients.");
            return false;
        }
        for (std::size_t q = 0; q < P; q++)
            for (unsigned int i = 0; i < n; i++)
                zc[1][q * n + i] = startTape_stepSize * ypoly[P * i + q];

        unsigned int order_corrector = 1;
        Scalar stepSizeSet[6] = {0, 0, 0, 0, 0, 0};
        Scalar h = startTape_stepSize;
        Scalar time = initial_time;

        intermediateTAPE = startTAPE;
        while (intermediateTAPE != NULL) {
            adjointTapePointer = NULL;

            if (intermediateTAPE->orderUP) {
                order_corrector += 1;
                for (unsigned int i = 0; i < n; i++)
                    for (unsigned int j = order_corrector; j <= 6; j++)
                        for (std::size_t q = 0; q < P; q++)
                            zc[j][q * n + i] = 0.0;
                intermediateTAPE = intermediateTAPE->forwardLINK;
                continue;
            }

            if (intermediateTAPE->orderDOWN) {
                // Same order-reduction linear map as the scalar replay, applied
                // coefficient-wise to the Taylor polynomials.
                Scalar d_kahan[4] = {0, 0, 0, 0};
                Scalar d_i[6] = {0, 0, 0, 0, 0, 0};
                d_i[0] = 0.0;
                d_i[1] = 0.0;
                for (unsigned int i = 2; i <= order_corrector; i++)
                    for (unsigned int j = i; j > 1; j--) {
                        if (j == i) {
                            if (j != 6)
                                d_i[j] = 1.0;
                        } else {
                            Scalar y = d_i[j - 1] - (d_kahan[j - 2] * stepSizeSet[i - 3] / h);
                            Scalar t = (d_i[j] * stepSizeSet[i - 3] / h) + y;
                            d_kahan[j - 2] = (t - d_i[j]) - y;
                            d_i[j] = t;
                        }
                    }
                for (unsigned int j = 2; j < order_corrector; j++)
                    if (d_i[j] != 0.0)
                        for (std::size_t q = 0; q < P; q++)
                            for (unsigned int i = 0; i < n; i++)
                                zc[j][q * n + i] -= d_i[j] * zc[order_corrector][q * n + i];
                order_corrector -= 1;
                intermediateTAPE = intermediateTAPE->forwardLINK;
                continue;
            }

            if (intermediateTAPE->calculate_output == true) {
                std::vector<Scalar> out = blank();
                for (unsigned int j = 0; j <= order_corrector; j++) {
                    Scalar theta = 1.0;
                    const Scalar ratio =
                        (intermediateTAPE->timePoint - time) / h;
                    for (unsigned int m = 0; m < j; m++)
                        theta *= ratio;
                    for (std::size_t q = 0; q < P; q++)
                        for (unsigned int i = 0; i < n; i++)
                            out[q * n + i] += theta * zc[j][q * n + i];
                }
                Scalar factorial = 1.0;
                for (unsigned int k = 2; k <= order; k++)
                    factorial *= (Scalar)k;
                for (unsigned int i = 0; i < n; i++)
                    forward_solution_Vector[i] = factorial * out[(std::size_t)order * n + i];

                for (unsigned int i = 0; i < n; i++)
                    if (!std::isfinite(forward_solution_Vector[i]) ||
                        std::fabs(forward_solution_Vector[i]) > 1e50) {
                        YABI_LOG_ERROR("ERROR: forward sensitivity diverged (component ", i,
                                       " = ", forward_solution_Vector[i],
                                       "). The taping replay is unstable for this problem; use the "
                                       "simultaneous (variational) sensitivity instead.");
                        return false;
                    }

                adjointTapePointer = intermediateTAPE;
                possible_adjoint_order = order + 1;
                start_adjoint_from_forward_mode = true;
                getTAPE = true;
                lastForwardOrder = order;
                return true;
            }

            // Step-size rescaling of every Taylor coefficient of every column.
            if (intermediateTAPE->stepSize != h) {
                const Scalar ratio = intermediateTAPE->stepSize / h;
                for (unsigned int j = 0; j <= order_corrector; j++) {
                    Scalar factor = 1.0;
                    for (unsigned int m = 0; m < j; m++)
                        factor *= ratio;
                    for (std::size_t q = 0; q < P; q++)
                        for (unsigned int i = 0; i < n; i++)
                            zc[j][q * n + i] *= factor;
                }
            }
            h = intermediateTAPE->stepSize;
            time = intermediateTAPE->timePoint;
            for (unsigned int j = 5; j > 0; j--)
                stepSizeSet[j] = stepSizeSet[j - 1] + h;
            stepSizeSet[0] = h;

            // Predictor: z_p[j] = sum_{m=j}^{order} C(m,j) z_c[m],
            // coefficient-wise (the Nordsieck Taylor shift).
            for (unsigned int j = 0; j <= order_corrector; j++) {
                std::fill(zp[j].begin(), zp[j].end(), (Scalar)0);
                Scalar binomial = 1.0;
                for (unsigned int m = j; m <= order_corrector; m++) {
                    if (m > j)
                        binomial = binomial * (Scalar)m / (Scalar)(m - j);
                    for (std::size_t q = 0; q < P; q++)
                        for (unsigned int i = 0; i < n; i++)
                            zp[j][q * n + i] += binomial * zc[m][q * n + i];
                }
            }

            // l_i vector and iteration-matrix hand-off for the adjoint sweep.
            order_corrector_tape = order_corrector;
            step_size_h_tape = h;
            time_tape = time;
            for (unsigned int j = 0; j < 6; j++)
                stepSizeSet_tape[j] = stepSizeSet[j];
            calculate_l_vector_forward();

            if (!intermediateTAPE->P.empty()) {
                P_tape.resize(n * n);
                ip_tape.resize(n);
                scaling_D1_tape.resize(n);
                scaling_D2_tape.resize(n);
                for (unsigned int i = 0; i < n; i++) {
                    scaling_D1_tape[i] = intermediateTAPE->scaling_D1[i];
                    scaling_D2_tape[i] = intermediateTAPE->scaling_D2[i];
                    ip_tape[i] = intermediateTAPE->ip[i];
                    for (unsigned int j = 0; j < n; j++)
                        P_tape[i * n + j] = intermediateTAPE->P[i * n + j];
                }
            } else if (P_tape.empty() || ip_tape.empty()) {
                YABI_LOG_ERROR("forward(order=", order,
                               "): tape is missing the Newton iteration matrix.");
                return false;
            }

            // Simplified Newton iteration on the Taylor polynomials, with the
            // frozen iteration matrix and the recorded number of iterations.
            std::vector<Scalar> c = zp[0];
            std::vector<Scalar> delta_total = blank();
            for (unsigned int it = 0; it < intermediateTAPE->NewtonIterates; it++) {
                for (unsigned int i = 0; i < n; i++)
                    for (std::size_t q = 0; q < P; q++)
                        xpoly[P * i + q] = c[q * n + i];
                if (RHS_type >= 1) {
                    xpoly[P * n] = time;
                    for (std::size_t q = 1; q < P; q++)
                        xpoly[P * n + q] = 0.0;
                }
                if (RHS_type == 2)
                    for (unsigned int j = 0; j < sizeOfParameter; j++) {
                        xpoly[P * (n + 1 + j)] = parameter[j];
                        if (parameter_seed != NULL)
                            xpoly[P * (n + 1 + j) + 1] = parameter_seed[j];
                        for (std::size_t q = 2; q < P; q++)
                            xpoly[P * (n + 1 + j) + q] = 0.0;
                    }

                if (!intermediateTAPE->CppAD_tape[it]->taylorForward(order, xpoly, ypoly)) {
                    YABI_LOG_ERROR("forward(order=", order,
                                   "): the replay tape failed to produce full Taylor coefficients.");
                    return false;
                }

                std::vector<Scalar> bpol = blank();
                const Scalar inv_l1 = 1.0 / l_i_tape[1];
                const Scalar minus_h_over_l1 = -h / l_i_tape[1];
                for (std::size_t q = 0; q < P; q++)
                    for (unsigned int i = 0; i < n; i++)
                        bpol[q * n + i] = zp[1][q * n + i] * inv_l1 +
                                          minus_h_over_l1 * ypoly[P * i + q] + c[q * n + i] -
                                          zp[0][q * n + i];

                for (std::size_t q = 0; q < P; q++) {
                    for (unsigned int i = 0; i < n; i++)
                        bpol[q * n + i] = -(Scalar)scalbn(bpol[q * n + i], scaling_D1_tape[i]);
                    linalg::GETRS_NT(n, blasONE, P_tape.data(), n, ip_tape.data(),
                                     &bpol[q * n], n);
                    for (unsigned int i = 0; i < n; i++)
                        bpol[q * n + i] = (Scalar)scalbn(bpol[q * n + i], scaling_D2_tape[i]);
                }

                for (std::size_t k = 0; k < P * n; k++) {
                    c[k] += bpol[k];
                    delta_total[k] += bpol[k];
                }
            }

            for (unsigned int j = 0; j <= order_corrector; j++)
                for (std::size_t q = 0; q < P; q++)
                    for (unsigned int i = 0; i < n; i++)
                        zc[j][q * n + i] =
                            zp[j][q * n + i] + l_i_tape[j] * delta_total[q * n + i];

            intermediateTAPE = intermediateTAPE->forwardLINK;
        }

        YABI_LOG_ERROR("forward(order=", order, "): tape has no output entry.");
        return false;
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
        std::vector<Scalar> b(numOfSpecies);
        std::vector<Scalar> temp2(numOfSpecies);

        if (!direct) {
            std::vector<Scalar> temp(numOfSpecies, (Scalar)0.0);
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

                linalg::SCAL(numOfSpecies, blasTEMP1, b.data(), blasONE);
                linalg::AXPY(numOfSpecies, blasTEMP2, temp2.data(), blasONE, b.data(), blasONE);
                linalg::AXPY(numOfSpecies, blasScalarONE, correctorVector_tape.data(), blasONE,
                             b.data(), blasONE);
                linalg::AXPY(numOfSpecies, blasScalarMinusONE, z_predictor_tape.data(), blasONE,
                             b.data(), blasONE);
                for (j = 0; j < numOfSpecies; j++)
                    b[j] = -(Scalar)scalbn(b[j], scaling_D1_tape[j]);

                linalg::GETRS_NT(numOfSpecies, blasONE, P_tape.data(), numOfSpecies, ip_tape.data(),
                                 b.data(), numOfSpecies);

                for (j = 0; j < numOfSpecies; j++)
                    temp2[j] = (Scalar)scalbn(b[j], scaling_D2_tape[j]);

                linalg::AXPY(numOfSpecies, blasScalarONE, temp2.data(), blasONE,
                             correctorVector_tape.data(), blasONE);
                linalg::AXPY(numOfSpecies, blasScalarONE, temp2.data(), blasONE, temp.data(),
                             blasONE);
            }
            for (i = 0; i < numOfSpecies; i++)
                correctorVector_tape[i] = temp[i];
        } else {
            if (order != 0) {
                std::vector<Scalar> df(numOfSpecies);
                std::vector<Scalar> P_(numOfSpecies * numOfSpecies);
                std::vector<int> ip_(numOfSpecies);
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
                    info = linalg::GETRF(numOfSpecies, numOfSpecies, P_.data(), numOfSpecies, ip_.data());
                    if (info != 0) {
                        YABI_LOG_ERROR("ERROR: LuFactor Failed! (dgetrf info=", info, ")");
                        internalError = true;
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

                linalg::GETRS_NT(numOfSpecies, blasONE, P_.data(), numOfSpecies, ip_.data(), b.data(), numOfSpecies);

                for (i = 0; i < numOfSpecies; i++)
                    correctorVector_tape[i] =
                        (Scalar)scalbn(b[i], intermediateTAPE->scaling_D2[i]) - z_predictor_tape[i];

                for (j = 0; j < numOfSpecies; j++)
                    intermediateTAPE->old_y_seed[j] =
                        (Scalar)scalbn(b[j], intermediateTAPE->scaling_D2[j]);

                intermediateTAPE->updatedTaylorFunction = false;
                intermediateTAPE->lastTaylorOrder = order;

            } else
                for (i = 0; i < numOfSpecies; i++) {
                    correctorVector_tape[i] =
                        intermediateTAPE->endSolution[i] - z_predictor_tape[i];
                }
        }

        return true;
    }

    // --- simultaneous-derivative state (also consumed by Newton and step control) ---
    std::vector<std::vector<Scalar>> z_predictor_derivatives;
    std::vector<std::vector<Scalar>> z_corrector_derivatives;
    std::vector<std::vector<Scalar>> z_corrector_derivatives_old;
    Scalar **y_seed_derivatives = nullptr; // non-owning view of the caller's data
    Scalar **p_seed_derivatives = nullptr; // non-owning view of the caller's data
    unsigned int numOfDerivatives = 0;
    std::vector<std::vector<Scalar>> corrector_derivatives;
    std::vector<std::vector<Scalar>> corrector_old_derivatives;
    std::vector<std::vector<Scalar>> error_weights_derivatives;
    Scalar **absolut_error_derivatives = nullptr;      // non-owning view of the caller's data
    Scalar *relative_error_derivatives = nullptr;      // non-owning view of the caller's data
    std::vector<std::vector<Scalar>> solution_Vector_derivatives;
    std::vector<Scalar> reachableError_derivatives;

    // --- forward replay state ---
    std::vector<Scalar> initial_state;
    Scalar initial_time = 0.0;
    std::array<Scalar, 7> l_i_tape{};
    Scalar time_tape = 0.0;
    std::array<Scalar, 6> stepSizeSet_tape{};
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
