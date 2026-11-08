/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  Copyright (C) 2010-2012 Dominik Skanda
 *
 *  SPDX-License-Identifier: Apache-2.0
 */
#ifndef YABI_DETAIL_NEWTON_SOLVER_HPP
#define YABI_DETAIL_NEWTON_SOLVER_HPP

#include <algorithm>
#include <cmath>
#include <list>
#include <memory>
#include <vector>

#include <yabi/abstract_rhs.hpp>
#include <yabi/csc_matrix.hpp>
#include <yabi/detail/integrator_config.hpp>
#include <yabi/detail/nordsieck_state.hpp>
#include <yabi/detail/replay_tape.hpp>
#include <yabi/detail/step_controller.hpp>
#include <yabi/jacobian_backend.hpp>
#include <yabi/linear_solver.hpp>
#include <yabi/log.hpp>
#include <yabi/sparsity_pattern.hpp>

namespace yabi {
namespace detail {

// Newton iteration for the BDF corrector.
//
// The solver owns the Newton matrix, its LU pivots, the scaling vectors and the
// iteration state.  Everything else it needs is borrowed explicitly: the
// Nordsieck history (through NordsieckState), the model RHS, the linear solver,
// the error weights, the simultaneous-sensitivity mirrors and the replay tape.
// Keeping those as named references lets each algorithm remain the historical
// expression, while ownership of the Newton state moves to this component.
template <typename Scalar> class NewtonSolver {
  public:
    using AdFunList = std::list<std::unique_ptr<AbstractRHS<Scalar>>>;

    NewtonSolver(NordsieckState<Scalar> &nordsieck, const IntegratorConfig<Scalar> &config,
                 IntegratorStatistics &stats, std::unique_ptr<AbstractRHS<Scalar>> &RHS,
                 std::unique_ptr<LinearSolver<Scalar>> &linearSolver,
                 const unsigned int &numOfSpecies, const unsigned int &RHS_type,
                 const unsigned int &sizeOfParameter, std::vector<Scalar> &parameter,
                 std::vector<Scalar> &error_weights, std::vector<Scalar> &absolut_error,
                 Scalar &relative_error, std::vector<std::vector<Scalar>> &error_weights_derivatives,
                 Scalar **&absolut_error_derivatives, Scalar *&relative_error_derivatives,
                 Scalar &t_ende, Scalar &minimalEta, Scalar &reachableError,
                 std::vector<Scalar> &reachableError_derivatives, unsigned int &numOfDerivatives,
                 bool &direct, bool &taping, std::vector<std::vector<Scalar>> &z_predictor_derivatives,
                 std::vector<std::vector<Scalar>> &corrector_derivatives, Scalar **&p_seed_derivatives,
                 ReplayTape<Scalar> *&startTAPE, ReplayTape<Scalar> *&endTAPE,
                 ReplayTapeStore<Scalar> &replayTapeStore, AdFunList &cppADFunList,
                 typename AdFunList::iterator &intermediateCppADFunIterator, bool &internalError)
        : nordsieck(nordsieck), config(config), stats(stats), RHS(RHS), linearSolver(linearSolver),
          numOfSpecies(numOfSpecies), RHS_type(RHS_type), sizeOfParameter(sizeOfParameter),
          parameter(parameter), error_weights(error_weights), absolut_error(absolut_error),
          relative_error(relative_error), error_weights_derivatives(error_weights_derivatives),
          absolut_error_derivatives(absolut_error_derivatives),
          relative_error_derivatives(relative_error_derivatives), t_ende(t_ende),
          minimalEta(minimalEta), reachableError(reachableError),
          reachableError_derivatives(reachableError_derivatives),
          numOfDerivatives(numOfDerivatives), direct(direct), taping(taping),
          z_predictor_derivatives(z_predictor_derivatives),
          corrector_derivatives(corrector_derivatives), p_seed_derivatives(p_seed_derivatives),
          startTAPE(startTAPE), endTAPE(endTAPE), replayTapeStore(replayTapeStore),
          cppADFunList(cppADFunList), intermediateCppADFunIterator(intermediateCppADFunIterator),
          internalError(internalError) {
    }

    // Prepare the Newton matrix P = I - (h/l1)*J with the current error weights
    // and factor it in place; J and P are stored in scaled form for the solve.
    inline bool calculateNewtonScaling() {
        unsigned int i, j;
        for (j = 0; j < numOfSpecies; j++) {
            // logb() is undefined for 0 and inf; casting its result to int
            // is undefined behaviour that used to flush the whole Newton
            // matrix to 0 via scalbn(INT_MIN).  A non-positive weight means
            // atol=0 and y_j=0 (or a broken caller), which must be reported,
            // not silently turned into a wrong answer.
            if (!(error_weights[j] > 0.0) || !std::isfinite(error_weights[j])) {
                YABI_LOG_ERROR("ERROR: non-positive or non-finite error weight for species ", j,
                               " (", error_weights[j], ")!");

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
                YABI_LOG_ERROR("ERROR: cannot scale column ", i,
                               " of the Newton matrix (column sum ", sum, ")!");

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

    // Newton corrector loop: assemble/reuse J and P, solve the linear systems,
    // and stop on the historical convergence/rate tests.  On success the
    // corrector lives in the Nordsieck state.
    bool calculateCorrectorVector() {
        Scalar blasScalarONE = (Scalar)1.0;
        Scalar blasScalarMinusONE = (Scalar)-1.0;
        Scalar minimalRate = 0.0;
        Scalar previousNewtonResidual = 0.0;
        unsigned int i, j, k, l, z;
        bool wasJacobianSet = false;
        bool wasNewPSet = false;
        bool contructionRateFulfilled = false;

        int sign;

        Scalar error_multiplier = 1.0;
        for (i = 1; i < nordsieck.order_corrector; i++)
            error_multiplier *= nordsieck.stepSizeSet_new[i] /
                                (nordsieck.stepSizeSet_new[i] - nordsieck.step_size_h);
        error_multiplier += 1.0;
        // Local-error multiplier 1/alpha_0 * 1/(1 - 1/alpha_tilde_0)
        // with alpha_0 = -l_i[1]: the magnitude is l_i[1] * (1 + product).
        error_multiplier *= nordsieck.l_i[1];
        error_multiplier = fabs(error_multiplier);

        // Compute Matrix P.data()

        unsigned int cppAD_argument_size = 0;

        if (RHS_type == 0)
            cppAD_argument_size = numOfSpecies;
        if (RHS_type == 1)
            cppAD_argument_size = numOfSpecies + 1;
        if (RHS_type == 2)
            cppAD_argument_size = numOfSpecies + 1 + sizeOfParameter;

        std::vector<Scalar> cppAD_argument(cppAD_argument_size);
        // T31/T32: native CSC assembly is used only for a plain state solve.
        // The matrix-free path still assembles the sparse pattern, but only to
        // build/scale the preconditioner; J acts on vectors through the
        // MatrixFreeOperator, never as a stored dense matrix.
        //
        // T33: an explicit JacobianBackend overrides the capability-driven
        // selection.  A forced Sparse/MatrixFree backend that the RHS + solver
        // cannot satisfy fails loudly instead of silently downgrading to dense.
        bool matrixFreeRequested;
        bool sparseRequested;
        switch (jacobianBackend) {
        case JacobianBackend::Dense:
            matrixFreeRequested = false;
            sparseRequested = false;
            break;
        case JacobianBackend::Sparse:
            if (!sparsePathAvailable() || !prepareSparsePattern()) {
                YABI_LOG_ERROR("JacobianBackend::Sparse requested but unavailable: need a plain "
                               "state solve, a CSC-capable linear solver, and a RHS Jacobian "
                               "sparsity pattern.");
                internalError = true;
                return false;
            }
            matrixFreeRequested = false;
            sparseRequested = true;
            break;
        case JacobianBackend::MatrixFree: {
            if (!matrixFreePathAvailable()) {
                YABI_LOG_ERROR("JacobianBackend::MatrixFree requested but unavailable: need a "
                               "plain state solve and a matrix-free (Krylov) linear solver.");
                internalError = true;
                return false;
            }
            const bool withoutAssembly =
                linearSolver && !linearSolver->requiresAssembledMatrix();
            if (!withoutAssembly && !prepareSparsePattern()) {
                YABI_LOG_ERROR("JacobianBackend::MatrixFree requested but unavailable: the "
                               "installed solver needs an assembled preconditioner matrix, but "
                               "the RHS exposes no Jacobian sparsity pattern.");
                internalError = true;
                return false;
            }
            matrixFreeRequested = true;
            sparseRequested = false;
            break;
        }
        case JacobianBackend::Auto:
        default:
            matrixFreeRequested = matrixFreePathAvailable();
            sparseRequested = !matrixFreeRequested && sparsePathAvailable();
            break;
        }

        const bool matrixFreeWithoutAssembly =
            matrixFreeRequested && linearSolver && !linearSolver->requiresAssembledMatrix();
        const bool needSparsePattern =
            !matrixFreeWithoutAssembly && (matrixFreeRequested || sparseRequested);
        const bool sparsePatternReadyForSolve = !needSparsePattern || prepareSparsePattern();
        const bool useMatrixFree =
            matrixFreeRequested && (matrixFreeWithoutAssembly || sparsePatternReadyForSolve);
        const bool useJFree = useMatrixFree && matrixFreeWithoutAssembly;
        const bool useSparse = sparseRequested && sparsePatternReadyForSolve;
        const bool useSparseAssembly = useSparse || (useMatrixFree && !useJFree);
        if (matrixFreeRequested && !matrixFreeWithoutAssembly && !sparsePatternReadyForSolve &&
            !matrixFreeWarned) {
            matrixFreeWarned = true;
            YABI_LOG_WARN("matrix-free solver installed but the right-hand side exposes no Jacobian "
                          "sparsity pattern; using the dense explicit path");
        }
        std::vector<Scalar> root(numOfSpecies);

        for (i = 0; i < numOfSpecies; i++)
            root[i] = nordsieck.z_predictor[i];

        StepController<Scalar>::calculateErrorWeights(numOfSpecies, config.ERROR_WEIGHTS_INDEX,
                                                      error_weights.data(), root.data(),
                                                      absolut_error.data(), relative_error);

        std::vector<Scalar> b(numOfSpecies);
        std::vector<Scalar> temp(numOfSpecies);

        std::vector<std::vector<Scalar>> b_derivatives;
        std::vector<std::vector<Scalar>> root_derivatives;
        std::vector<std::vector<Scalar>> convergenceResidual_derivatives;
        if (numOfDerivatives > 0 && !direct) {
            root_derivatives.assign(numOfDerivatives, std::vector<Scalar>(numOfSpecies, (Scalar)0.0));
            b_derivatives.assign(numOfDerivatives, std::vector<Scalar>(numOfSpecies, (Scalar)0.0));
            convergenceResidual_derivatives.assign(numOfDerivatives,
                                                   std::vector<Scalar>(2, (Scalar)0.0));
            for (i = 0; i < numOfDerivatives; i++) {
                for (j = 0; j < numOfSpecies; j++) {
                    root_derivatives[i][j] = z_predictor_derivatives[i][j];
                    corrector_derivatives[i][j] = 0.0;
                }
                StepController<Scalar>::calculateErrorWeights(
                    numOfSpecies, config.ERROR_WEIGHTS_INDEX, error_weights_derivatives[i].data(),
                    z_predictor_derivatives[i].data(), absolut_error_derivatives[i],
                    relative_error_derivatives[i]);
                // Same guard as the main path (T09): a zero/non-finite weight makes the later
                // logb(error_weights_derivatives[z][i]) undefined (flushes the sensitivity Newton
                // matrix via scalbn(INT_MIN)). Report it instead of silently returning a wrong
                // answer.
                for (j = 0; j < numOfSpecies; j++)
                    if (!(error_weights_derivatives[i][j] > 0.0) ||
                        !std::isfinite(error_weights_derivatives[i][j])) {
                        YABI_LOG_ERROR("ERROR: non-positive or non-finite derivative error weight "
                                       "for direction ",
                                       i, ", species ", j, " (", error_weights_derivatives[i][j],
                                       ")!");

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
            nordsieck.corrector[i] = 0.0;

        std::vector<Scalar> tempMACHINEprecision(numOfSpecies);

        while (newtonIterate <= config.MAXNEWTONITERATION) {
            stats.newtonIterations++;
            if (setJacobian && !wasJacobianSet) {
                stats.updateJacobian++;
                wasJacobianSet = true;

                for (i = 0; i < numOfSpecies; i++)
                    cppAD_argument[i] = root[i];
                if (RHS_type >= 1)
                    cppAD_argument[numOfSpecies] = nordsieck.t_i[0];
                if (RHS_type == 2)
                    for (i = 0; i < sizeOfParameter; i++)
                        cppAD_argument[numOfSpecies + 1 + i] = parameter[i];
                RHS->Forward(0, cppAD_argument);

                if (useSparseAssembly) {
                    std::vector<Scalar> values;
                    if (!RHS->sparseJacobian(cppAD_argument, sparsePattern, values) ||
                        values.size() != J_csc.val.size()) {
                        YABI_LOG_ERROR("ERROR: sparse Jacobian evaluation failed!");

                        internalError = true;
                        return false;
                    }
                    for (std::size_t k = 0; k < values.size(); ++k)
                        if (std::isnan(values[k]) || std::isinf(values[k])) {
                            YABI_LOG_ERROR("element of jacobian is nan or inf!");

                            internalError = true;
                            return false;
                        }
                    J_csc.val = std::move(values);
                    formPFromJ();

                    if (!calculateNewtonScalingSparse()) {
                        YABI_LOG_ERROR("Newton scaling failed (invalid error weight)!");

                        internalError = true;
                        return false;
                    }
                    if (useMatrixFree) {
                        matrixFreeBase = cppAD_argument;
                        configureMatrixFreeOperator();
                    }
                    if (!linearSolver->factorizeCsc(P_csc, ip.data(), scaling_D1.data(),
                                                    scaling_D2.data(), numOfSpecies)) {
                        YABI_LOG_ERROR("ERROR: LuFactor Failed!");
                        internalError = true;
                        return false;
                    }
                } else if (useJFree) {
                    if (!configureJFreeMatrix(cppAD_argument)) {
                        YABI_LOG_ERROR("ERROR: J-free matrix-free setup failed!");
                        internalError = true;
                        return false;
                    }
                } else {
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
                                    1.0 - ((nordsieck.step_size_h / nordsieck.l_i[1]) *
                                           J[j * numOfSpecies + k]);
                            else
                                P[k * numOfSpecies + j] =
                                    -(nordsieck.step_size_h / nordsieck.l_i[1]) *
                                    J[j * numOfSpecies + k];

                    // LU factor (and overwrite) the matrix P.data() and ip.data() and jp via
                    // pluggable solver
                    if (!calculateNewtonScaling()) {
                        YABI_LOG_ERROR("Newton scaling failed (invalid error weight)!");

                        internalError = true;
                        return false;
                    }
                    if (linearSolver) {
                        if (!linearSolver->initialize(numOfSpecies)) {
                            internalError = true;
                            return false;
                        }
                        if (!linearSolver->factorize(P.data(), ip.data(), scaling_D1.data(),
                                                     scaling_D2.data(), numOfSpecies)) {
                            YABI_LOG_ERROR("ERROR: LuFactor Failed!");
                            internalError = true;
                            return false;
                        }
                    } else {
                        sign = linalg::GETRF(numOfSpecies, numOfSpecies, P.data(), numOfSpecies,
                                             ip.data());
                        if (sign != 0) {
                            YABI_LOG_ERROR("ERROR: LuFactor Failed! (dgetrf info=", sign, ")");
                            internalError = true;
                            return false;
                        }
                    }
                }
                // Sensitivity Generation

                if (taping) {
                    if (!direct) {
                        if (endTAPE->P.empty())
                            endTAPE->P.resize(numOfSpecies * numOfSpecies);
                        if (endTAPE->ip.empty())
                            endTAPE->ip.resize(numOfSpecies);
                        if (endTAPE->scaling_D1.empty())
                            endTAPE->scaling_D1.resize(numOfSpecies);
                        if (endTAPE->scaling_D2.empty())
                            endTAPE->scaling_D2.resize(numOfSpecies);

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
                stats.updateP++;
                wasNewPSet = true;
                if (useSparseAssembly) {
                    formPFromJ();
                    if (!calculateNewtonScalingSparse()) {
                        YABI_LOG_ERROR("Newton scaling failed (invalid error weight)!");

                        internalError = true;
                        return false;
                    }
                    if (useMatrixFree)
                        configureMatrixFreeOperator();
                    if (!linearSolver->factorizeCsc(P_csc, ip.data(), scaling_D1.data(),
                                                    scaling_D2.data(), numOfSpecies)) {
                        YABI_LOG_ERROR("ERROR: LuFactor Failed!");
                        internalError = true;
                        return false;
                    }
                } else if (useJFree) {
                    if (!configureJFreeMatrix(matrixFreeBase)) {
                        YABI_LOG_ERROR("ERROR: J-free matrix-free setup failed!");
                        internalError = true;
                        return false;
                    }
                } else {
                    for (j = 0; j < numOfSpecies; j++)
                        for (k = 0; k < numOfSpecies; k++)
                            if (j == k)
                                P[k * numOfSpecies + j] =
                                    1.0 - ((nordsieck.step_size_h / nordsieck.l_i[1]) *
                                           J[j * numOfSpecies + k]);
                            else
                                P[k * numOfSpecies + j] =
                                    -(nordsieck.step_size_h / nordsieck.l_i[1]) *
                                    J[j * numOfSpecies + k];

                    // LU factor (and overwrite) the matrix P.data() via pluggable solver
                    if (!calculateNewtonScaling()) {
                        YABI_LOG_ERROR("Newton scaling failed (invalid error weight)!");

                        internalError = true;
                        return false;
                    }
                    if (linearSolver) {
                        if (!linearSolver->initialize(numOfSpecies)) {
                            internalError = true;
                            return false;
                        }
                        if (!linearSolver->factorize(P.data(), ip.data(), scaling_D1.data(),
                                                     scaling_D2.data(), numOfSpecies)) {
                            YABI_LOG_ERROR("ERROR: LuFactor Failed!");
                            internalError = true;
                            return false;
                        }
                    } else {
                        sign = linalg::GETRF(numOfSpecies, numOfSpecies, P.data(), numOfSpecies,
                                             ip.data());
                        if (sign != 0) {
                            YABI_LOG_ERROR("ERROR: LuFactor Failed! (dgetrf info=", sign, ")");
                            internalError = true;
                            return false;
                        }
                    }
                }
                // Sensitivity Generation

                if (taping) {
                    if (!direct) {
                        if (endTAPE->P.empty())
                            endTAPE->P.resize(numOfSpecies * numOfSpecies);
                        if (endTAPE->ip.empty())
                            endTAPE->ip.resize(numOfSpecies);
                        if (endTAPE->scaling_D1.empty())
                            endTAPE->scaling_D1.resize(numOfSpecies);
                        if (endTAPE->scaling_D2.empty())
                            endTAPE->scaling_D2.resize(numOfSpecies);

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
                cppAD_argument[numOfSpecies] = nordsieck.t_i[0];
            if (RHS_type == 2)
                for (j = 0; j < sizeOfParameter; j++)
                    cppAD_argument[numOfSpecies + 1 + j] = parameter[j];

            if (taping && !direct) {
                if (endTAPE->CppAD_tape[newtonIterate - 1] == NULL) {
                    if (intermediateCppADFunIterator == cppADFunList.end()) {
                        std::unique_ptr<AbstractRHS<Scalar>> tempADFun(RHS->clone());
                        tempADFun->capacity_order(config.MINTAYLOR);
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
                b[j] = nordsieck.z_predictor[numOfSpecies + j];
            }
            Scalar blasTEMP1 = 1.0 / nordsieck.l_i[1];
            Scalar blasTEMP2 = -nordsieck.step_size_h / nordsieck.l_i[1];

            linalg::SCAL(numOfSpecies, blasTEMP1, b.data(), 1);
            linalg::AXPY(numOfSpecies, blasTEMP2, temp.data(), 1, b.data(), 1);
            linalg::AXPY(numOfSpecies, blasScalarONE, root.data(), 1, b.data(), 1);
            linalg::AXPY(numOfSpecies, blasScalarMinusONE, nordsieck.z_predictor.data(), 1, b.data(), 1);
            const Scalar newtonResidual =
                linalg::NRM2(numOfSpecies, b.data(), 1) / (Scalar)numOfSpecies;
            for (j = 0; j < numOfSpecies; j++)
                b[j] = -(Scalar)scalbn(b[j], scaling_D1[j]);

            if (linearSolver) {
                NewtonLinearizationContext<Scalar> linearization;
                linearization.iteration = newtonIterate;
                linearization.residual = newtonResidual;
                linearization.previous_residual = previousNewtonResidual;
                linearization.nonlinear_tolerance =
                    error_multiplier * config.NEWTON_CONVERGED * relative_error;
                linearSolver->setNewtonLinearizationContext(linearization);
            }

            if (useSparseAssembly || useJFree) {
                sign = linearSolver->solveCsc(P_csc, ip.data(), scaling_D1.data(),
                                              scaling_D2.data(), b.data(), numOfSpecies, 1);
            } else if (linearSolver) {
                sign = linearSolver->solve(P.data(), ip.data(), scaling_D1.data(),
                                           scaling_D2.data(), b.data(), numOfSpecies, 1);
            } else {
                sign = linalg::GETRS_NT(numOfSpecies, 1, P.data(), numOfSpecies, ip.data(), b.data(),
                                        numOfSpecies);
            }
            if (sign != 0) {
                YABI_LOG_ERROR("ERROR: linear solve failed (info=", sign, ")!");

                internalError = true;
                return false;
            }
            previousNewtonResidual = newtonResidual;

            for (j = 0; j < numOfSpecies; j++)
                temp[j] = (Scalar)scalbn(b[j], scaling_D2[j]);

            // Damped Newton (backtracking line search), opt-in (T45).  On an
            // ill-conditioned Newton system the full step can overshoot, so when
            // it would increase the BDF residual, backtrack (halve the correction)
            // until the residual decreases (Nocedal & Wright §11).  Gated to the
            // plain, non-taped state solve: the sensitivity directions are linear
            // (exact in one solve) and the tape replay reconstructs the full-step
            // sequence, so both are left on the historical path.  Off by default
            // → bit-identical to before.
            if (config.NEWTON_LINE_SEARCH && numOfDerivatives == 0 && !taping) {
                const Scalar residual_before = newtonResidual;
                Scalar lambda = (Scalar)1.0;
                std::vector<Scalar> f_trial(numOfSpecies);
                std::vector<Scalar> b_trial(numOfSpecies);
                std::vector<Scalar> root_trial(numOfSpecies);
                for (unsigned int bt = 0; bt < config.NEWTON_LINE_SEARCH_MAX; ++bt) {
                    for (j = 0; j < numOfSpecies; j++)
                        root_trial[j] = root[j] + lambda * temp[j];
                    for (j = 0; j < numOfSpecies; j++)
                        cppAD_argument[j] = root_trial[j];
                    if (RHS_type >= 1)
                        cppAD_argument[numOfSpecies] = nordsieck.t_i[0];
                    f_trial = RHS->Forward(0, cppAD_argument);

                    for (j = 0; j < numOfSpecies; j++)
                        b_trial[j] = nordsieck.z_predictor[numOfSpecies + j];
                    linalg::SCAL(numOfSpecies, blasTEMP1, b_trial.data(), 1);
                    linalg::AXPY(numOfSpecies, blasTEMP2, f_trial.data(), 1, b_trial.data(), 1);
                    linalg::AXPY(numOfSpecies, blasScalarONE, root_trial.data(), 1, b_trial.data(), 1);
                    linalg::AXPY(numOfSpecies, blasScalarMinusONE, nordsieck.z_predictor.data(), 1,
                                 b_trial.data(), 1);
                    const Scalar residual_trial =
                        linalg::NRM2(numOfSpecies, b_trial.data(), 1) / (Scalar)numOfSpecies;
                    if (residual_trial <= residual_before)
                        break;
                    lambda *= (Scalar)0.5;
                }
                if (lambda < (Scalar)1.0)
                    linalg::SCAL(numOfSpecies, lambda, temp.data(), 1);
            }

            linalg::AXPY(numOfSpecies, blasScalarONE, temp.data(), 1, root.data(), 1);
            linalg::AXPY(numOfSpecies, blasScalarONE, temp.data(), 1, nordsieck.corrector.data(), 1);

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

                    linalg::SCAL(numOfSpecies, blasTEMP1, b_derivatives[z].data(), 1);
                    linalg::AXPY(numOfSpecies, blasTEMP2, temp.data(), 1, b_derivatives[z].data(), 1);
                    linalg::AXPY(numOfSpecies, blasScalarONE, root_derivatives[z].data(), 1,
                                 b_derivatives[z].data(), 1);
                    linalg::AXPY(numOfSpecies, blasScalarMinusONE, z_predictor_derivatives[z].data(), 1,
                                 b_derivatives[z].data(), 1);
                    for (j = 0; j < numOfSpecies; j++)
                        b_derivatives[z][j] = -(Scalar)scalbn(b_derivatives[z][j], scaling_D1[j]);

                    sign =
                        linearSolver->solve(P.data(), ip.data(), scaling_D1.data(),
                                            scaling_D2.data(), b_derivatives[z].data(), numOfSpecies, 1);

                    for (j = 0; j < numOfSpecies; j++)
                        temp[j] = (Scalar)scalbn(b_derivatives[z][j], scaling_D2[j]);

                    linalg::AXPY(numOfSpecies, blasScalarONE, temp.data(), 1, root_derivatives[z].data(), 1);
                    linalg::AXPY(numOfSpecies, blasScalarONE, temp.data(), 1, corrector_derivatives[z].data(), 1);
                }
            }

            if (newtonIterate == 1) {
                convergenceResidual[0] = linalg::NRM2(numOfSpecies, b.data(), 1) / numOfSpecies;

                minimalRate =
                    (Scalar)pow((error_multiplier * config.NEWTON_CONVERGED * relative_error) /
                                    convergenceResidual[0],
                                1.0 / (config.MAXNEWTONITERATION - 1));

                if (numOfDerivatives > 0 && !direct) {
                    for (z = 0; z < numOfDerivatives; z++) {
                        for (i = 0; i < numOfSpecies; i++)
                            temp[i] = (Scalar)scalbn(
                                scalbn(b_derivatives[z][i], scaling_D2[i]),
                                -(int)floor(logb(error_weights_derivatives[z][i]) + 0.5));

                        convergenceResidual_derivatives[z][0] =
                            linalg::NRM2(numOfSpecies, temp.data(), 1) / numOfSpecies;

                        Scalar tempRATE = 0;
                        tempRATE = (Scalar)pow((error_multiplier * config.NEWTON_CONVERGED *
                                                relative_error_derivatives[z]) /
                                                   convergenceResidual_derivatives[z][0],
                                               1.0 / (config.MAXNEWTONITERATION - 1));
                        if (tempRATE < minimalRate)
                            minimalRate = tempRATE;
                    }
                }
                if (config.NEWTON_SAFETY > 0)
                    minimalRate = config.NEWTON_SAFETY;
                bool pass = true;
                if (numOfDerivatives > 0 && !direct)
                    for (z = 0; z < numOfDerivatives; z++) {
                        for (j = 0; j < numOfSpecies; j++)
                            tempMACHINEprecision[j] = (Scalar)scalbn(
                                root_derivatives[z][j],
                                -(int)floor(logb(error_weights_derivatives[z][j]) + 0.5));

                        Scalar tempMACHINEprecisionNorm =
                            linalg::NRM2(numOfSpecies, tempMACHINEprecision.data(), 1) / numOfSpecies;

                        if (!(convergenceResidual_derivatives[z][0] <
                                  error_multiplier * config.NEWTON_CONVERGED *
                                      relative_error_derivatives[z] ||
                              convergenceResidual_derivatives[z][0] <
                                  2.0 * config.MACHINEPRECISION * tempMACHINEprecisionNorm))
                            pass = false;
                    }

                for (j = 0; j < numOfSpecies; j++)
                    tempMACHINEprecision[j] = (Scalar)scalbn(root[j], -scaling_D2[j]);

                Scalar tempMACHINEprecisionNorm =
                    linalg::NRM2(numOfSpecies, tempMACHINEprecision.data(), 1) / numOfSpecies;

                if (pass == true)
                    if (convergenceResidual[0] <
                            error_multiplier * config.NEWTON_CONVERGED * relative_error ||
                        convergenceResidual[0] <
                            2.0 * config.MACHINEPRECISION * tempMACHINEprecisionNorm) {
                        convergenceIndicator = (Scalar)0.1;
                        if (taping) {
                            if (direct) {
                                if (endTAPE->endSolution.empty())
                                    endTAPE->endSolution.resize(numOfSpecies);
                                for (i = 0; i < numOfSpecies; i++)
                                    endTAPE->endSolution[i] = root[i];

                                endTAPE->timePoint = nordsieck.t_i[0];
                            } else
                                endTAPE->NewtonIterates = newtonIterate;
                        }
                        // DIRECT SIMULTANEOUS MODE

                        if (numOfDerivatives > 0 && direct) {
                            std::vector<Scalar> df(numOfSpecies);
                            std::vector<Scalar> P_(numOfSpecies * numOfSpecies);
                            std::vector<int> ip_(numOfSpecies);
                            std::vector<long> scaling_D1_(numOfSpecies);
                            std::vector<long> scaling_D2_(numOfSpecies);

                            // CALCULATION OF JACOBIAN

                            for (j = 0; j < numOfSpecies; j++)
                                cppAD_argument[j] = root[j];
                            if (RHS_type >= 1)
                                cppAD_argument[numOfSpecies] = nordsieck.t_i[0];
                            if (RHS_type == 2)
                                for (j = 0; j < sizeOfParameter; j++)
                                    cppAD_argument[numOfSpecies + 1 + j] = parameter[j];

                            std::vector<Scalar> J_(numOfSpecies * numOfSpecies);

                            if (taping) {
                                if (endTAPE->direct_fun == NULL) {
                                    if (intermediateCppADFunIterator == cppADFunList.end()) {
                                        std::unique_ptr<AbstractRHS<Scalar>> tempADFun(
                                            RHS->clone());
                                        tempADFun->capacity_order(config.MINTAYLOR);
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
                                            YABI_LOG_ERROR(
                                                "element of jacobian is nan or inf! (direct "
                                                "sensitivities simultaneous1)");

                                            internalError = true;
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
                                            YABI_LOG_ERROR(
                                                "element of jacobian is nan or inf! (direct "
                                                "sensitivities)");

                                            internalError = true;
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
                                            1 - ((nordsieck.step_size_h / nordsieck.l_i[1]) *
                                                 J_[j * numOfSpecies + k]);
                                    else
                                        P_[k * numOfSpecies + j] =
                                            -(nordsieck.step_size_h / nordsieck.l_i[1]) *
                                            J_[j * numOfSpecies + k];
                                }

                            for (j = 0; j < numOfSpecies; j++)
                                scaling_D2_[j] = 0;

                            if (taping) {
                                if (endTAPE->scaling_D1.empty())
                                    endTAPE->scaling_D1.resize(numOfSpecies);
                                if (endTAPE->scaling_D2.empty())
                                    endTAPE->scaling_D2.resize(numOfSpecies);

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
                                        " of the Newton matrix (direct sensitivities, column sum ",
                                        sum, ")!");

                                    internalError = true;
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

                            // LU factor (and overwrite) the matrix P.data() and ip.data() and jp
                            int info;
                            info = linalg::GETRF(numOfSpecies, numOfSpecies, P_.data(), numOfSpecies, ip_.data());
                            if (info != 0) {
                                YABI_LOG_ERROR("ERROR: LuFactor Failed! (dgetrf info=", info, ")");
                                internalError = true;
                                return false;
                            }

                            if (taping) {
                                if (endTAPE->P.empty())
                                    endTAPE->P.resize(numOfSpecies * numOfSpecies);
                                if (endTAPE->ip.empty())
                                    endTAPE->ip.resize(numOfSpecies);

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
                                        cppAD_argument[numOfSpecies + 1 + j] =
                                            p_seed_derivatives[z][j];
                                }

                                if (taping)
                                    df = endTAPE->direct_fun->Forward(1, cppAD_argument);
                                else
                                    df = RHS->Forward(1, cppAD_argument);

                                for (j = 0; j < numOfSpecies; j++)
                                    b[j] = -(Scalar)scalbn(
                                        ((-z_predictor_derivatives[z][j]) -
                                         (((nordsieck.step_size_h / nordsieck.l_i[1]) * df[j]) -
                                          (z_predictor_derivatives[z][numOfSpecies + j] /
                                           nordsieck.l_i[1]))),
                                        scaling_D1_[j]);

                                linalg::GETRS_NT(numOfSpecies, 1, P_.data(), numOfSpecies, ip_.data(), b.data(),
                                                 numOfSpecies);

                                for (i = 0; i < numOfSpecies; i++)
                                    corrector_derivatives[z][i] =
                                        (Scalar)scalbn(b[i], scaling_D2_[i]) -
                                        z_predictor_derivatives[z][i];
                            }
                        }

                        return true;
                    }
            } else {
                convergenceResidual[1] = convergenceResidual[0];
                convergenceResidual[0] = linalg::NRM2(numOfSpecies, b.data(), 1) / numOfSpecies;

                if (numOfDerivatives > 0 && !direct) {
                    for (z = 0; z < numOfDerivatives; z++) {
                        convergenceResidual_derivatives[z][1] =
                            convergenceResidual_derivatives[z][0];

                        for (i = 0; i < numOfSpecies; i++)
                            temp[i] = (Scalar)scalbn(
                                scalbn(b_derivatives[z][i], scaling_D2[i]),
                                -(int)floor(logb(error_weights_derivatives[z][i]) + 0.5));
                        convergenceResidual_derivatives[z][0] =
                            linalg::NRM2(numOfSpecies, temp.data(), 1) / numOfSpecies;
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
                            linalg::NRM2(numOfSpecies, tempMACHINEprecision.data(), 1) / numOfSpecies;

                        if (!(ratio * convergenceResidual_derivatives[z][0] <
                                  error_multiplier * config.NEWTON_CONVERGED *
                                      relative_error_derivatives[z] ||
                              convergenceResidual_derivatives[z][0] <
                                  error_multiplier * config.NEWTON_CONVERGED *
                                      relative_error_derivatives[z] ||
                              (ratio < config.ACCEPT_NEWTON_CONVERGENCE_RATE &&
                               contructionRateFulfilled) ||
                              convergenceResidual_derivatives[z][0] <
                                  2.0 * config.MACHINEPRECISION * tempMACHINEprecisionNorm))
                            pass = false;
                    }

                for (j = 0; j < numOfSpecies; j++)
                    tempMACHINEprecision[j] = (Scalar)scalbn(root[j], -scaling_D2[j]);
                Scalar tempMACHINEprecisionNorm =
                    linalg::NRM2(numOfSpecies, tempMACHINEprecision.data(), 1) / numOfSpecies;

                if (pass == true)
                    if (ratio * convergenceResidual[0] <
                            error_multiplier * config.NEWTON_CONVERGED * relative_error ||
                        convergenceResidual[0] <
                            error_multiplier * config.NEWTON_CONVERGED * relative_error ||
                        (ratio < config.ACCEPT_NEWTON_CONVERGENCE_RATE &&
                         contructionRateFulfilled) ||
                        convergenceResidual[0] <
                            2.0 * config.MACHINEPRECISION * tempMACHINEprecisionNorm) {
                        convergenceIndicator = (Scalar)0.1;
                        if (taping) {
                            if (direct) {
                                if (endTAPE->endSolution.empty())
                                    endTAPE->endSolution.resize(numOfSpecies);
                                for (i = 0; i < numOfSpecies; i++)
                                    endTAPE->endSolution[i] = root[i];

                                endTAPE->timePoint = nordsieck.t_i[0];
                            } else
                                endTAPE->NewtonIterates = newtonIterate;
                        }

                        if (numOfDerivatives > 0 && direct) {
                            std::vector<Scalar> df(numOfSpecies);
                            std::vector<Scalar> P_(numOfSpecies * numOfSpecies);
                            std::vector<int> ip_(numOfSpecies);
                            std::vector<long> scaling_D1_(numOfSpecies);
                            std::vector<long> scaling_D2_(numOfSpecies);

                            // CALCULATION OF JACOBIAN

                            for (j = 0; j < numOfSpecies; j++)
                                cppAD_argument[j] = root[j];
                            if (RHS_type >= 1)
                                cppAD_argument[numOfSpecies] = nordsieck.t_i[0];
                            if (RHS_type == 2)
                                for (j = 0; j < sizeOfParameter; j++)
                                    cppAD_argument[numOfSpecies + 1 + j] = parameter[j];

                            std::vector<Scalar> J_(numOfSpecies * numOfSpecies);

                            if (taping) {
                                if (endTAPE->direct_fun == NULL) {
                                    if (intermediateCppADFunIterator == cppADFunList.end()) {
                                        std::unique_ptr<AbstractRHS<Scalar>> tempADFun(
                                            RHS->clone());
                                        tempADFun->capacity_order(config.MINTAYLOR);
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
                                            YABI_LOG_ERROR(
                                                "element of jacobian is nan or inf! (direct "
                                                "sensitivities simultaneous2)");

                                            internalError = true;
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
                                            YABI_LOG_ERROR(
                                                "element of jacobian is nan or inf! (direct "
                                                "sensitivities)");

                                            internalError = true;
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
                                            1.0 - ((nordsieck.step_size_h / nordsieck.l_i[1]) *
                                                   J_[j * numOfSpecies + k]);
                                    else
                                        P_[k * numOfSpecies + j] =
                                            -(nordsieck.step_size_h / nordsieck.l_i[1]) *
                                            J_[j * numOfSpecies + k];
                                }

                            for (j = 0; j < numOfSpecies; j++)
                                scaling_D2_[j] = 0;

                            if (taping) {
                                if (endTAPE->scaling_D1.empty())
                                    endTAPE->scaling_D1.resize(numOfSpecies);
                                if (endTAPE->scaling_D2.empty())
                                    endTAPE->scaling_D2.resize(numOfSpecies);

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
                                        " of the Newton matrix (direct sensitivities, column sum ",
                                        sum, ")!");

                                    internalError = true;
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

                            // LU factor (and overwrite) the matrix P.data() and ip.data() and jp
                            int info;
                            info = linalg::GETRF(numOfSpecies, numOfSpecies, P_.data(), numOfSpecies, ip_.data());
                            if (info != 0) {
                                YABI_LOG_ERROR("ERROR: LuFactor Failed! (dgetrf info=", info, ")");
                                internalError = true;
                                return false;
                            }

                            if (taping) {
                                if (endTAPE->P.empty())
                                    endTAPE->P.resize(numOfSpecies * numOfSpecies);
                                if (endTAPE->ip.empty())
                                    endTAPE->ip.resize(numOfSpecies);

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
                                        cppAD_argument[numOfSpecies + 1 + j] =
                                            p_seed_derivatives[z][j];
                                }

                                if (taping)
                                    df = endTAPE->direct_fun->Forward(1, cppAD_argument);
                                else
                                    df = RHS->Forward(1, cppAD_argument);

                                for (j = 0; j < numOfSpecies; j++)
                                    b[j] = -(Scalar)scalbn(
                                        ((-z_predictor_derivatives[z][j]) -
                                         (((nordsieck.step_size_h / nordsieck.l_i[1]) * df[j]) -
                                          (z_predictor_derivatives[z][numOfSpecies + j] /
                                           nordsieck.l_i[1]))),
                                        scaling_D1_[j]);

                                linalg::GETRS_NT(numOfSpecies, 1, P_.data(), numOfSpecies, ip_.data(), b.data(),
                                                 numOfSpecies);

                                for (i = 0; i < numOfSpecies; i++)
                                    corrector_derivatives[z][i] =
                                        (Scalar)scalbn(b[i], scaling_D2_[i]) -
                                        z_predictor_derivatives[z][i];
                            }
                        }

                        return true;
                    }
                if (ratio > minimalRate && ratio > config.ACCEPT_NEWTON_CONVERGENCE_RATE) {
                    if (setJacobian == true) {
                        convergenceIndicator = -1.0;
                        return true;
                    }
                    if (setNewP == true && setJacobian == false) {
                        newtonIterate = 1;
                        contructionRateFulfilled = false;
                        setJacobian = true;
                        for (i = 0; i < numOfSpecies; i++) {
                            nordsieck.corrector[i] = 0.0;
                            root[i] = nordsieck.z_predictor[i];
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
                            root[i] = nordsieck.z_predictor[i];
                            nordsieck.corrector[i] = 0.0;
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
                if (ratio < config.ACCEPT_NEWTON_CONVERGENCE_RATE) {
                    contructionRateFulfilled = true;
                }
            }
            newtonIterate += 1;
        }

        convergenceIndicator = -1;

        return true;
    }

    // Local-error acceptance, step/order bookkeeping on success, and the
    // order reduction on rejection.  Returns
    // 0 repeat step, 1 next step, 2 output, 3/4 fatal.
    unsigned int acceptCorrectorVector() {
        Scalar blasScalarONE = (Scalar)1.0;
        if (convergenceIndicator < 0 &&
            !(nordsieck.step_size_h < 4.0 * config.MACHINEPRECISION * nordsieck.t_i[1])) {
            nordsieck.num_of_steps_per_order = 0;

            if (newtonFAIL > config.allowedNEWTON)
                return 3;
            newtonFAIL += 1;
            stats.rejectedNewtonSteps++;
            Scalar eta = config.NEWTON_STEP_REDUCTION;
            nordsieck.step_size_h = nordsieck.step_size_h * eta;

            if (nordsieck.step_size_h < 4.0 * config.MACHINEPRECISION * nordsieck.t_i[1]) {
                YABI_LOG_ERROR("relaxing step size!");

                nordsieck.step_size_h = 4.0 * config.MACHINEPRECISION * nordsieck.t_i[1];
            }
            nordsieck.t_i[0] = nordsieck.step_size_h + nordsieck.t_i[1];
            return 0; // repeat step
        }
        if (convergenceIndicator < 0 &&
            (nordsieck.step_size_h < 4.0 * config.MACHINEPRECISION * nordsieck.t_i[1])) {
            YABI_LOG_ERROR("WARNING: accept newton iteration due to minimal step size");
        }
        unsigned int i, z;
        bool pass = true;

        Scalar temp1;
        temp1 = 1.0;
        for (i = 1; i < nordsieck.order_corrector; i++)
            temp1 *= nordsieck.stepSizeSet_new[i] /
                     (nordsieck.stepSizeSet_new[i] - nordsieck.step_size_h);
        temp1 += 1.0;
        // |LE| = |e| / (l_i[1] * (1 + product)); see the local-error
        // estimator in the dissertation (alpha_0 = -l_i[1]).
        temp1 *= nordsieck.l_i[1];

        std::vector<Scalar> temp2(numOfSpecies);

        for (i = 0; i < numOfSpecies; i++)
            temp2[i] = nordsieck.z_predictor[i];
        linalg::AXPY(numOfSpecies, blasScalarONE, nordsieck.corrector.data(), 1, temp2.data(), 1);
        StepController<Scalar>::calculateErrorWeights(numOfSpecies, config.ERROR_WEIGHTS_INDEX,
                                                      error_weights.data(), temp2.data(),
                                                      absolut_error.data(), relative_error);
        for (i = 0; i < numOfSpecies; i++)
            temp2[i] /= error_weights[i];
        reachableError =
            2 * config.MACHINEPRECISION * linalg::NRM2(numOfSpecies, temp2.data(), 1) / numOfSpecies;
        if (reachableError < relative_error)
            reachableError = relative_error;

        else
            YABI_LOG_ERROR("WARNING: relax error threshold due to limiting machine precision");

        for (i = 0; i < numOfSpecies; i++)
            temp2[i] = nordsieck.corrector[i] / temp1 / error_weights[i];

        Scalar normWeightedLocalTruncError = linalg::NRM2(numOfSpecies, temp2.data(), 1) / numOfSpecies;
        std::vector<Scalar> normWeightedLocalTruncError_derivatives;

        if (normWeightedLocalTruncError > reachableError)
            pass = false;

        if (numOfDerivatives > 0) {
            normWeightedLocalTruncError_derivatives.resize(numOfDerivatives);
            for (z = 0; z < numOfDerivatives; z++) {
                for (i = 0; i < numOfSpecies; i++)
                    temp2[i] = z_predictor_derivatives[z][i];
                linalg::AXPY(numOfSpecies, blasScalarONE, corrector_derivatives[z].data(), 1,
                             temp2.data(), 1);
                StepController<Scalar>::calculateErrorWeights(
                    numOfSpecies, config.ERROR_WEIGHTS_INDEX, error_weights_derivatives[z].data(),
                    temp2.data(), absolut_error_derivatives[z], relative_error_derivatives[z]);
                for (i = 0; i < numOfSpecies; i++)
                    temp2[i] /= error_weights_derivatives[z][i];
                reachableError_derivatives[z] = 2 * config.MACHINEPRECISION *
                                                linalg::NRM2(numOfSpecies, temp2.data(), 1) /
                                                numOfSpecies;
                if (reachableError_derivatives[z] < relative_error_derivatives[z])
                    reachableError_derivatives[z] = relative_error_derivatives[z];

                else
                    YABI_LOG_ERROR(
                        "WARNING: relax error threshold due to limiting machine precision");

                for (i = 0; i < numOfSpecies; i++)
                    temp2[i] =
                        corrector_derivatives[z][i] / temp1 / error_weights_derivatives[z][i];

                normWeightedLocalTruncError_derivatives[z] =
                    linalg::NRM2(numOfSpecies, temp2.data(), 1) / numOfSpecies;

                if (normWeightedLocalTruncError_derivatives[z] > reachableError_derivatives[z])
                    pass = false;
            }
        }
        if (nordsieck.step_size_h < 4.0 * config.MACHINEPRECISION * nordsieck.t_i[1]) {
            pass = true;

            std::cout << "WARNING: accept step due to limiting minimal step size";
        }
        if (pass) {
            stats.acceptedSteps++;

            Scalar mu =
                config.ORDER_STAY_SAVETY_STEP * (reachableError / normWeightedLocalTruncError);
            Scalar sumTemp = 0.0;
            for (i = 1; i <= nordsieck.order_corrector; i++)
                sumTemp += (1.0 / i);

            minimalEta = sumTemp;
            sumTemp = 0.0;
            for (i = 0; i < nordsieck.order_corrector; i++)
                sumTemp += (nordsieck.step_size_h / nordsieck.stepSizeSet_new[i]);
            minimalEta /= sumTemp;
            for (i = 2; i <= nordsieck.order_corrector; i++)
                minimalEta *= nordsieck.stepSizeSet_new[i - 1] / nordsieck.step_size_h / i;
            minimalEta = pow(minimalEta, 1.0 / (nordsieck.order_corrector + 1));
            minimalEta *= pow(mu, 1.0 / (nordsieck.order_corrector + 1));

            if (numOfDerivatives > 0) {
                Scalar minimalEta_derivatives = 0.0;
                for (z = 0; z < numOfDerivatives; z++) {
                    mu = config.ORDER_STAY_SAVETY_STEP *
                         (reachableError_derivatives[z] /
                          normWeightedLocalTruncError_derivatives[z]);
                    sumTemp = 0.0;
                    for (i = 1; i <= nordsieck.order_corrector; i++)
                        sumTemp += (1.0 / i);
                    minimalEta_derivatives = sumTemp;
                    sumTemp = 0;
                    for (i = 0; i < nordsieck.order_corrector; i++)
                        sumTemp += (nordsieck.step_size_h / nordsieck.stepSizeSet_new[i]);
                    minimalEta_derivatives /= sumTemp;
                    for (i = 2; i <= nordsieck.order_corrector; i++)
                        minimalEta_derivatives *=
                            nordsieck.stepSizeSet_new[i - 1] / nordsieck.step_size_h / i;
                    minimalEta_derivatives =
                        pow(minimalEta_derivatives, 1.0 / (nordsieck.order_corrector + 1));
                    minimalEta_derivatives *= pow(mu, 1.0 / (nordsieck.order_corrector + 1));
                    if (minimalEta > minimalEta_derivatives)
                        minimalEta = minimalEta_derivatives;
                }
            }

            if (nordsieck.order_corrector == 1)
                if (minimalEta > config.RESTRICT_ORDER_1_RATIO)
                    minimalEta = config.RESTRICT_ORDER_1_RATIO;

            if (nordsieck.order_corrector == 2)
                if (minimalEta > config.RESTRICT_ORDER_2_RATIO)
                    minimalEta = config.RESTRICT_ORDER_2_RATIO;

            if (nordsieck.order_corrector == 3)
                if (minimalEta > config.RESTRICT_ORDER_3_RATIO)
                    minimalEta = config.RESTRICT_ORDER_3_RATIO;

            if (nordsieck.order_corrector == 4)
                if (minimalEta > config.RESTRICT_ORDER_4_RATIO)
                    minimalEta = config.RESTRICT_ORDER_4_RATIO;

            if (nordsieck.order_corrector == 5)
                if (minimalEta > config.RESTRICT_ORDER_5_RATIO)
                    minimalEta = config.RESTRICT_ORDER_5_RATIO;

            if (nordsieck.order_corrector == 6)
                if (minimalEta > config.RESTRICT_ORDER_6_RATIO)
                    minimalEta = config.RESTRICT_ORDER_6_RATIO;

            if (taping) {
                endTAPE->timePoint = nordsieck.t_i[0];
                endTAPE->stepSize = nordsieck.step_size_h;
            }

            // ACCEPTING STEP
            nordsieck.num_of_steps_per_order += 1;
            nordsieck.t_i[1] = nordsieck.t_i[0];
            nordsieck.t_i[0] = nordsieck.t_i[1] + nordsieck.step_size_h;
            if (nordsieck.t_i[1] >= t_ende - 2 * config.MACHINEPRECISION * t_ende)
                return 2; // calculate OutPUT-Vector
            else
                return 1; // next Step
        } else {
            // NOT ACCEPTING STEP

            nordsieck.num_of_steps_per_order = 0;

            if (stepFAIL > config.allowedSTEPS)
                return 4;

            stats.rejectedSteps++;

            Scalar mu = config.STEP_NOT_ACCEPTED_SAVETY_STEP *
                        (reachableError / normWeightedLocalTruncError);
            Scalar eta = 2.0;
            if (mu >= 0.05 && mu <= 1.0) {
                eta = sqrt(pow(mu, Scalar(0.5 + (1.0 / (nordsieck.order_corrector + 1)))));
            }
            if (mu < 0.05) {
                Scalar nu = 1;
                for (i = 1; i < nordsieck.order_corrector; i++)
                    nu *= nordsieck.stepSizeSet_new[i] / nordsieck.step_size_h;
                for (i = 1; i < nordsieck.order_corrector; i++)
                    nu /= (nordsieck.stepSizeSet_new[i] / nordsieck.step_size_h) - 1;
                nu /= nordsieck.l_i[1];
                nu = sqrt(nu);
                nu -= 1;
                nu *= pow(mu, Scalar(nordsieck.order_corrector - 1) /
                                  Scalar(2 * nordsieck.order_corrector + 2));
                if (nu > 1)
                    nu = 1;
                eta = nu * pow(mu, 1.0 / Scalar(nordsieck.order_corrector + 1)) +
                      (1 - nu) * pow(mu, 0.5);
            }
            if (mu > 1.0)
                eta = 2.0;

            if (numOfDerivatives > 0) {
                Scalar eta_derivatives = 0.0;
                for (z = 0; z < numOfDerivatives; z++) {
                    mu = config.STEP_NOT_ACCEPTED_SAVETY_STEP *
                         (reachableError_derivatives[z] /
                          normWeightedLocalTruncError_derivatives[z]);
                    if (mu >= 0.05 && mu <= 1.0) {
                        eta_derivatives =
                            sqrt(pow(mu, Scalar(0.5 + (1.0 / (nordsieck.order_corrector + 1)))));
                    }
                    if (mu < 0.05) {
                        Scalar nu = 1;
                        for (i = 1; i < nordsieck.order_corrector; i++)
                            nu *= nordsieck.stepSizeSet_new[i] / nordsieck.step_size_h;
                        for (i = 1; i < nordsieck.order_corrector; i++)
                            nu /= (nordsieck.stepSizeSet_new[i] / nordsieck.step_size_h) - 1;
                        nu /= nordsieck.l_i[1];
                        nu = sqrt(nu);
                        nu -= 1.0;
                        nu *= pow(mu, Scalar(nordsieck.order_corrector - 1) /
                                          Scalar(2 * nordsieck.order_corrector + 2));
                        if (nu > 1.0)
                            nu = 1.0;
                        eta_derivatives =
                            nu * pow(mu, 1.0 / Scalar(nordsieck.order_corrector + 1)) +
                            (1.0 - nu) * pow(mu, 0.5);
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

            // Order reduction on rejection (Gear 1971; Byrne & Hindmarsh 1975): on a rejected
            // step, estimate the local truncation error at order-1 from the highest Nordsieck term
            // and, if dropping the order gives a smaller (or already acceptable) error, reduce the
            // order instead of only shrinking the step. The reduction is recorded in the tape
            // (orderDOWN marker below) so the sensitivity replay reconstructs it.
            if (nordsieck.order_corrector > config.MINORDER) {
                Scalar c_down = -1.0;
                for (i = 0; i < nordsieck.order_corrector - 1; i++)
                    c_down *= nordsieck.stepSizeSet_new[i] / nordsieck.step_size_h;
                c_down /=
                    (nordsieck.l_i[1] - (nordsieck.step_size_h /
                                         nordsieck.stepSizeSet_new[nordsieck.order_corrector - 1]));
                std::vector<Scalar> err_down(numOfSpecies);
                for (i = 0; i < numOfSpecies; i++)
                    err_down[i] =
                        nordsieck.z_corrector[numOfSpecies * nordsieck.order_corrector + i] *
                        c_down / error_weights[i];
                Scalar norm_err_down = linalg::NRM2(numOfSpecies, err_down.data(), 1) / numOfSpecies;
                if (norm_err_down < normWeightedLocalTruncError ||
                    norm_err_down <= reachableError) {
                    nordsieck.order_corrector -= 1;
                    stepFAIL = 0;
                    YABI_LOG_WARN("reducing BDF order to ", nordsieck.order_corrector,
                                  " (order-1 error estimate)");

                    // Record the mid-step order reduction in the sensitivity tape so the replay
                    // reconstructs the correct order. Insert an orderDOWN marker entry immediately
                    // before the current step entry (mirrors prepare_next_step's order-change
                    // marker).
                    if (taping && endTAPE != NULL) {
                        ReplayTape<Scalar> *marker = replayTapeStore.create();
                        marker->CppAD_tape.assign(config.MAXNEWTONITERATION, nullptr);
                        marker->orderDOWN = true;
                        marker->stepSize = nordsieck.step_size_h;
                        marker->timePoint = nordsieck.t_i[1];
                        marker->forwardLINK = endTAPE;
                        marker->backLINK = endTAPE->backLINK;
                        if (endTAPE->backLINK != nullptr)
                            endTAPE->backLINK->forwardLINK = marker;
                        else
                            startTAPE = marker;
                        endTAPE->backLINK = marker;
                    }
                }
            }

            if (eta > 1.0) {
                eta = config.STEP_NOT_ACCEPTED_REDUCTION;

                YABI_LOG_ERROR("force step size reduction!");
            }
            nordsieck.step_size_h = nordsieck.step_size_h * eta;
            if (nordsieck.step_size_h < 4.0 * config.MACHINEPRECISION * nordsieck.t_i[1]) {
                YABI_LOG_ERROR("relaxing step size!");

                nordsieck.step_size_h = 4.0 * config.MACHINEPRECISION * nordsieck.t_i[1];
            }
            nordsieck.t_i[0] = nordsieck.step_size_h + nordsieck.t_i[1];

            return 0; // repeat step
        }
        return true;
    }

  private:
    // ---- Native sparse Newton path (T31) ----
    //
    // Sparse assembly is used only for a plain state solve: sensitivity
    // generation still records a dense P on the replay tape, and is left on the
    // historical dense path.  Everything below is byte-for-byte the same
    // algebra as the dense branches, just applied to CSC storage.
    bool sparsePathAvailable() const {
        return sparseAssembly && numOfDerivatives == 0 && !taping && linearSolver &&
               linearSolver->supportsCsc();
    }

    bool prepareSparsePattern() {
        if (sparsePatternReady)
            return true;
        SparsityPattern detected;
        if (!RHS->jacobianSparsity(detected))
            return false;
        if (detected.rows != numOfSpecies || detected.cols != numOfSpecies || detected.empty())
            return false;
        sparsePattern = detected;
        if (!J_csc.assignPattern(sparsePattern, false))
            return false;
        if (!P_csc.assignPattern(sparsePattern, true))
            return false;
        sparsePatternReady = true;
        return true;
    }

    // P = I - (h/l1) * J, in the same column-major orientation as the dense
    // path.  The diagonal is structural in P even when J has no diagonal entry.
    void formPFromJ() {
        const Scalar factor = nordsieck.step_size_h / nordsieck.l_i[1];
        for (std::size_t j = 0; j < (std::size_t)numOfSpecies; ++j) {
            int pj = P_csc.col_ptr[j];
            const int pend = P_csc.col_ptr[j + 1];
            int jj = J_csc.col_ptr[j];
            const int jend = J_csc.col_ptr[j + 1];
            for (; pj < pend; ++pj) {
                const int row = P_csc.row_idx[(std::size_t)pj];
                while (jj < jend && J_csc.row_idx[(std::size_t)jj] < row)
                    ++jj;
                const Scalar jvalue = (jj < jend && J_csc.row_idx[(std::size_t)jj] == row)
                                          ? J_csc.val[(std::size_t)jj]
                                          : (Scalar)0.0;
                P_csc.val[(std::size_t)pj] =
                    (row == (int)j) ? ((Scalar)1.0 - factor * jvalue) : (-factor * jvalue);
            }
        }
    }

    // ---- Matrix-free Newton path (T32) ----
    //
    // The state Newton solve may use a Krylov solver that only needs P*v.
    // Sparse J/P are still assembled for the preconditioner and for the Newton
    // scaling, but the Krylov iteration itself never sees a stored matrix J.
    // The path is restricted to a plain state solve: sensitivity equations and
    // the replay tape still require the explicit J/H blocks.
    bool matrixFreePathAvailable() const {
        return matrixFreeAssembly && numOfDerivatives == 0 && !taping && linearSolver &&
               linearSolver->supportsMatrixFree();
    }

    // P_scaled * v = D1 * (I - (h/l1)*J(base)) * D2 * v, evaluated entirely
    // through the RHS Jacobian-vector product.  The base point is captured when
    // J is assembled (not when P is refreshed), so the operator and the
    // preconditioner always describe the same Jacobian.
    class NewtonMatrixFreeOperator : public MatrixFreeOperator<Scalar> {
      public:
        NewtonMatrixFreeOperator(AbstractRHS<Scalar> &rhs, unsigned int n,
                                 const std::vector<Scalar> &base,
                                 const std::vector<long> &scaling_D1,
                                 const std::vector<long> &scaling_D2)
            : rhs_(rhs), n_(n), base_(base), scaling_D1_(scaling_D1),
              scaling_D2_(scaling_D2) {}

        void setFactor(Scalar factor) { factor_ = factor; }

        bool apply(const Scalar *x, Scalar *y) const override {
            const std::size_t domain = rhs_.Domain();
            if (base_.size() != domain || domain < n_)
                return false;

            std::vector<Scalar> direction(domain, Scalar(0));
            std::vector<Scalar> scaled(n_, Scalar(0));
            for (unsigned int j = 0; j < n_; ++j) {
                scaled[j] = (Scalar)std::scalbn(x[j], (int)scaling_D2_[j]);
                direction[j] = scaled[j];
            }

            std::vector<Scalar> jv;
            if (!rhs_.jacobianVectorProduct(base_, direction, jv) || jv.size() < n_)
                return false;

            for (unsigned int i = 0; i < n_; ++i) {
                const Scalar w = scaled[i] - factor_ * jv[i];
                y[i] = (Scalar)std::scalbn(w, (int)scaling_D1_[i]);
            }
            return true;
        }

      private:
        AbstractRHS<Scalar> &rhs_;
        unsigned int n_;
        const std::vector<Scalar> &base_;
        const std::vector<long> &scaling_D1_;
        const std::vector<long> &scaling_D2_;
        Scalar factor_ = Scalar(0);
    };

    void configureMatrixFreeOperator() {
        if (!matrixFreeOperator)
            matrixFreeOperator.reset(new NewtonMatrixFreeOperator(
                *RHS, numOfSpecies, matrixFreeBase, scaling_D1, scaling_D2));
        matrixFreeOperator->setFactor(nordsieck.step_size_h / nordsieck.l_i[1]);
        linearSolver->setMatrixFreeOperator(matrixFreeOperator.get());
    }

    bool calculateNewtonScalingSparse() {
        unsigned int i, j;
        for (j = 0; j < numOfSpecies; j++) {
            if (!(error_weights[j] > 0.0) || !std::isfinite(error_weights[j])) {
                YABI_LOG_ERROR("ERROR: non-positive or non-finite error weight for species ", j,
                               " (", error_weights[j], ")!");

                internalError = true;
                return false;
            }
            scaling_D2[j] = (int)std::floor(std::logb(error_weights[j]) + 0.5);
        }

        // D2 scales columns, exactly as the dense loop P[j*n+i] *= 2^D2[j].
        P_csc.scaleColumns(scaling_D2);

        // D1 is derived from the row sums and then scales rows.
        std::vector<Scalar> rowSums;
        P_csc.rowSums(rowSums);
        for (i = 0; i < numOfSpecies; i++) {
            if (!(rowSums[i] > 0.0) || !std::isfinite(rowSums[i])) {
                YABI_LOG_ERROR("ERROR: cannot scale column ", i,
                               " of the Newton matrix (column sum ", rowSums[i], ")!");

                internalError = true;
                return false;
            }
            scaling_D1[i] = -(int)std::ceil(std::logb(rowSums[i]));
        }
        P_csc.scaleRows(scaling_D1);

        return true;
    }

    // J-free variant of the scaling above: no CSC J/P exists, so the D1 row
    // sums are estimated with one J*v product in direction 2^D2.  The
    // assembled path sums |P_ij|; this uses |sum_j P_ij 2^D2_j|.  The two
    // scalings can differ, but both are valid Newton scalings and the
    // converged root is unchanged.  This only runs when a user preconditioner
    // explicitly declares requiresAssembledMatrix() == false.
    bool calculateNewtonScalingJFree() {
        unsigned int i, j;
        for (j = 0; j < numOfSpecies; j++) {
            if (!(error_weights[j] > 0.0) || !std::isfinite(error_weights[j])) {
                YABI_LOG_ERROR("ERROR: non-positive or non-finite error weight for species ", j,
                               " (", error_weights[j], ")!");

                internalError = true;
                return false;
            }
            scaling_D2[j] = (long)std::floor(std::logb(error_weights[j]) + 0.5);
        }

        const std::size_t domain = RHS->Domain();
        std::vector<Scalar> direction(domain, Scalar(0));
        for (j = 0; j < numOfSpecies; ++j)
            direction[j] = (Scalar)std::scalbn((Scalar)1.0, (int)scaling_D2[j]);

        std::vector<Scalar> Jd;
        if (!RHS->jacobianVectorProduct(matrixFreeBase, direction, Jd) ||
            Jd.size() < numOfSpecies) {
            YABI_LOG_ERROR("ERROR: J-free Newton scaling needs a Jacobian-vector product!");
            internalError = true;
            return false;
        }

        const Scalar factor = nordsieck.step_size_h / nordsieck.l_i[1];
        for (i = 0; i < numOfSpecies; i++) {
            const Scalar sum =
                std::fabs((Scalar)std::scalbn((Scalar)1.0, (int)scaling_D2[i]) - factor * Jd[i]);
            if (!(sum > 0.0) || !std::isfinite(sum)) {
                YABI_LOG_ERROR("ERROR: cannot scale column ", i,
                               " of the J-free Newton matrix (row sum ", sum, ")!");

                internalError = true;
                return false;
            }
            scaling_D1[i] = (long)-(int)std::ceil(std::logb(sum));
        }

        return true;
    }

    bool configureJFreeMatrix(const std::vector<Scalar> &base) {
        matrixFreeBase = base;
        configureMatrixFreeOperator();
        if (!calculateNewtonScalingJFree())
            return false;
        // The user preconditioner declared that it does not need an assembled
        // matrix; the empty temporary is not retained by the solver.
        CscMatrix<Scalar> no_matrix;
        return linearSolver->factorizeCsc(no_matrix, ip.data(), scaling_D1.data(),
                                          scaling_D2.data(), numOfSpecies);
    }

    NordsieckState<Scalar> &nordsieck;
    const IntegratorConfig<Scalar> &config;
    IntegratorStatistics &stats;

    std::unique_ptr<AbstractRHS<Scalar>> &RHS;
    std::unique_ptr<LinearSolver<Scalar>> &linearSolver;

    const unsigned int &numOfSpecies;
    const unsigned int &RHS_type;
    const unsigned int &sizeOfParameter;
    std::vector<Scalar> &parameter;

    std::vector<Scalar> &error_weights;
    std::vector<Scalar> &absolut_error;
    Scalar &relative_error;
    std::vector<std::vector<Scalar>> &error_weights_derivatives;
    Scalar **&absolut_error_derivatives;
    Scalar *&relative_error_derivatives;

    Scalar &t_ende;
    Scalar &minimalEta;
    Scalar &reachableError;
    std::vector<Scalar> &reachableError_derivatives;

    unsigned int &numOfDerivatives;
    bool &direct;
    bool &taping;
    std::vector<std::vector<Scalar>> &z_predictor_derivatives;
    std::vector<std::vector<Scalar>> &corrector_derivatives;
    Scalar **&p_seed_derivatives;

    ReplayTape<Scalar> *&startTAPE;
    ReplayTape<Scalar> *&endTAPE;
    ReplayTapeStore<Scalar> &replayTapeStore;
    AdFunList &cppADFunList;
    typename AdFunList::iterator &intermediateCppADFunIterator;
    bool &internalError;

  public:
    // Newton-owned state.
    std::vector<Scalar> P;
    std::vector<Scalar> J;
    std::vector<int> ip;
    std::vector<long> scaling_D1;
    std::vector<long> scaling_D2;

    // Native sparse state (T31).  When sparseAssembly is enabled and the RHS
    // exposes a pattern, J_csc/P_csc are the source of truth and the dense P/J
    // above are left untouched.
    bool sparseAssembly = false;
    bool sparsePatternReady = false;
    SparsityPattern sparsePattern;
    CscMatrix<Scalar> J_csc;
    CscMatrix<Scalar> P_csc;

    // Explicit Jacobian-backend selection (T33).  Auto reproduces the historical
    // capability-driven behaviour; Dense/Sparse/MatrixFree force that path and
    // fail loudly when the RHS + solver cannot satisfy it.
    JacobianBackend jacobianBackend = JacobianBackend::Auto;

    // Matrix-free state (T32).  The operator is owned here and borrowed by the
    // installed Krylov solver; matrixFreeBase is the J evaluation point.
    bool matrixFreeAssembly = false;
    bool matrixFreeWarned = false;
    std::vector<Scalar> matrixFreeBase;
    std::unique_ptr<NewtonMatrixFreeOperator> matrixFreeOperator;

    bool setJacobian = false;
    bool setNewP = false;
    bool newtonIteration_Started = false;
    Scalar convergenceIndicator = 0;
    unsigned int newtonFAIL = 0;
    unsigned int stepFAIL = 0;
};

} // namespace detail
} // namespace yabi

#endif
