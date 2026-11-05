/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

#ifndef YABI_TESTS_UNIT_ENGINE_FIXTURE_HPP
#define YABI_TESTS_UNIT_ENGINE_FIXTURE_HPP

#include <yabi/detail/adjoint_engine.hpp>
#include <yabi/detail/dense_linear_solver.ipp>
#include <yabi/detail/integrator_config.hpp>
#include <yabi/detail/newton_solver.hpp>
#include <yabi/detail/nordsieck_state.hpp>
#include <yabi/detail/replay_tape.hpp>
#include <yabi/detail/sensitivity_engine.hpp>
#include <yabi/detail/step_controller.hpp>

#include <list>
#include <memory>
#include <vector>

namespace yabi_test {

// Shared dummy dependencies for the Newton / sensitivity / adjoint engines.
// The engines only borrow these in their constructors; the tests touch only
// engine-owned state.
struct EngineFixture {
    yabi::detail::IntegratorConfig<double> config;
    yabi::detail::IntegratorStatistics stats;
    yabi::detail::NordsieckState<double> nordsieck;
    std::unique_ptr<yabi::AbstractRHS<double>> RHS;
    std::unique_ptr<yabi::LinearSolver<double>> linearSolver;
    unsigned int numOfSpecies = 2;
    unsigned int RHS_type = 0;
    unsigned int sizeOfParameter = 1;
    std::vector<double> parameter;
    std::vector<double> error_weights;
    std::vector<double> absolut_error;
    double relative_error = 0.0;
    double **error_weights_derivatives = nullptr;
    double **absolut_error_derivatives = nullptr;
    double *relative_error_derivatives = nullptr;
    double t_ende = 0.0;
    double minimalEta = 0.0;
    double reachableError = 0.0;
    std::vector<double> reachableError_derivatives;
    unsigned int numOfDerivatives = 0;
    bool direct = false;
    bool taping = false;
    bool internalError = false;
    double **z_predictor_derivatives = nullptr;
    double **corrector_derivatives = nullptr;
    double **p_seed_derivatives = nullptr;
    yabi::detail::ReplayTape<double> *startTAPE = nullptr;
    yabi::detail::ReplayTape<double> *endTAPE = nullptr;
    yabi::detail::ReplayTapeStore<double> replayTapeStore;
    std::list<std::unique_ptr<yabi::AbstractRHS<double>>> cppADFunList;
    std::list<std::unique_ptr<yabi::AbstractRHS<double>>>::iterator intermediateCppADFunIterator;
    yabi::AbstractRHS<double> *start_CppAD_tape = nullptr;
    double startTape_stepSize = 0.0;
    double start_integration_time = 0.0;
    yabi::detail::ReplayTape<double> *intermediateTAPE = nullptr;
    int blasONE = 1;
    double blasScalarONE = 1.0;
    double blasScalarMinusONE = -1.0;
    double blasScalarZERO = 0.0;

    yabi::detail::StepController<double> stepController;
    yabi::detail::SensitivityEngine<double> sensitivity;
    yabi::detail::NewtonSolver<double> newton;
    yabi::detail::AdjointEngine<double> adjoint;

    EngineFixture()
        : stepController(nordsieck, config),
          sensitivity(nordsieck, config, RHS, numOfSpecies, RHS_type, sizeOfParameter, parameter,
                      start_integration_time, direct, internalError, startTAPE, intermediateTAPE,
                      start_CppAD_tape, startTape_stepSize, cppADFunList,
                      intermediateCppADFunIterator, blasONE, blasScalarONE, blasScalarMinusONE,
                      blasScalarZERO),
          newton(nordsieck, config, stats, RHS, linearSolver, numOfSpecies, RHS_type,
                 sizeOfParameter, parameter, error_weights, absolut_error, relative_error,
                 error_weights_derivatives, absolut_error_derivatives, relative_error_derivatives,
                 t_ende, minimalEta, reachableError, reachableError_derivatives, numOfDerivatives,
                 direct, taping, z_predictor_derivatives, corrector_derivatives, p_seed_derivatives,
                 startTAPE, endTAPE, replayTapeStore, cppADFunList, intermediateCppADFunIterator,
                 internalError),
          adjoint(sensitivity, nordsieck, newton, RHS, numOfSpecies, RHS_type, sizeOfParameter,
                  parameter, config, direct, internalError, startTape_stepSize, start_CppAD_tape,
                  cppADFunList, intermediateCppADFunIterator, blasONE, blasScalarONE,
                  blasScalarMinusONE, blasScalarZERO) {
        intermediateCppADFunIterator = cppADFunList.end();
        nordsieck.initialize(numOfSpecies);
        parameter.assign(sizeOfParameter, 0.0);
        error_weights.assign(numOfSpecies, 1.0);
        absolut_error.assign(numOfSpecies, 0.0);
        newton.P.assign(numOfSpecies * numOfSpecies, 0.0);
        newton.J.assign(numOfSpecies * numOfSpecies, 0.0);
        newton.ip.assign(numOfSpecies, 0);
        newton.scaling_D1.assign(numOfSpecies, 0);
        newton.scaling_D2.assign(numOfSpecies, 0);
    }
};

} // namespace yabi_test

#endif
