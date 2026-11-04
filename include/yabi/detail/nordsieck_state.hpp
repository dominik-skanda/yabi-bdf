/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  Copyright (C) 2010-2012 Dominik Skanda
 *
 *  SPDX-License-Identifier: Apache-2.0
 */
#ifndef YABI_DETAIL_NORDSIECK_STATE_HPP
#define YABI_DETAIL_NORDSIECK_STATE_HPP

#include <cstddef>
#include <vector>

#include <yabi/linear_algebra.hpp>

namespace yabi {
namespace detail {

// The BDF Nordsieck history array and the step/order/time state that belongs to
// it, together with the predictor/corrector/output algebra that operates on
// that state.
//
// This component owns only the base Nordsieck data.  The simultaneous-
// sensitivity mirrors of the history array are owned by the sensitivity engine
// and passed in explicitly by the caller, so the Nordsieck algebra has no
// dependency on the sensitivity machinery.  Similarly, recording the replay
// tape and updating the adjoint cursors stay in the integrator: this class only
// computes values.
template <typename Scalar> class NordsieckState {
  public:
    // Size the base history arrays for `numSpecies` and restore the historical
    // construction defaults (zero history, unit initial correction).
    void initialize(std::size_t numSpecies) {
        z_predictor.assign(numSpecies * 7, Scalar(0));
        z_corrector.assign(numSpecies * 7, Scalar(0));
        z_corrector_old.assign(numSpecies * 7, Scalar(0));

        corrector.assign(numSpecies, Scalar(1));
        corrector_old.assign(numSpecies, Scalar(0));
        solution_Vector.assign(numSpecies, Scalar(0));
    }

    // l_i: coefficients of the BDF corrector with the current step-size history.
    bool calculate_l_vector() {
        unsigned int i, j, k;

        Scalar l_kahan[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
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
        return true;
    }

    // z_predictor = A_q * z_corrector for the current order q.
    bool calculate_next_predictor(unsigned int numSpecies, unsigned int numDerivatives,
                                  Scalar **z_corrector_derivatives,
                                  Scalar **z_predictor_derivatives) {
        unsigned int z;

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

        if (order_corrector + 1 > 7)
            return false;

        Scalar one = 1.0;
        Scalar zero = 0.0;
        int blasTEMP;
        switch (order_corrector) {
        case 1:
            blasTEMP = 2;
            linalg::DGEMM_NT_T(numSpecies, blasTEMP, blasTEMP, one, z_corrector.data(), numSpecies,
                               A1, blasTEMP, zero, z_predictor.data(), numSpecies);
            break;
        case 2:
            blasTEMP = 3;
            linalg::DGEMM_NT_T(numSpecies, blasTEMP, blasTEMP, one, z_corrector.data(), numSpecies,
                               A2, blasTEMP, zero, z_predictor.data(), numSpecies);
            break;
        case 3:
            blasTEMP = 4;
            linalg::DGEMM_NT_T(numSpecies, blasTEMP, blasTEMP, one, z_corrector.data(), numSpecies,
                               A3, blasTEMP, zero, z_predictor.data(), numSpecies);
            break;
        case 4:
            blasTEMP = 5;
            linalg::DGEMM_NT_T(numSpecies, blasTEMP, blasTEMP, one, z_corrector.data(), numSpecies,
                               A4, blasTEMP, zero, z_predictor.data(), numSpecies);
            break;
        case 5:
            blasTEMP = 6;
            linalg::DGEMM_NT_T(numSpecies, blasTEMP, blasTEMP, one, z_corrector.data(), numSpecies,
                               A5, blasTEMP, zero, z_predictor.data(), numSpecies);
            break;
        case 6:
            blasTEMP = 7;
            linalg::DGEMM_NT_T(numSpecies, blasTEMP, blasTEMP, one, z_corrector.data(), numSpecies,
                               A6, blasTEMP, zero, z_predictor.data(), numSpecies);
            break;
        }

        if (numDerivatives > 0) {
            for (z = 0; z < numDerivatives; z++) {
                switch (order_corrector) {
                case 1:
                    blasTEMP = 2;
                    linalg::DGEMM_NT_T(numSpecies, blasTEMP, blasTEMP, one,
                                       z_corrector_derivatives[z], numSpecies, A1, blasTEMP, zero,
                                       z_predictor_derivatives[z], numSpecies);
                    break;
                case 2:
                    blasTEMP = 3;
                    linalg::DGEMM_NT_T(numSpecies, blasTEMP, blasTEMP, one,
                                       z_corrector_derivatives[z], numSpecies, A2, blasTEMP, zero,
                                       z_predictor_derivatives[z], numSpecies);
                    break;
                case 3:
                    blasTEMP = 4;
                    linalg::DGEMM_NT_T(numSpecies, blasTEMP, blasTEMP, one,
                                       z_corrector_derivatives[z], numSpecies, A3, blasTEMP, zero,
                                       z_predictor_derivatives[z], numSpecies);
                    break;
                case 4:
                    blasTEMP = 5;
                    linalg::DGEMM_NT_T(numSpecies, blasTEMP, blasTEMP, one,
                                       z_corrector_derivatives[z], numSpecies, A4, blasTEMP, zero,
                                       z_predictor_derivatives[z], numSpecies);
                    break;
                case 5:
                    blasTEMP = 6;
                    linalg::DGEMM_NT_T(numSpecies, blasTEMP, blasTEMP, one,
                                       z_corrector_derivatives[z], numSpecies, A5, blasTEMP, zero,
                                       z_predictor_derivatives[z], numSpecies);
                    break;
                case 6:
                    blasTEMP = 7;
                    linalg::DGEMM_NT_T(numSpecies, blasTEMP, blasTEMP, one,
                                       z_corrector_derivatives[z], numSpecies, A6, blasTEMP, zero,
                                       z_predictor_derivatives[z], numSpecies);
                    break;
                }
            }
        }

        return true;
    }

    // z_corrector = z_predictor + corrector * l_i^T.
    bool calculate_next_corrector(unsigned int numSpecies, unsigned int numDerivatives,
                                  Scalar **corrector_derivatives, Scalar **z_predictor_derivatives,
                                  Scalar **z_corrector_derivatives) {
        unsigned int i, j, z;

        for (i = 0; i < numSpecies; i++)
            for (j = 0; j <= order_corrector; j++)
                z_corrector[numSpecies * j + i] = z_predictor[numSpecies * j + i];

        int blasTEMP = order_corrector + 1;
        Scalar blasTEMP2 = 1.0;
        linalg::GER(numSpecies, blasTEMP, blasTEMP2, corrector.data(), 1, l_i, 1,
                    z_corrector.data(), numSpecies);

        if (numDerivatives > 0)
            for (z = 0; z < numDerivatives; z++) {
                for (i = 0; i < numSpecies; i++)
                    for (j = 0; j <= order_corrector; j++)
                        z_corrector_derivatives[z][numSpecies * j + i] =
                            z_predictor_derivatives[z][numSpecies * j + i];
                blasTEMP = order_corrector + 1;
                blasTEMP2 = 1.0;
                linalg::GER(numSpecies, blasTEMP, blasTEMP2, corrector_derivatives[z], 1, l_i, 1,
                            z_corrector_derivatives[z], numSpecies);
            }

        return true;
    }

    // Evaluate the Nordsieck interpolant at `tEnd` (dense output).
    bool calculate_output_vector(unsigned int numSpecies, unsigned int numDerivatives,
                                 Scalar **z_corrector_derivatives,
                                 Scalar **solution_Vector_derivatives, Scalar tEnd) {
        unsigned int i, j, z;

        std::vector<Scalar> temp(order_corrector + 1);
        for (i = 0; i <= order_corrector; i++) {
            temp[i] = 1.0;
            for (j = 0; j < i; j++)
                temp[i] *= (tEnd - t_i[1]) / step_size_h;
        }

        Scalar one = 1.0;
        Scalar zero = 0.0;
        int blasTEMP = order_corrector + 1;
        linalg::GEMV_NT(numSpecies, blasTEMP, one, z_corrector.data(), numSpecies, temp.data(), 1,
                        zero, solution_Vector.data(), 1);

        if (numDerivatives > 0)
            for (z = 0; z < numDerivatives; z++)
                linalg::GEMV_NT(numSpecies, blasTEMP, one, z_corrector_derivatives[z], numSpecies,
                                temp.data(), 1, zero, solution_Vector_derivatives[z], 1);

        return true;
    }

    // ---- State -----------------------------------------------------------

    std::vector<Scalar> z_predictor;
    std::vector<Scalar> z_corrector;
    std::vector<Scalar> z_corrector_old;

    std::vector<Scalar> corrector;
    std::vector<Scalar> corrector_old;

    std::vector<Scalar> solution_Vector;

    unsigned int order_corrector = 1;
    unsigned int num_of_steps_per_order = 0;
    Scalar step_size_h = 0;
    Scalar step_size_h_old = 0;
    Scalar c_n_old = 1;
    Scalar l_i[7] = {};
    Scalar t_i[2] = {};
    Scalar stepSizeSet_new[6] = {};
    Scalar stepSizeSet_old[6] = {};
};

} // namespace detail
} // namespace yabi

#endif
