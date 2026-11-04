/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  Copyright (C) 2010-2012 Dominik Skanda
 *
 *  SPDX-License-Identifier: Apache-2.0
 */
#ifndef YABI_DETAIL_REPLAY_TAPE_HPP
#define YABI_DETAIL_REPLAY_TAPE_HPP

#include <cstddef>
#include <memory>
#include <vector>

#include <yabi/abstract_rhs.hpp>

namespace yabi {
namespace detail {

// One recorded integration step used by the forward/adjoint sensitivity replay.
//
// The node owns its per-step numeric data (solutions, iteration matrices and
// scaling vectors) through std::vector; it only *borrows* the AD functions
// (`CppAD_tape`, `direct_fun`), which live in the integrator's CppAD function
// pool for the duration of the integration.  Nodes form a doubly linked list so
// the replay can walk forward and backward in O(1).
template <typename Scalar> struct ReplayTape {
    Scalar timePoint = 0;
    Scalar stepSize = 0;

    // Per-Newton-iterate AD functions (non-owning).
    std::vector<AbstractRHS<Scalar> *> CppAD_tape;
    AbstractRHS<Scalar> *direct_fun = nullptr; // non-owning

    std::vector<Scalar> endSolution;
    std::vector<Scalar> P;
    std::vector<int> ip;
    std::vector<long> scaling_D1;
    std::vector<long> scaling_D2;

    // NAVIGATION
    ReplayTape *forwardLINK = nullptr;
    ReplayTape *backLINK = nullptr;

    bool calculate_output = false;
    bool orderUP = false;
    bool orderDOWN = false;
    unsigned int NewtonIterates = 0;

    std::vector<Scalar> old_parameter_seed;
    std::vector<Scalar> old_y_seed;
    bool updatedTaylorFunction = false;
    unsigned int lastTaylorOrder = 0;
};

// Owns every replay-tape node for one integration.  Nodes keep stable addresses
// (they are individually heap-allocated and owned through unique_ptr), so the
// raw forward/back/link cursors used by the replay stay valid until clear().
template <typename Scalar> class ReplayTapeStore {
  public:
    ReplayTape<Scalar> *create() {
        storage_.push_back(std::make_unique<ReplayTape<Scalar>>());
        return storage_.back().get();
    }

    void clear() { storage_.clear(); }

    bool empty() const { return storage_.empty(); }
    std::size_t size() const { return storage_.size(); }

  private:
    std::vector<std::unique_ptr<ReplayTape<Scalar>>> storage_;
};

} // namespace detail
} // namespace yabi

#endif
