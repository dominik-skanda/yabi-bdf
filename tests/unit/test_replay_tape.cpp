/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

#include "test_support.hpp"

#include <yabi/detail/replay_tape.hpp>

#include <vector>

int main() {
    yabi::detail::ReplayTapeStore<double> store;
    yabi_test::check(store.empty(), "store starts empty");
    yabi_test::check(store.size() == 0, "store size starts at 0");

    yabi::detail::ReplayTape<double> *first = store.create();
    yabi::detail::ReplayTape<double> *second = store.create();
    yabi::detail::ReplayTape<double> *third = store.create();

    yabi_test::check(!store.empty(), "store not empty after create");
    yabi_test::check(store.size() == 3, "store size after three creates");

    // Nodes are individually heap-allocated, so later growth must not move them.
    first->timePoint = 1.0;
    second->timePoint = 2.0;
    third->timePoint = 3.0;
    first->forwardLINK = second;
    second->backLINK = first;
    second->forwardLINK = third;
    third->backLINK = second;
    store.create(); // force the unique_ptr vector to grow
    yabi_test::check(first->timePoint == 1.0 && second->timePoint == 2.0 && third->timePoint == 3.0,
                     "node addresses stable across store growth");
    yabi_test::check(first->forwardLINK == second && third->backLINK == second,
                     "tape links preserved across store growth");
    yabi_test::check(store.size() == 4, "store size after growth");

    // Vector-owned per-step data is independent per node.
    first->endSolution.assign(2, 0.5);
    second->endSolution.assign(2, 1.5);
    yabi_test::check(first->endSolution[0] == 0.5 && second->endSolution[0] == 1.5,
                     "per-node buffers are independent");

    store.clear();
    yabi_test::check(store.empty() && store.size() == 0, "store clear releases every node");

    return yabi_test::finish("ReplayTape unit tests");
}
