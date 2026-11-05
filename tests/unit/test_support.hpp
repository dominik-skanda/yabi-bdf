/*
 *  This file is part of YABI (Yet Another BDF Integrator).
 *
 *  SPDX-License-Identifier: Apache-2.0
 */

#ifndef YABI_TESTS_UNIT_TEST_SUPPORT_HPP
#define YABI_TESTS_UNIT_TEST_SUPPORT_HPP

#include <cmath>
#include <iostream>
#include <string>

namespace yabi_test {

// One failure counter per test executable.  Keeping this in a header lets each
// unit test stay a self-contained, independently runnable ctest case while the
// reporting stays uniform.
inline int &failures() {
    static int value = 0;
    return value;
}

inline void check(bool ok, const std::string &what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << std::endl;
        ++failures();
    }
}

inline bool near(double actual, double expected, double tolerance = 1e-12) {
    return std::fabs(actual - expected) <= tolerance;
}

inline int finish(const char *suite) {
    if (failures() != 0) {
        std::cerr << failures() << " " << suite << " check(s) failed" << std::endl;
        return 1;
    }
    std::cout << "PASS: " << suite << std::endl;
    return 0;
}

} // namespace yabi_test

#endif
