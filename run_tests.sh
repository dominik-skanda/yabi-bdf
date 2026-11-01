#!/bin/bash
# run_tests.sh — verbose self-contained verification
# Every command executed is printed; nothing is suppressed.
set -euo pipefail
set -x  # print every command before execution

cd "$(dirname "$0")" || { echo "FAIL: cd $(dirname $0)"; exit 1; }

echo "========================================"
echo "VERBOSE TEST RUN — $(date -Iseconds) — $(pwd)"
echo "========================================"

echo "=== 1. CLEAN + CONFIGURE ==="
rm -rf build
echo "COMMAND: cmake -S . -B build"
cmake -S . -B build

echo "=== 2. BUILD (verbose) ==="
echo "COMMAND: cmake --build build -j$(nproc)"
cmake --build build -j"$(nproc)"

echo "=== 3. QUICK BINARY CHECKS (full output) ==="
for b in build/yabi_basic build/yabi_nonautonomous build/yabi_sensitivity_demo build/yabi_test_20_brusselator_sensitivity; do
    echo "--- RUNNING $(basename $b) ---"
    if [[ -x "$b" ]]; then
        echo "COMMAND: $b"
        $b || { echo "FAIL: $b exit $?"; exit 1; }
    else
        echo "FAIL: $b not executable"
        exit 1
    fi
done

echo "=== 4. FULL CTEST (full output, no suppression) ==="
echo "COMMAND: ctest --test-dir build --output-on-failure"
ctest --test-dir build --output-on-failure

echo "========================================"
echo "ALL PASS — sound at $(date -Iseconds)"
echo "========================================"
exit 0
