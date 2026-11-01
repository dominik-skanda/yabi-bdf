#!/bin/bash
# run_examples.sh — run all examples verbosely
set -euo pipefail
set -x

cd "$(dirname "$0")" || exit 1

echo "========================================"
echo "RUN EXAMPLES — $(date -Iseconds) — $(pwd)"
echo "========================================"

if [[ ! -d build ]]; then
    echo "Building first..."
    rm -rf build
    cmake -S . -B build
    cmake --build build -j"$(nproc)"
fi

for b in build/yabi_basic build/yabi_nonautonomous build/yabi_sensitivity_demo; do
    echo "--- RUNNING $(basename $b) ---"
    if [[ -x "$b" ]]; then
        echo "COMMAND: $b"
        $b
        echo "--- OK ---"
    else
        echo "SKIP: $b not found"
    fi
done

# Heat1D examples if present
for b in build/yabi_heat1d_dense_lit build/yabi_heat1d_umfpack_lit build/yabi_heat1d_umfpack; do
    echo "--- RUNNING $(basename $b) ---"
    if [[ -x "$b" ]]; then
        echo "COMMAND: $b"
        timeout 60 $b 2>&1 | tail -5
        echo "--- OK ---"
    fi
done

echo "========================================"
echo "EXAMPLES DONE — $(date -Iseconds)"
echo "========================================"
exit 0
