#!/usr/bin/env bash
# Minimal SYCL reproducer for the sycl::pow(10,1) module-wide-contraction effect.
#
# The SAME float LogspaceFunctor<float,float> kernel is compiled twice; the only
# difference is -DEXTRA_DTYPES, which adds the other-dtype sibling kernels
# (double/half/bf16/int*/complex) into the SAME translation unit -- exactly how
# torch-xpu-ops' RangeFactoriesKernel.cpp instantiates every dtype variant.
#
# Expected on Linux / PVC (icx 2026.0):
#   float-only  -> pow(10,1) = 0x41200001 (1 ULP high)
#   all-dtypes  -> pow(10,1) = 0x41200000 (exact 10.0)
set -euo pipefail
cd "$(dirname "$0")"

CG="-options -cl-poison-unsupported-fp64-kernels \
    -options -cl-intel-enable-auto-large-GRF-mode \
    -options -cl-fp32-correctly-rounded-divide-sqrt \
    -options -cl-intel-greater-than-4GB-buffer-required"

# torch-xpu-ops' exact device kernel flags (GNU/Linux branch of BuildFlags.cmake).
FLAGS="-fsycl -fno-sycl-unnamed-lambda -sycl-std=2020 -std=c++20 -Wno-absolute-value \
       -fno-fast-math -ffp-contract=fast -no-ftz -fsycl-targets=spir64_gen,spir64"

echo "=== building float-only (EXTRA_DTYPES=0) ==="
icx $FLAGS -DEXTRA_DTYPES=0 \
    -Xsycl-target-backend=spir64_gen "-device pvc $CG" \
    pow_dtypes.cpp -o pow_floatonly

echo "=== building all-dtypes (EXTRA_DTYPES=1) ==="
icx $FLAGS -DEXTRA_DTYPES=1 \
    -Xsycl-target-backend=spir64_gen "-device pvc $CG" \
    pow_dtypes.cpp -o pow_alldtypes

echo; echo "==== RUN: float-only ===="; ./pow_floatonly
echo; echo "==== RUN: all-dtypes ===="; ./pow_alldtypes
