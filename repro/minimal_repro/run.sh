#!/usr/bin/env bash
# Linux driver: runs all 3 experiments (-ffp-contract={fast,on,off}) x 3 variants
# (float-only / float+cfloat / all-dtypes) in one call and prints TWO tables:
#   Table 1: pow(10,1) result per (mode, variant)
#   Table 2: ContractionOff (SPIR-V ExecutionMode 31) count per (mode, variant)
#
# Uses icpx (the GNU-style driver) to match the real torch-xpu-ops Linux build.
#
# Usage:
#   ./run.sh              # AOT device defaults to pvc
#   DEV=bmg ./run.sh      # set to your GPU: pvc, bmg, mtl, dg2, ...
#
# AOT (-fsycl-targets=spir64_gen) is REQUIRED for Table 1: under pure JIT the
# module-wide ContractionOff is not honored and the sibling variants stay 1 ULP
# high. Table 2 inspects the device SPIR-V, which is target-independent.
set -uo pipefail
cd "$(dirname "$0")"
DEV="${DEV:-pvc}"

VARIANTS=(0 1 2)
declare -A VNAME=([0]="float-only" [1]="float+cfloat" [2]="all-dtypes")

# Locate llvm-spirv matching the icpx in use (version mismatch => empty SPIR-V).
LS=""
ICX_DIR="$(dirname "$(command -v icpx 2>/dev/null)" 2>/dev/null)"
[ -n "$ICX_DIR" ] && [ -x "$ICX_DIR/compiler/llvm-spirv" ] && LS="$ICX_DIR/compiler/llvm-spirv"
[ -z "$LS" ] && LS="$(command -v llvm-spirv || true)"
if [ -z "$LS" ]; then
  for c in $(ls -d /opt/intel/oneapi/compiler/*/bin/compiler/llvm-spirv 2>/dev/null | sort -r); do
    [ -x "$c" ] && LS="$c" && break
  done
fi

echo "=== compiler ==="; icpx --version | head -1
echo "=== AOT device: $DEV ==="
[ -n "$LS" ] && echo "=== llvm-spirv: $LS ===" || echo "=== llvm-spirv: NOT FOUND (Table 2 skipped) ==="
echo

echo "###### Table 1: pow(10,1) ######"
printf "%-14s  %-13s  %-12s  %s\n" "ffp-contract" "variant" "bits" "verdict"
for mode in fast on off; do
  for V in "${VARIANTS[@]}"; do
    icpx -fsycl -ffp-contract=$mode \
         -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device $DEV" \
         -DVARIANT=$V pow_dtypes.cpp -o pd_${mode}_$V 2>/dev/null
    bits=$(./pd_${mode}_$V 2>/dev/null | grep -oE '0x[0-9a-f]+' | head -1)
    verdict=$([ "$bits" = "0x41200000" ] && echo "exact 10.0" || echo "1 ULP high")
    printf "%-14s  %-13s  %-12s  %s\n" "$mode" "${VNAME[$V]}" "${bits:-FAIL}" "$verdict"
  done
done

echo
echo "###### Table 2: ContractionOff (SPIR-V ExecutionMode 31) ######"
if [ -z "$LS" ]; then
  echo "(skipped: llvm-spirv not found)"
else
  printf "%-14s  %-11s  %-13s  %s\n" "ffp-contract" "float-only" "float+cfloat" "all-dtypes"
  for mode in fast on off; do
    declare -A n=()
    for V in "${VARIANTS[@]}"; do
      icpx -fsycl -ffp-contract=$mode -DVARIANT=$V -fsycl-device-only \
           -o dev_${mode}_$V.bc pow_dtypes.cpp 2>/dev/null
      "$LS" dev_${mode}_$V.bc -o dev_${mode}_$V.spv 2>/dev/null
      "$LS" -to-text dev_${mode}_$V.spv -o dev_${mode}_$V.spt 2>/dev/null
      if [ -s dev_${mode}_$V.spt ]; then
        n[$V]="$(grep -cE 'ExecutionMode [0-9]+ 31( |$)' dev_${mode}_$V.spt)"
      else
        n[$V]="ERR"
      fi
    done
    printf "%-14s  %-11s  %-13s  %s\n" "$mode" "${n[0]}" "${n[1]}" "${n[2]}"
  done
fi

rm -f pd_* dev_*.bc dev_*.spv dev_*.spt 2>/dev/null
