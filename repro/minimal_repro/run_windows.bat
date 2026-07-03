@echo off
REM ============================================================================
REM Windows counterpart of build_and_run.sh: sweep -ffp-contract={fast,on,off}
REM over the float-only (EXTRA_DTYPES=0) and all-dtypes (EXTRA_DTYPES=1) builds
REM of the SAME pow_dtypes.cpp, and emit the two tables:
REM   Table 1: pow(10,1) float bits per (mode, variant)
REM   Table 2: ContractionOff (SPIR-V ExecutionMode 31) count per (mode, variant)
REM
REM Run from an "Intel oneAPI command prompt" (icx + llvm-spirv on PATH), from
REM inside this minimal_repro directory.
REM
REM Torch's real Windows kernel flags are /fp:strict /Qfma (== -ffp-model=strict
REM -ffp-contract=fast). We keep /fp:strict and set -ffp-contract explicitly; the
REM explicit flag overrides the model's implied contraction (icx prints
REM   warning: overriding '-ffp-model=strict' option with '-ffp-contract=...'
REM which is expected). This mirrors the Linux sweep one-for-one.
REM ============================================================================
setlocal enabledelayedexpansion

REM ---- Pick the AOT device for YOUR GPU. Windows has no pvc. Default bmg.
REM      Valid examples: mtl, mtl-h, bmg, dg2, arl-h, lnl-m, ptl
if "%DEV%"=="" set DEV=bmg

set BASE=-fsycl -fno-sycl-unnamed-lambda -sycl-std=2020 -Qstd=c++20 /fp:strict /Qftz- -fsycl-targets=spir64_gen,spir64
set CG=-options -cl-poison-unsupported-fp64-kernels -options -cl-intel-enable-auto-large-GRF-mode -options -cl-fp32-correctly-rounded-divide-sqrt -options -cl-intel-greater-than-4GB-buffer-required

echo ==========================================================
icx --version
echo Device (AOT): %DEV%
echo ==========================================================

REM -------------------- build + run: results table --------------------
echo.
echo ###### BUILD + RUN (Table 1: pow(10,1) bits) ######
for %%m in (fast on off) do (
  for %%d in (0 1) do (
    icx %BASE% -ffp-contract=%%m -DEXTRA_DTYPES=%%d ^
        -Xsycl-target-backend=spir64_gen "-device %DEV% %CG%" ^
        pow_dtypes.cpp -o pd_%%m_%%d.exe >nul 2>&1
    if "%%d"=="0" (set VAR=float-only) else (set VAR=all-dtypes)
    echo(
    echo -- -ffp-contract=%%m  !VAR! --
    pd_%%m_%%d.exe
  )
)

REM -------------------- device SPIR-V: ContractionOff counts --------------------
echo.
echo ###### SPIR-V (Table 2: ContractionOff / ExecutionMode 31 count) ######
for %%m in (fast on off) do (
  for %%d in (0 1) do (
    icx -fsycl -fno-sycl-unnamed-lambda -sycl-std=2020 -Qstd=c++20 /fp:strict /Qftz- ^
        -ffp-contract=%%m -DEXTRA_DTYPES=%%d -fsycl-device-only ^
        -o dev_%%m_%%d.bc pow_dtypes.cpp >nul 2>&1
    llvm-spirv dev_%%m_%%d.bc -o dev_%%m_%%d.spv >nul 2>&1
    llvm-spirv -to-text dev_%%m_%%d.spv -o dev_%%m_%%d.spt >nul 2>&1
    REM Count lines like: "3 ExecutionMode <id> 31 "  (mode 31 == ContractionOff)
    for /f %%c in ('findstr /R /C:"ExecutionMode [0-9]* 31 " dev_%%m_%%d.spt ^| find /c /v ""') do set N=%%c
    if "%%d"=="0" (set VAR=float-only) else (set VAR=all-dtypes)
    echo -ffp-contract=%%m  !VAR!  ContractionOff=!N!
  )
)

echo.
echo ==========================================================
echo Fill these into README.md (Windows section):
echo   Table 1 = the "pow(10, 1) = ... [bits 0x........]" line per case
echo   Table 2 = the ContractionOff count per case
echo Linux/PVC reference for comparison:
echo   Table1: fast/on float-only=0x41200001 all=0x41200000; off both=0x41200000
echo   Table2: fast 0/6  on 0/8  off 2/22
echo ==========================================================
endlocal
