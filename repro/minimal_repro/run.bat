@echo off
REM Windows driver: runs all 3 experiments (-ffp-contract={fast,on,off}) x 3
REM variants (float-only / float+cfloat / all-dtypes) in one call and prints TWO
REM tables:
REM   Table 1: pow(10,1) result per (mode, variant)
REM   Table 2: ContractionOff (SPIR-V ExecutionMode 31) count per (mode, variant)
REM
REM Uses icx (the clang-cl / MSVC-style driver) to match the real torch-xpu-ops
REM Windows build. Run from an "Intel oneAPI command prompt" in this directory.
REM
REM Usage:   set DEV=bmg  &  run.bat        (DEV: bmg, mtl, dg2, arl-h, lnl-m ...)
REM
REM VARIANT: 0=float-only  1=float+complex<float> (minimal trigger)  2=all-dtypes
REM
REM AOT (-fsycl-targets=spir64_gen) is REQUIRED for Table 1: under pure JIT the
REM module-wide ContractionOff is not honored and the sibling variants stay 1 ULP
REM high. Table 2 inspects the device SPIR-V, which is target-independent.
setlocal enabledelayedexpansion
if "%DEV%"=="" set DEV=bmg

REM --- locate llvm-spirv matching icx (prefer the one next to icx; a version
REM     mismatch with a PATH copy produces empty SPIR-V text) ---
set LS=
for /f "delims=" %%p in ('where icx 2^>nul') do if "!LS!"=="" (
  if exist "%%~dpp\compiler\llvm-spirv.exe" set LS=%%~dpp\compiler\llvm-spirv.exe
)
if "!LS!"=="" for /f "delims=" %%p in ('where llvm-spirv 2^>nul') do if "!LS!"=="" set LS=%%p

echo === compiler ===
icx --version | findstr /i "Compiler"
echo === AOT device: %DEV% ===
if "!LS!"=="" (echo === llvm-spirv: NOT FOUND ^(Table 2 skipped^) ===) else (echo === llvm-spirv: !LS! ===)
echo.

echo ###### Table 1: pow(10,1) ######
for %%m in (fast on off) do (
  for %%v in (0 1 2) do (
    icx -fsycl -ffp-contract=%%m ^
        -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device %DEV%" ^
        -DVARIANT=%%v pow_dtypes.cpp -o pd_%%m_%%v.exe >nul 2>&1
    echo(
    echo -- -ffp-contract=%%m  VARIANT=%%v --
    pd_%%m_%%v.exe
  )
)

echo.
echo ###### Table 2: ContractionOff ^(SPIR-V ExecutionMode 31^) ######
if "!LS!"=="" (
  echo ^(skipped: llvm-spirv not found^)
) else (
  echo ffp-contract    float-only   float+cfloat   all-dtypes
  for %%m in (fast on off) do (
    for %%v in (0 1 2) do (
      icx -fsycl -ffp-contract=%%m -DVARIANT=%%v -fsycl-device-only ^
          -o dev_%%m_%%v.bc pow_dtypes.cpp >nul 2>&1
      "!LS!" dev_%%m_%%v.bc -o dev_%%m_%%v.spv >nul 2>&1
      "!LS!" -to-text dev_%%m_%%v.spv -o dev_%%m_%%v.spt >nul 2>&1
      REM Empty .spt => tool failure (e.g. version mismatch); report ERR not 0.
      set N=ERR
      for %%z in (dev_%%m_%%v.spt) do if %%~zz GTR 0 (
        for /f %%c in ('findstr /R /C:"ExecutionMode [0-9][0-9]* 31 " dev_%%m_%%v.spt ^| find /c /v ""') do set N=%%c
      )
      set CO_%%v=!N!
    )
    echo %%m             !CO_0!            !CO_1!             !CO_2!
  )
)

del /q pd_*.exe dev_*.bc dev_*.spv dev_*.spt >nul 2>&1
endlocal
