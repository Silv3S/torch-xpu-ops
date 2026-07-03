# Minimal reproducer: `sycl::pow(10,1)` module-wide contraction effect

## What it shows

One source file (`pow_dtypes.cpp`), one **byte-identical** float
`LogspaceFunctor<float,float>` kernel. The only variable is `-DEXTRA_DTYPES`,
which adds the other-dtype sibling kernels (double / half / bf16 / int* /
complex) into the SAME translation unit -- exactly how torch-xpu-ops'
`RangeFactoriesKernel.cpp` instantiates every dtype variant.

| build (identical float kernel)          | `pow(10,1)`  | verdict    |
|-----------------------------------------|--------------|------------|
| `-DEXTRA_DTYPES=0` (float kernel only)  | `0x41200001` | 1 ULP high |
| `-DEXTRA_DTYPES=1` (+ all dtype kernels)| `0x41200000` | exact 10.0 |

## Run

```bash
./build_and_run.sh
```

(Linux / Intel Data Center GPU Max 1550 / oneAPI icx 2026.0.)

## Mechanism

Device SPIR-V (`-fsycl-device-only` -> `llvm-spirv -to-text`), counting
`ExecutionMode <id> 31` == `ContractionOff`:

| build          | ContractionOff modes | on which kernels                        |
|----------------|----------------------|-----------------------------------------|
| EXTRA_DTYPES=0 | 0                    | (none)                                  |
| EXTRA_DTYPES=1 | 6                    | `Tbf` (bf16), `Tcf` (cfloat), `Tcd` (cdouble) |

The float kernel (`_ZTS2Tf`) carries **no** `ContractionOff` of its own in either
build. In the all-dtypes build it nonetheless rounds exact, because IGC applies
the bf16/complex kernels' `ContractionOff` **module-wide**. Strip those siblings
(float-only build) and the float `pow` stays contracted -> 1 ULP high.

Root cause is therefore not float-specific and not a build flag: it is the
multi-dtype composition of the translation unit. Only bf16 / complex<float> /
complex<double> emit `ContractionOff` (float, double, half, and the integral
variants do not), so their presence in the TU is what flips the float result.

## `-ffp-contract` sweep (Linux / PVC / icx 2026.0)

Same source, both variants, built with `-ffp-contract={fast,on,off}` (all other
flags held at torch's exact set). `fast` is torch-xpu-ops' actual setting.

| `-ffp-contract` | variant     | `pow(10,1)`  | verdict    |
|-----------------|-------------|--------------|------------|
| **fast**        | float-only  | `0x41200001` | 1 ULP high |
| **fast**        | all-dtypes  | `0x41200000` | exact 10.0 |
| **on**          | float-only  | `0x41200001` | 1 ULP high |
| **on**          | all-dtypes  | `0x41200000` | exact 10.0 |
| **off**         | float-only  | `0x41200000` | exact 10.0 |
| **off**         | all-dtypes  | `0x41200000` | exact 10.0 |

`ContractionOff` execution modes emitted in each device module:

| `-ffp-contract` | float-only | all-dtypes |
|-----------------|------------|------------|
| fast            | 0          | 6          |
| on              | 0          | 8          |
| off             | 2          | 22         |

Reading:

- **`off`** is the only mode that makes the float-only build exact. It puts
  `ContractionOff` on the float kernel itself (count 2 covers the float kernel +
  its `pf_kernel_wrapper`), so `pow` rounds correctly with no sibling kernels
  needed. This is the robust, composition-independent fix.
- **`fast` and `on`** never de-contract the float kernel directly (float-only
  stays `0x41200001`). The all-dtypes build only lands on exact 10.0 because the
  bf16/complex siblings inject `ContractionOff` that IGC applies module-wide -- an
  accidental leak, not a guarantee. `fast` -> 6, `on` -> 8: the count grows but,
  critically, none of those modes sit on the float kernel; the float kernel is
  de-contracted purely by the module-wide application.
- The `fast` vs `on` difference (6 vs 8 modes) does not change the float result
  here: both leak enough to de-contract the float kernel in the all-dtypes TU.

## `-ffp-contract` sweep (Windows)

Run `run_windows.bat` from an Intel oneAPI command prompt inside this directory
(set `DEV` to your GPU, e.g. `set DEV=bmg`). The source is unchanged -- only the
driver flags differ (`/fp:strict /Qftz-` + explicit `-ffp-contract=<mode>`, which
overrides the `-ffp-model=strict` implied contraction; the "overriding" warning
is expected). It builds and runs all 3 modes x 2 variants and prints both tables.

- Compiler: icx `<fill in: icx --version>`
- GPU / AOT device (`DEV`): `<fill in>`

Table 1 -- `pow(10,1)`:

| `-ffp-contract` | variant     | `pow(10,1)`  | verdict |
|-----------------|-------------|--------------|---------|
| fast            | float-only  | `0x________` |         |
| fast            | all-dtypes  | `0x________` |         |
| on              | float-only  | `0x________` |         |
| on              | all-dtypes  | `0x________` |         |
| off             | float-only  | `0x________` |         |
| off             | all-dtypes  | `0x________` |         |

Table 2 -- `ContractionOff` execution modes:

| `-ffp-contract` | float-only | all-dtypes |
|-----------------|------------|------------|
| fast            |            |            |
| on              |            |            |
| off             |            |            |

Key comparison to make against Linux: does the **all-dtypes** build reach exact
`0x41200000` under `fast`/`on` on Windows (i.e. does the module-wide
`ContractionOff` leak reach the float kernel), or does it stay `0x41200001`? If
Windows all-dtypes stays 1 ULP high while Linux goes exact -- with identical
`ContractionOff` counts -- the divergence is in how the runtime/driver applies
module-wide contraction, not in flag semantics.
