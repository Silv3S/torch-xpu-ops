# Minimal reproducer: `sycl::pow(10,1)` cross-kernel contraction leak

## What it shows

One source file (`pow_dtypes.cpp`), one **byte-identical** float
`LogspaceFunctor<float,float>` kernel. A `-DVARIANT=<n>` toggle changes only which
sibling dtype kernels share the translation unit:

- `VARIANT=0` **float-only** -- just the float kernel.
- `VARIANT=1` **float+cfloat** -- float + `std::complex<float>` (minimal trigger).
- `VARIANT=2` **all-dtypes** -- float + every dtype torch's `RangeFactoriesKernel`
  instantiates (double / half / bf16 / int* / complex), the faithful mirror.

Each `run` script sweeps `-ffp-contract={fast,on,off}` x the 3 variants = 9 builds
and prints two tables: the `pow(10,1)` result, and the count of `ContractionOff`
SPIR-V execution modes in each device module. `fast` is torch-xpu-ops' real
setting.

Table 2 requires `llvm-spirv` (ships with oneAPI, in the `compiler\` subdir next
to icx/icpx). Both scripts auto-locate it; if it is missing, Table 2 is skipped
and Table 1 still prints.

## How to run

- **Linux:** `./run.sh` (uses **icpx**, the GNU-style driver -- matches the real
  torch-xpu-ops Linux build).
- **Windows:** `run.bat` from an Intel oneAPI command prompt (uses **icx**, the
  clang-cl / MSVC-style driver -- matches the real torch-xpu-ops Windows build).

Set the AOT device to your GPU:

```bash
DEV=pvc ./run.sh        # Linux:   pvc | bmg | mtl | dg2 ...
```
```bat
set DEV=bmg  &  run.bat   REM Windows: bmg | mtl | dg2 | arl-h | lnl-m ...
```

## Minimal flags (and why)

The only flags that matter are:

```
-fsycl -ffp-contract=<mode> -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device <dev>"
```

Everything else torch-xpu-ops passes (`-fno-sycl-unnamed-lambda`, `-sycl-std`,
`-std=c++20`, `-Wno-absolute-value`, `-fno-fast-math`, `-no-ftz`, and all the
`-cl-*` codegen `-options`) was verified **irrelevant** to this effect and dropped.

**AOT (`-fsycl-targets=spir64_gen`) is REQUIRED.** Under pure JIT the SYCL/L0
runtime online compiler does not honor the cross-kernel `ContractionOff`, so the
sibling variants stay `0x41200001`. The split only appears in the AOT
`spir64_gen` image.

The dash-style flags (`-fsycl`, `-ffp-contract`, `-fsycl-targets`) are accepted
by both driver modes, so the SAME flag strings work under icpx and icx-clang-cl;
only the driver binary differs per platform.

## Mechanism (the leak is code-path-specific, not module-global)

The float kernel (`_ZTS2Tf`) carries **no** `ContractionOff` of its own, yet with
the right sibling present its `pow` rounds exact. Isolating each sibling shows the
leak is NOT simply "any kernel with `ContractionOff` disables it module-wide":

| sibling added (AOT, `fast`) | ContractionOff | float `pow(10,1)` | flips? |
|-----------------------------|----------------|-------------------|--------|
| none (float only)           | 0              | `0x41200001`      | --     |
| + half                      | 0              | `0x41200001`      | no     |
| + double                    | 0              | `0x41200001`      | no     |
| + bf16                      | 2              | `0x41200001`      | no     |
| + complex&lt;double&gt;     | 2              | `0x41200001`      | no     |
| **+ complex&lt;float&gt;**  | 2              | **`0x41200000`**  | **YES**|

bf16 and complex&lt;double&gt; DO emit `ContractionOff`, but the float `pow` stays
1 ULP high. Only **complex&lt;float&gt;** flips it. The reason: complex&lt;float&gt;
lowers to the **`powf`** builtin -- the SAME builtin the float kernel uses -- so its
`ContractionOff` de-contracts the shared `powf` code path. complex&lt;double&gt; and
bf16 lower to the **`pow`** (double) path, which the float kernel does not share.
(Verified via the pow builtins in each module's SPIR-V: `+cfloat` has `powf`,
`+cdouble`/`+bf16` have only `pow`.)

Root cause is therefore not float-specific and not a build flag: it is whether a
`ContractionOff`-bearing kernel shares the float kernel's `powf` lowering.

## Results

Each block below is one recorded run of `run.sh` / `run.bat` on a given machine.

### [1] Linux / PVC / icpx 2026.0  (baseline, recorded 2026-07-03)

- GPU: Intel(R) Data Center GPU Max 1550, L0 driver 1.6.33578+51 (agama 12.60.7)
- Compiler: Intel oneAPI DPC++/C++ 2026.0.0 (2026.0.0.20260331), driver `icpx`
- Command: `DEV=pvc ./run.sh`

Table 1 -- `pow(10,1)`:

| `-ffp-contract` | variant       | `pow(10,1)`  | verdict    |
|-----------------|---------------|--------------|------------|
| **fast**        | float-only    | `0x41200001` | 1 ULP high |
| **fast**        | float+cfloat  | `0x41200000` | exact 10.0 |
| **fast**        | all-dtypes    | `0x41200000` | exact 10.0 |
| **on**          | float-only    | `0x41200001` | 1 ULP high |
| **on**          | float+cfloat  | `0x41200000` | exact 10.0 |
| **on**          | all-dtypes    | `0x41200000` | exact 10.0 |
| **off**         | float-only    | `0x41200000` | exact 10.0 |
| **off**         | float+cfloat  | `0x41200000` | exact 10.0 |
| **off**         | all-dtypes    | `0x41200000` | exact 10.0 |

Table 2 -- `ContractionOff` execution modes:

| `-ffp-contract` | float-only | float+cfloat | all-dtypes |
|-----------------|------------|--------------|------------|
| fast            | 0          | 2            | 6          |
| on              | 0          | 2            | 8          |
| off             | 2          | 4            | 22         |

Reading: `off` fixes even float-only (it puts `ContractionOff` on the float kernel
itself). Under `fast`/`on`, float-only stays 1 ULP high; adding complex&lt;float&gt;
(and hence all-dtypes) reaches exact 10.0 via the shared-`powf` leak.

### [2] Linux / BMG / icpx 2026.0  (recorded 2026-07-03)

- Compiler: Intel oneAPI DPC++/C++ 2026.0.0 (2026.0.0.20260331), driver `icpx`
- Command: `DEV=bmg ./run.sh` (true BMG-targeted AOT image)

Result: identical to the PVC baseline -- under `fast`/`on`, float-only is 1 ULP
high while float+cfloat and all-dtypes are exact; `off` fixes all. The
cross-kernel `powf` leak reaches the float kernel on BMG+Linux too, so the
Linux-vs-Windows divergence is not a BMG-arch effect.

Table 1 -- `pow(10,1)`:

| `-ffp-contract` | variant       | `pow(10,1)`  | verdict    |
|-----------------|---------------|--------------|------------|
| **fast**        | float-only    | `0x41200001` | 1 ULP high |
| **fast**        | float+cfloat  | `0x41200000` | exact 10.0 |
| **fast**        | all-dtypes    | `0x41200000` | exact 10.0 |
| **on**          | float-only    | `0x41200001` | 1 ULP high |
| **on**          | float+cfloat  | `0x41200000` | exact 10.0 |
| **on**          | all-dtypes    | `0x41200000` | exact 10.0 |
| **off**         | float-only    | `0x41200000` | exact 10.0 |
| **off**         | float+cfloat  | `0x41200000` | exact 10.0 |
| **off**         | all-dtypes    | `0x41200000` | exact 10.0 |

Table 2 -- `ContractionOff` execution modes:

| `-ffp-contract` | float-only | float+cfloat | all-dtypes |
|-----------------|------------|--------------|------------|
| fast            | 0          | 2            | 6          |
| on              | 0          | 2            | 8          |
| off             | 2          | 4            | 22         |

### [3] Windows / BMG / icx 2026.0  (recorded 2026-07-03)

- GPU: Intel(R) Arc(TM) B570 Graphics (Battlemage)
- Compiler: Intel oneAPI DPC++/C++ 2026.0.0 (2026.0.0.20260331), driver `icx`
  (clang-cl / MSVC-style)
- Command: `set DEV=bmg  &  run.bat`

**DECISIVE DIVERGENCE. Windows never reaches exact 10.0 -- in ANY config.** Same
compiler version and same GPU as run [2]; only the driver mode differs (icx
clang-cl vs icpx GNU). Every one of the 9 cases is `0x41200001` (1 ULP high),
including `-ffp-contract=off` on float-only -- the fix that is guaranteed on
Linux.

Table 1 -- `pow(10,1)`:

| `-ffp-contract` | variant       | `pow(10,1)`  | verdict    |
|-----------------|---------------|--------------|------------|
| **fast**        | float-only    | `0x41200001` | 1 ULP high |
| **fast**        | float+cfloat  | `0x41200001` | 1 ULP high |
| **fast**        | all-dtypes    | `0x41200001` | 1 ULP high |
| **on**          | float-only    | `0x41200001` | 1 ULP high |
| **on**          | float+cfloat  | `0x41200001` | 1 ULP high |
| **on**          | all-dtypes    | `0x41200001` | 1 ULP high |
| **off**         | float-only    | `0x41200001` | 1 ULP high |
| **off**         | float+cfloat  | `0x41200001` | 1 ULP high |
| **off**         | all-dtypes    | `0x41200001` | 1 ULP high |

Table 2 -- `ContractionOff` execution modes:

| `-ffp-contract` | float-only | float+cfloat | all-dtypes |
|-----------------|------------|--------------|------------|
| fast            | 0          | 2            | 6          |
| on              | 0          | 2            | 6          |
| off             | 0          | 2            | 6          |

Two independent Windows/clang-cl divergences from Linux:

1. **Front-end (SPIR-V emission):** the counts are **mode-invariant** on Windows
   (always 0/2/6), whereas on Linux `-ffp-contract` changes them -- notably `off`
   stamps `ContractionOff` onto the float kernel itself (Linux off float-only=2,
   Windows=0). So the clang-cl driver does NOT translate `-ffp-contract=off` into
   a kernel-scoped `ContractionOff` on the float kernel. This is why the Linux
   `off` fix does nothing on Windows.
2. **Back-end (AOT honoring):** for `fast`, BOTH platforms emit the SAME counts
   (0/2/6), yet Linux float+cfloat/all-dtypes come out exact while Windows stays
   1 ULP high. Same `ContractionOff` on the cfloat sibling, opposite result --
   the Windows AOT backend does not de-contract the shared `powf` path even when
   the mode is present.

Bottom line: on Windows/clang-cl there is no `-ffp-contract` setting -- and no
sibling-kernel composition -- that yields exact `pow(10,1)` in this setup. The
exactness Linux gets is a Linux-driver behavior, not something the flags or the
source can force on Windows. (To fully separate divergence 1 from 2, pull the
Windows device SPIR-V for the `fast`/float+cfloat build and diff its
`ContractionOff` placement + `powf` lowering against Linux's -- same counts do
not guarantee byte-identical modules.)
