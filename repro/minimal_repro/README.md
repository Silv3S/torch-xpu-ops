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

### [N] <platform / GPU / compiler version> -- fill in

- GPU: `<name + L0 driver>`
- Compiler: `<icx/icpx --version>`, driver `<icpx | icx>`
- Command: `<DEV=... ./run.sh | set DEV=... & run.bat>`

Table 1 -- `pow(10,1)`:

| `-ffp-contract` | variant       | `pow(10,1)`  | verdict |
|-----------------|---------------|--------------|---------|
| fast            | float-only    | `0x________` |         |
| fast            | float+cfloat  | `0x________` |         |
| fast            | all-dtypes    | `0x________` |         |
| on              | float-only    | `0x________` |         |
| on              | float+cfloat  | `0x________` |         |
| on              | all-dtypes    | `0x________` |         |
| off             | float-only    | `0x________` |         |
| off             | float+cfloat  | `0x________` |         |
| off             | all-dtypes    | `0x________` |         |

Table 2 -- `ContractionOff` execution modes:

| `-ffp-contract` | float-only | float+cfloat | all-dtypes |
|-----------------|------------|--------------|------------|
| fast            |            |              |            |
| on              |            |              |            |
| off             |            |              |            |

Copy this block per machine (PVC+Linux, BMG+Linux, BMG+Windows, MTL+Linux,
MTL+Windows). The key question: under `fast`/`on`, do **float+cfloat** and
**all-dtypes** reach exact `0x41200000`, or stay `0x41200001`? If a platform stays
1 ULP high while Linux/PVC goes exact -- with the same Table 2 counts -- the
cross-kernel `ContractionOff` leak is not reaching the float kernel there, pinning
the divergence to the driver/runtime, not the source or flag semantics.
