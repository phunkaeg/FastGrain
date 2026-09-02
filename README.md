# Fast Grain

A GPU-accelerated film grain effect for After Effects, built as a replacement for the
CPU-only, 8-bit **Add Grain** effect.

| | Add Grain (Adobe / Grain Surgery) | Fast Grain |
|---|---|---|
| Render | CPU node graph | CUDA, DirectX 12, or multithreaded CPU fallback |
| Bit depth | 8-bit internally | full float (8 / 16 / 32-bit projects, HDR safe) |
| Multi-Frame Rendering | no | yes (thread-safe, no shared mutable state) |
| Grain pattern | changes with preview resolution / ROI | locked to full-res layer pixels: identical at any downsample or region of interest |
| Deployment | .aex + assets | single .aex (DirectX shaders embedded) |

## Parameters

- **View** – Result, or Grain Only (the noise on 50 % grey, for matching by eye)
- **Intensity / Size / Softness / Aspect Ratio** – the usual matching controls
- **Color** – Saturation, Monochromatic, Tint Color, Tint Amount
- **Channel Intensity / Channel Size** – per-channel multipliers (e.g. coarser blue grain)
- **Application** – Blending Mode (Film, Multiply, Add, Screen, Overlay), Shadows / Midtones / Highlights response with Midpoint, Blend with Original
- **Channel Balance** – per-channel Shadows / Midtones / Highlights
- **Animation** – Random Seed, Animation Speed, Animate Smoothly (variance-preserving crossfade between grain frames)

`Film` mode applies the grain multiplicatively in log space (`src * exp(g)`), which is the
natural mode for linear-light 32-bit projects.

## How it renders

All maths lives in `src/FastGrain_Shared.h`, compiled three times (MSVC, nvcc, dxc):

1. **GenLattice** – a deterministic Gaussian lattice per channel (R, G, B, mono), one cell per
   grain, seeded by cell index + frame + seed. Cells are laid out in absolute layer coordinates,
   so any render region or downsample factor reproduces the same grain.
2. **BlurH / BlurV** – optional separable Gaussian on the lattice (Softness), normalised so the
   variance does not change.
3. **Composite** – Catmull-Rom sampling of the lattice (variance-compensated), saturation /
   tint, luma response curve, blend mode, mix with original.

## Building (Windows)

Requirements: Visual Studio 2022 (v143), CUDA Toolkit 12.8 (`CUDA_PATH_V12_8`), a Windows SDK
with `dxc.exe` (10.0.26100 used), the AE SDK extracted next to this folder
(`..\AfterEffectsSDK_25.6_61_win\ae25.6_61.64bit.AfterEffectsSDK`). Boost is **not** required.

```powershell
.\build.ps1                 # Release -> build\Release\FastGrain.aex
.\build.ps1 -Install        # also copies into ...\Common\Plug-ins\7.0\MediaCore (UAC prompt)
```

Override tool locations with `/p:` style properties, e.g. `msbuild FastGrain.vcxproj /p:CudaRoot=...`.

## Tests

- `test\fg_test.cpp` – standalone CPU harness for the shared maths (statistics, ROI
  independence, softness / temporal variance, blend modes, timing).
- `test\ae_test.jsx` – run with `AfterFX.exe -r test\ae_test.jsx`; applies the effect, renders
  CPU and CUDA frames to PNG and benchmarks against Adobe's Add Grain.

## Notes

- OpenCL and Metal are not implemented; AE falls back to the CPU path automatically.
- The PiPL flag values in `src\FastGrainPiPL.r` must match `FG_OUT_FLAGS*` in `src\FastGrain.h`.
