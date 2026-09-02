#pragma once
/* Host-side entry into the CUDA kernels (implemented in FastGrain_Kernels.cu). */

struct FGParams;

/* Runs all passes on `stream` and blocks until finished.
   src/dst are BGRA float4 device buffers, latt/temp are raw float device buffers.
   temp may be null when P.blurRadius == 0. Returns 0 on success, else a cudaError_t. */
int FG_CudaRender(const FGParams& P,
                  const void* src, void* dst,
                  void* latt, void* temp,
                  void* stream);
