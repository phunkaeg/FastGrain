/*
    FastGrain_Kernels.cu - CUDA implementation of the four grain passes.
    All maths lives in FastGrain_Shared.h; this file only does addressing and launches.
*/
#include <cuda_runtime.h>
#include "FastGrain_Shared.h"
#include "FastGrain_CUDA.h"

__global__ void fgGenLattice(FGParams P, float* latt)
{
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= P.lattMaxW || row >= P.lattTotalH) return;
    latt[row * P.lattPitch + i] = fg_gen_cell(i, row, P);
}

__global__ void fgBlurH(FGParams P, const float* latt, float* outBuf)
{
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= P.lattMaxW || row >= P.lattTotalH) return;
    outBuf[row * P.lattPitch + i] = fg_blur_h_cell(latt, i, row, P);
}

__global__ void fgBlurV(FGParams P, const float* latt, float* outBuf)
{
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= P.lattMaxW || row >= P.lattTotalH) return;
    outBuf[row * P.lattPitch + i] = fg_blur_v_cell(latt, i, row, P);
}

__global__ void fgComposite(FGParams P, const float4* src, float4* dst, const float* latt)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= P.width || y >= P.height) return;

    float4 s = src[y * P.srcPitch + x];          /* BGRA */
    FGPixel px;
    px.r = s.z; px.g = s.y; px.b = s.x; px.a = s.w;

    float X = P.originX + (float)x * P.dsx;
    float Y = P.originY + (float)y * P.dsy;

    FGPixel r = fg_composite(latt, P, px, X, Y);
    dst[y * P.dstPitch + x] = make_float4(r.b, r.g, r.r, r.a);
}

int FG_CudaRender(const FGParams& P,
                  const void* src, void* dst,
                  void* latt, void* temp,
                  void* stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    dim3 block(16, 16, 1);
    dim3 gridL((P.lattMaxW + 15) / 16, (P.lattTotalH + 15) / 16, 1);
    dim3 gridC((P.width + 15) / 16, (P.height + 15) / 16, 1);

    fgGenLattice<<<gridL, block, 0, s>>>(P, (float*)latt);
    if (P.blurRadius > 0 && temp) {
        fgBlurH<<<gridL, block, 0, s>>>(P, (const float*)latt, (float*)temp);
        fgBlurV<<<gridL, block, 0, s>>>(P, (const float*)temp, (float*)latt);
    }
    fgComposite<<<gridC, block, 0, s>>>(P, (const float4*)src, (float4*)dst, (const float*)latt);

    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    e = cudaStreamSynchronize(s);
    return (int)e;
}
