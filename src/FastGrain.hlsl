/*
    FastGrain.hlsl - DirectX 12 compute entry points.
    Compiled four times by dxc (-E GenLattice / BlurH / BlurV / Composite), cs_6_5.
    All four share one root signature so the host binds identically for every pass:
        b0  : FGParams constant buffer
        t0  : lattice input (raw float buffer)      t1 : source image (BGRA float4)
        u0  : output (lattice/temp/destination)
*/
#define FG_HLSL 1

ByteAddressBuffer   gLattice : register(t0);
ByteAddressBuffer   gSrc     : register(t1);
RWByteAddressBuffer gOut     : register(u0);

#include "FastGrain_Shared.h"

cbuffer FGConstants : register(b0)
{
    FGParams P;
};

#define FG_ROOT_SIG "RootFlags(0), CBV(b0), DescriptorTable(SRV(t0, numDescriptors=2)), DescriptorTable(UAV(u0, numDescriptors=1))"

[RootSignature(FG_ROOT_SIG)]
[numthreads(16, 16, 1)]
void GenLattice(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= (uint)P.lattMaxW || id.y >= (uint)P.lattTotalH) return;
    float v = fg_gen_cell((int)id.x, (int)id.y, P);
    gOut.Store((id.y * (uint)P.lattPitch + id.x) * 4, asuint(v));
}

[RootSignature(FG_ROOT_SIG)]
[numthreads(16, 16, 1)]
void BlurH(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= (uint)P.lattMaxW || id.y >= (uint)P.lattTotalH) return;
    float v = fg_blur_h_cell((int)id.x, (int)id.y, P);
    gOut.Store((id.y * (uint)P.lattPitch + id.x) * 4, asuint(v));
}

[RootSignature(FG_ROOT_SIG)]
[numthreads(16, 16, 1)]
void BlurV(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= (uint)P.lattMaxW || id.y >= (uint)P.lattTotalH) return;
    float v = fg_blur_v_cell((int)id.x, (int)id.y, P);
    gOut.Store((id.y * (uint)P.lattPitch + id.x) * 4, asuint(v));
}

[RootSignature(FG_ROOT_SIG)]
[numthreads(16, 16, 1)]
void Composite(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= (uint)P.width || id.y >= (uint)P.height) return;

    uint   si = (id.y * (uint)P.srcPitch + id.x) * 16;
    float4 s  = asfloat(gSrc.Load4(si));            /* BGRA */

    FGPixel px;
    px.r = s.z; px.g = s.y; px.b = s.x; px.a = s.w;

    float X = P.originX + (float)id.x * P.dsx;
    float Y = P.originY + (float)id.y * P.dsy;

    FGPixel r = fg_composite(P, px, X, Y);

    uint di = (id.y * (uint)P.dstPitch + id.x) * 16;
    gOut.Store4(di, asuint(float4(r.b, r.g, r.r, r.a)));
}
