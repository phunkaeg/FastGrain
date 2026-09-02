/*
    FastGrain_Shared.h

    Grain math shared verbatim by three compilers:
      - MSVC  (CPU fallback, 8/16/32-bit)
      - nvcc  (CUDA kernels)
      - dxc   (DirectX 12 compute shaders, HLSL)

    Rules for code in this file (so it stays valid in all three languages):
      - scalars only: float / int / uint; structs passed by value
      - no pointers, references or "out" params (HLSL/C++ differ); return structs
      - no variables named in/out/inout/sample/line/point/linear (HLSL keywords)
      - math through the FG_* macros below
      - lattice access only through FG_LATT_READ / FG_LATT_PARAMS / FG_LATT_ARGS
*/

#ifndef FASTGRAIN_SHARED_H
#define FASTGRAIN_SHARED_H

/* ------------------------------------------------------------------ */
/* per-language prelude                                                */
/* ------------------------------------------------------------------ */
#if defined(FG_HLSL)
    #define FG_FUNC
    #define FG_SQRT(x)   sqrt(x)
    #define FG_RSQRT(x)  rsqrt(x)
    #define FG_LOG(x)    log(x)
    #define FG_EXP(x)    exp(x)
    #define FG_COS(x)    cos(x)
    #define FG_FLOOR(x)  floor(x)
    #define FG_ABS(x)    abs(x)
    #define FG_MIN(a,b)  min(a,b)
    #define FG_MAX(a,b)  max(a,b)
    #define FG_UNROLL    [unroll]
    #define FG_LATT_PARAMS
    #define FG_LATT_ARGS
    #define FG_LATT_READ(idx) asfloat(gLattice.Load((idx) * 4))
#elif defined(__CUDACC__)
    #define FG_FUNC      static __device__ __forceinline__
    #define FG_SQRT(x)   sqrtf(x)
    #define FG_RSQRT(x)  rsqrtf(x)
    #define FG_LOG(x)    logf(x)
    #define FG_EXP(x)    expf(x)
    #define FG_COS(x)    cosf(x)
    #define FG_FLOOR(x)  floorf(x)
    #define FG_ABS(x)    fabsf(x)
    #define FG_MIN(a,b)  fminf(a,b)
    #define FG_MAX(a,b)  fmaxf(a,b)
    #define FG_UNROLL    _Pragma("unroll")
    typedef unsigned int uint;
    #define FG_LATT_PARAMS const float* latt,
    #define FG_LATT_ARGS   latt,
    #define FG_LATT_READ(idx) latt[idx]
#else /* plain C++ on the CPU */
    #include <math.h>
    #define FG_FUNC      static inline
    #define FG_SQRT(x)   sqrtf(x)
    #define FG_RSQRT(x)  (1.0f / sqrtf(x))
    #define FG_LOG(x)    logf(x)
    #define FG_EXP(x)    expf(x)
    #define FG_COS(x)    cosf(x)
    #define FG_FLOOR(x)  floorf(x)
    #define FG_ABS(x)    fabsf(x)
    #define FG_MIN(a,b)  ((a) < (b) ? (a) : (b))
    #define FG_MAX(a,b)  ((a) > (b) ? (a) : (b))
    #define FG_UNROLL
    typedef unsigned int uint;
    #define FG_LATT_PARAMS const float* latt,
    #define FG_LATT_ARGS   latt,
    #define FG_LATT_READ(idx) latt[idx]
#endif

/* ------------------------------------------------------------------ */
/* constants                                                           */
/* ------------------------------------------------------------------ */
#define FG_BLEND_FILM      0
#define FG_BLEND_MULTIPLY  1
#define FG_BLEND_ADD       2
#define FG_BLEND_SCREEN    3
#define FG_BLEND_OVERLAY   4

#define FG_VIEW_RESULT     0
#define FG_VIEW_GRAIN      1

#define FG_MAX_BLUR_RADIUS 9

/* ------------------------------------------------------------------ */
/* parameter block: 4-byte scalars only, in this exact order.          */
/* Mirrored 1:1 into a DX12 constant buffer, a CUDA kernel argument     */
/* and the CPU render refcon.                                           */
/* ------------------------------------------------------------------ */
struct FGParams
{
    /* output world */
    int   width;
    int   height;
    int   srcPitch;        /* in pixels (float4) */
    int   dstPitch;        /* in pixels (float4) */

    /* lattice scratch world (planar float, 4 planes stacked vertically) */
    int   lattPitch;       /* in floats */
    int   lattMaxW;        /* widest plane, in floats */
    int   lattTotalH;      /* total rows across all planes */
    int   blurRadius;      /* 0 = no softness pass */
    float blurSigma;       /* in lattice cells */
    float blurNorm;        /* host-computed fg_blur_norm(blurRadius, blurSigma) */

    /* output pixel (0,0) -> full-resolution layer coordinate */
    float originX;
    float originY;
    float dsx;             /* full-res pixels per output pixel */
    float dsy;

    /* plane 0 = red, 1 = green, 2 = blue, 3 = mono */
    int   p0RowOff; int p0DimsW; int p0DimsH; int p0BaseI; int p0BaseJ; float p0InvSx; float p0InvSy;
    int   p1RowOff; int p1DimsW; int p1DimsH; int p1BaseI; int p1BaseJ; float p1InvSx; float p1InvSy;
    int   p2RowOff; int p2DimsW; int p2DimsH; int p2BaseI; int p2BaseJ; float p2InvSx; float p2InvSy;
    int   p3RowOff; int p3DimsW; int p3DimsH; int p3BaseI; int p3BaseJ; float p3InvSx; float p3InvSy;

    /* animation */
    int   frameA;
    int   frameB;
    float frameBlend;      /* 0 = frameA only */
    int   seed;

    /* colour */
    float saturation;      /* 0 = mono, 1 = fully independent per channel */
    float tintR; float tintG; float tintB;
    float ampR;  float ampG;  float ampB;      /* K * intensity * channel intensity */
    float shR;   float shG;   float shB;       /* shadows   (global * channel) */
    float miR;   float miG;   float miB;       /* midtones */
    float hiR;   float hiG;   float hiB;       /* highlights */
    float midpoint;

    /* application */
    int   blendMode;
    float mixOriginal;     /* 0..1, fraction of original blended back */
    int   viewMode;
    int   clampOutput;     /* 1 for integer bit depths */
};

struct FGPixel { float r; float g; float b; float a; };
struct FGTone  { float s; float m; float h; };

/* ------------------------------------------------------------------ */
/* hashing / gaussian                                                  */
/* ------------------------------------------------------------------ */
FG_FUNC uint fg_mix(uint h)
{
    /* lowbias32 finaliser */
    h ^= h >> 16;
    h *= 0x7FEB352Du;
    h ^= h >> 15;
    h *= 0x846CA68Bu;
    h ^= h >> 16;
    return h;
}

FG_FUNC uint fg_hash(int I, int J, int F, uint S)
{
    uint h = S * 0x9E3779B1u + 0x7F4A7C15u;
    h = (h ^ (uint)I) * 0x85EBCA77u; h ^= h >> 13;
    h = (h ^ (uint)J) * 0xC2B2AE3Du; h ^= h >> 16;
    h = (h ^ (uint)F) * 0x27D4EB2Fu; h ^= h >> 15;
    return fg_mix(h);
}

/* unit-variance gaussian, deterministic in (I, J, frame, seed) */
FG_FUNC float fg_gauss(int I, int J, int F, uint S)
{
    uint  h1 = fg_hash(I, J, F, S);
    uint  h2 = fg_mix(h1 ^ 0x5BD1E995u);
    float u1 = (float)(h1 >> 8) * (1.0f / 16777216.0f) + (0.5f / 16777216.0f); /* (0,1) */
    float u2 = (float)(h2 >> 8) * (1.0f / 16777216.0f);
    float r  = FG_SQRT(-2.0f * FG_LOG(u1));
    return r * FG_COS(6.28318530718f * u2);
}

/* lattice value for one cell, including optional temporal blend */
FG_FUNC float fg_lattice_value(int I, int J, int plane, FGParams P)
{
    uint S = (uint)P.seed * 16u + (uint)plane;
    float a = fg_gauss(I, J, P.frameA, S);
    if (P.frameBlend <= 0.0f) {
        return a;
    }
    float b  = fg_gauss(I, J, P.frameB, S);
    float wa = 1.0f - P.frameBlend;
    float wb = P.frameBlend;
    return (a * wa + b * wb) * FG_RSQRT(wa * wa + wb * wb);
}

/* ------------------------------------------------------------------ */
/* separable gaussian blur helpers (softness)                          */
/* ------------------------------------------------------------------ */
FG_FUNC float fg_blur_weight(int k, float sigma)
{
    float x = (float)k;
    return FG_EXP(-(x * x) / (2.0f * sigma * sigma));
}

/* returns the 1-D normalisation so the blurred field keeps unit variance:
   (1/sum w) * (1/sqrt(sum (w/sum w)^2)) = 1/sqrt(sum w^2) */
FG_FUNC float fg_blur_norm(int radius, float sigma)
{
    float sumSq = 0.0f;
    for (int k = -FG_MAX_BLUR_RADIUS; k <= FG_MAX_BLUR_RADIUS; k++) {
        if (k >= -radius && k <= radius) {
            float w = fg_blur_weight(k, sigma);
            sumSq += w * w;
        }
    }
    return FG_RSQRT(sumSq);
}

/* ------------------------------------------------------------------ */
/* plane lookup by lattice row                                         */
/* ------------------------------------------------------------------ */
FG_FUNC int fg_plane_of_row(int row, FGParams P)
{
    if (row >= P.p3RowOff) return 3;
    if (row >= P.p2RowOff) return 2;
    if (row >= P.p1RowOff) return 1;
    return 0;
}

FG_FUNC int fg_plane_rowoff(int p, FGParams P) { return p == 0 ? P.p0RowOff : (p == 1 ? P.p1RowOff : (p == 2 ? P.p2RowOff : P.p3RowOff)); }
FG_FUNC int fg_plane_dimsw (int p, FGParams P) { return p == 0 ? P.p0DimsW  : (p == 1 ? P.p1DimsW  : (p == 2 ? P.p2DimsW  : P.p3DimsW)); }
FG_FUNC int fg_plane_dimsh (int p, FGParams P) { return p == 0 ? P.p0DimsH  : (p == 1 ? P.p1DimsH  : (p == 2 ? P.p2DimsH  : P.p3DimsH)); }
FG_FUNC int fg_plane_basei (int p, FGParams P) { return p == 0 ? P.p0BaseI  : (p == 1 ? P.p1BaseI  : (p == 2 ? P.p2BaseI  : P.p3BaseI)); }
FG_FUNC int fg_plane_basej (int p, FGParams P) { return p == 0 ? P.p0BaseJ  : (p == 1 ? P.p1BaseJ  : (p == 2 ? P.p2BaseJ  : P.p3BaseJ)); }

/* ------------------------------------------------------------------ */
/* pass 1: generate lattice cell (i,row) of the stacked planar buffer  */
/* returns the value to store; cells outside a plane's width get 0     */
/* ------------------------------------------------------------------ */
FG_FUNC float fg_gen_cell(int i, int row, FGParams P)
{
    int p  = fg_plane_of_row(row, P);
    int w  = fg_plane_dimsw(p, P);
    if (i >= w) return 0.0f;
    int I  = fg_plane_basei(p, P) + i;
    int J  = fg_plane_basej(p, P) + (row - fg_plane_rowoff(p, P));
    return fg_lattice_value(I, J, p, P);
}

/* pass 2: horizontal blur of cell (i,row); reads FG_LATT_READ */
FG_FUNC float fg_blur_h_cell(FG_LATT_PARAMS int i, int row, FGParams P)
{
    int p = fg_plane_of_row(row, P);
    int w = fg_plane_dimsw(p, P);
    if (i >= w) return 0.0f;
    int   rowBase = row * P.lattPitch;
    float acc = 0.0f;
    for (int k = -FG_MAX_BLUR_RADIUS; k <= FG_MAX_BLUR_RADIUS; k++) {
        if (k >= -P.blurRadius && k <= P.blurRadius) {
            int ii = i + k;
            ii = ii < 0 ? 0 : (ii >= w ? w - 1 : ii);
            acc += fg_blur_weight(k, P.blurSigma) * FG_LATT_READ(rowBase + ii);
        }
    }
    return acc * P.blurNorm;
}

/* pass 3: vertical blur of cell (i,row) */
FG_FUNC float fg_blur_v_cell(FG_LATT_PARAMS int i, int row, FGParams P)
{
    int p = fg_plane_of_row(row, P);
    int w = fg_plane_dimsw(p, P);
    if (i >= w) return 0.0f;
    int   off = fg_plane_rowoff(p, P);
    int   h   = fg_plane_dimsh(p, P);
    int   j   = row - off;
    float acc = 0.0f;
    for (int k = -FG_MAX_BLUR_RADIUS; k <= FG_MAX_BLUR_RADIUS; k++) {
        if (k >= -P.blurRadius && k <= P.blurRadius) {
            int jj = j + k;
            jj = jj < 0 ? 0 : (jj >= h ? h - 1 : jj);
            acc += fg_blur_weight(k, P.blurSigma) * FG_LATT_READ((off + jj) * P.lattPitch + i);
        }
    }
    return acc * P.blurNorm;
}

/* ------------------------------------------------------------------ */
/* pass 4 helpers: Catmull-Rom sampling of one plane, variance-        */
/* compensated so intensity does not pulse with sub-cell phase          */
/* ------------------------------------------------------------------ */
FG_FUNC float fg_sample_plane(FG_LATT_PARAMS int rowOff, int dimsW, int dimsH, int baseI, int baseJ,
                              float invSx, float invSy, int pitch, float X, float Y)
{
    float u  = X * invSx - (float)baseI;
    float v  = Y * invSy - (float)baseJ;
    float fu = FG_FLOOR(u);
    float fv = FG_FLOOR(v);
    float tx = u - fu;
    float ty = v - fv;
    int   i0 = (int)fu - 1;
    int   j0 = (int)fv - 1;

    float wx0 = ((-0.5f * tx + 1.0f) * tx - 0.5f) * tx;
    float wx1 = (1.5f * tx - 2.5f) * tx * tx + 1.0f;
    float wx2 = ((-1.5f * tx + 2.0f) * tx + 0.5f) * tx;
    float wx3 = (0.5f * tx - 0.5f) * tx * tx;
    float wy0 = ((-0.5f * ty + 1.0f) * ty - 0.5f) * ty;
    float wy1 = (1.5f * ty - 2.5f) * ty * ty + 1.0f;
    float wy2 = ((-1.5f * ty + 2.0f) * ty + 0.5f) * ty;
    float wy3 = (0.5f * ty - 0.5f) * ty * ty;

    int ia = i0 < 0 ? 0 : (i0 > dimsW - 1 ? dimsW - 1 : i0);
    int ib = i0 + 1 < 0 ? 0 : (i0 + 1 > dimsW - 1 ? dimsW - 1 : i0 + 1);
    int ic = i0 + 2 < 0 ? 0 : (i0 + 2 > dimsW - 1 ? dimsW - 1 : i0 + 2);
    int id = i0 + 3 < 0 ? 0 : (i0 + 3 > dimsW - 1 ? dimsW - 1 : i0 + 3);
    int ja = j0 < 0 ? 0 : (j0 > dimsH - 1 ? dimsH - 1 : j0);
    int jb = j0 + 1 < 0 ? 0 : (j0 + 1 > dimsH - 1 ? dimsH - 1 : j0 + 1);
    int jc = j0 + 2 < 0 ? 0 : (j0 + 2 > dimsH - 1 ? dimsH - 1 : j0 + 2);
    int jd = j0 + 3 < 0 ? 0 : (j0 + 3 > dimsH - 1 ? dimsH - 1 : j0 + 3);

    int ra = (rowOff + ja) * pitch;
    int rb = (rowOff + jb) * pitch;
    int rc = (rowOff + jc) * pitch;
    int rd = (rowOff + jd) * pitch;

    float s0 = wx0 * FG_LATT_READ(ra + ia) + wx1 * FG_LATT_READ(ra + ib) + wx2 * FG_LATT_READ(ra + ic) + wx3 * FG_LATT_READ(ra + id);
    float s1 = wx0 * FG_LATT_READ(rb + ia) + wx1 * FG_LATT_READ(rb + ib) + wx2 * FG_LATT_READ(rb + ic) + wx3 * FG_LATT_READ(rb + id);
    float s2 = wx0 * FG_LATT_READ(rc + ia) + wx1 * FG_LATT_READ(rc + ib) + wx2 * FG_LATT_READ(rc + ic) + wx3 * FG_LATT_READ(rc + id);
    float s3 = wx0 * FG_LATT_READ(rd + ia) + wx1 * FG_LATT_READ(rd + ib) + wx2 * FG_LATT_READ(rd + ic) + wx3 * FG_LATT_READ(rd + id);

    float acc  = wy0 * s0 + wy1 * s1 + wy2 * s2 + wy3 * s3;
    float sqx  = wx0 * wx0 + wx1 * wx1 + wx2 * wx2 + wx3 * wx3;
    float sqy  = wy0 * wy0 + wy1 * wy1 + wy2 * wy2 + wy3 * wy3;
    return acc * FG_RSQRT(sqx * sqy);
}

FG_FUNC float fg_luma(float r, float g, float b)
{
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

FG_FUNC FGTone fg_tone_weights(float lum, float mid)
{
    FGTone t;
    lum = FG_MIN(FG_MAX(lum, 0.0f), 1.0f);
    float x = lum <= mid ? 0.5f * lum / mid : 0.5f + 0.5f * (lum - mid) / (1.0f - mid);
    t.s = (1.0f - x) * (1.0f - x);
    t.m = 2.0f * x * (1.0f - x);
    t.h = x * x;
    return t;
}

FG_FUNC float fg_apply_blend(int mode, float src, float g, float a)
{
    if (mode == FG_BLEND_FILM)     return src * FG_EXP(g);
    if (mode == FG_BLEND_MULTIPLY) return src * (1.0f + g);
    if (mode == FG_BLEND_ADD)      return src + g * a;
    if (mode == FG_BLEND_SCREEN)   return src + g * (a - src);
    /* overlay: multiply below mid-grey, screen above */
    return src < 0.5f * a ? src * (1.0f + 2.0f * g) : src + 2.0f * g * (a - src);
}

/* ------------------------------------------------------------------ */
/* pass 4: composite one pixel. X,Y are full-resolution layer coords.  */
/* src is premultiplied RGBA in [0,1] (float depth may exceed 1).       */
/* ------------------------------------------------------------------ */
FG_FUNC FGPixel fg_composite(FG_LATT_PARAMS FGParams P, FGPixel src, float X, float Y)
{
    float nR = fg_sample_plane(FG_LATT_ARGS P.p0RowOff, P.p0DimsW, P.p0DimsH, P.p0BaseI, P.p0BaseJ, P.p0InvSx, P.p0InvSy, P.lattPitch, X, Y);
    float nG = fg_sample_plane(FG_LATT_ARGS P.p1RowOff, P.p1DimsW, P.p1DimsH, P.p1BaseI, P.p1BaseJ, P.p1InvSx, P.p1InvSy, P.lattPitch, X, Y);
    float nB = fg_sample_plane(FG_LATT_ARGS P.p2RowOff, P.p2DimsW, P.p2DimsH, P.p2BaseI, P.p2BaseJ, P.p2InvSx, P.p2InvSy, P.lattPitch, X, Y);
    float nM = fg_sample_plane(FG_LATT_ARGS P.p3RowOff, P.p3DimsW, P.p3DimsH, P.p3BaseI, P.p3BaseJ, P.p3InvSx, P.p3InvSy, P.lattPitch, X, Y);

    /* saturation: variance-preserving blend between mono and per-channel noise */
    float sat = P.saturation;
    float wm  = 1.0f - sat;
    float nrm = FG_RSQRT(wm * wm + sat * sat);
    float gR  = (nM * wm + nR * sat) * nrm * P.tintR;
    float gG  = (nM * wm + nG * sat) * nrm * P.tintG;
    float gB  = (nM * wm + nB * sat) * nrm * P.tintB;

    /* tonal response from un-premultiplied luma */
    float a   = src.a;
    float inv = a > 1.0e-6f ? 1.0f / a : 0.0f;
    float lum = fg_luma(src.r * inv, src.g * inv, src.b * inv);
    FGTone w  = fg_tone_weights(lum, P.midpoint);

    gR *= P.ampR * (w.s * P.shR + w.m * P.miR + w.h * P.hiR);
    gG *= P.ampG * (w.s * P.shG + w.m * P.miG + w.h * P.hiG);
    gB *= P.ampB * (w.s * P.shB + w.m * P.miB + w.h * P.hiB);

    FGPixel res;
    if (P.viewMode == FG_VIEW_GRAIN) {
        res.r = 0.5f + gR;
        res.g = 0.5f + gG;
        res.b = 0.5f + gB;
        res.a = 1.0f;
        if (P.clampOutput != 0) {
            res.r = FG_MIN(FG_MAX(res.r, 0.0f), 1.0f);
            res.g = FG_MIN(FG_MAX(res.g, 0.0f), 1.0f);
            res.b = FG_MIN(FG_MAX(res.b, 0.0f), 1.0f);
        }
        return res;
    }

    res.r = fg_apply_blend(P.blendMode, src.r, gR, a);
    res.g = fg_apply_blend(P.blendMode, src.g, gG, a);
    res.b = fg_apply_blend(P.blendMode, src.b, gB, a);
    res.a = a;

    float mix = P.mixOriginal;
    res.r = res.r + (src.r - res.r) * mix;
    res.g = res.g + (src.g - res.g) * mix;
    res.b = res.b + (src.b - res.b) * mix;

    if (P.clampOutput != 0) {
        res.r = FG_MIN(FG_MAX(res.r, 0.0f), a);
        res.g = FG_MIN(FG_MAX(res.g, 0.0f), a);
        res.b = FG_MIN(FG_MAX(res.b, 0.0f), a);
    }
    return res;
}

#endif /* FASTGRAIN_SHARED_H */
