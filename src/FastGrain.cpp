/*
    FastGrain.cpp - After Effects glue for the Fast Grain effect.

    Render pipeline (identical on CPU, CUDA and DirectX):
      1. GenLattice : deterministic gaussian lattice, one plane per channel (R,G,B,mono),
                      cell spacing = grain size, seeded by (cell, frame, seed)
      2. BlurH/V    : optional separable gaussian on the lattice (Softness)
      3. Composite  : Catmull-Rom sample the lattice per pixel, tonal response,
                      saturation/tint, blend mode, mix with original
*/

#include "FastGrain.h"
#include "FastGrain_CUDA.h"
#include "FastGrain_DX.h"
#include "FastGrain_Shaders.h"     /* generated at build time from FastGrain.hlsl */

#include <vector>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <algorithm>

/* ------------------------------------------------------------------ */
/* data passed from PreRender to SmartRender                           */
/* ------------------------------------------------------------------ */
struct FGPreRenderData
{
    FGParams P;          /* everything except pitches / actual world dims */
    PF_LRect rect;       /* result rect in (downsampled) layer coords */
    float    sizeX[4];   /* per-plane cell size in full-res pixels */
    float    sizeY[4];
};

/* per-GPU-device data, allocated in GPU_DEVICE_SETUP */
struct FGGPUData
{
    FGDX* dx;            /* null for CUDA */
};

/* ------------------------------------------------------------------ */
/* small helpers                                                       */
/* ------------------------------------------------------------------ */
static inline bool RectEmpty(const PF_LRect& r)
{
    return r.right <= r.left || r.bottom <= r.top;
}

static PF_Err CheckoutFloat(PF_InData* in_data, int idx, float& value)
{
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    PF_Err err = PF_CHECKOUT_PARAM(in_data, idx, in_data->current_time, in_data->time_step, in_data->time_scale, &def);
    if (!err) {
        value = (float)def.u.fs_d.value;
        PF_Err err2 = PF_CHECKIN_PARAM(in_data, &def);
        if (!err) err = err2;
    }
    return err;
}

static PF_Err CheckoutPopup(PF_InData* in_data, int idx, int& value)
{
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    PF_Err err = PF_CHECKOUT_PARAM(in_data, idx, in_data->current_time, in_data->time_step, in_data->time_scale, &def);
    if (!err) {
        value = (int)def.u.pd.value;
        PF_Err err2 = PF_CHECKIN_PARAM(in_data, &def);
        if (!err) err = err2;
    }
    return err;
}

static PF_Err CheckoutBool(PF_InData* in_data, int idx, bool& value)
{
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    PF_Err err = PF_CHECKOUT_PARAM(in_data, idx, in_data->current_time, in_data->time_step, in_data->time_scale, &def);
    if (!err) {
        value = def.u.bd.value != 0;
        PF_Err err2 = PF_CHECKIN_PARAM(in_data, &def);
        if (!err) err = err2;
    }
    return err;
}

static PF_Err CheckoutColor(PF_InData* in_data, int idx, PF_Pixel& value)
{
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    PF_Err err = PF_CHECKOUT_PARAM(in_data, idx, in_data->current_time, in_data->time_step, in_data->time_scale, &def);
    if (!err) {
        value = def.u.cd.value;
        PF_Err err2 = PF_CHECKIN_PARAM(in_data, &def);
        if (!err) err = err2;
    }
    return err;
}

/* ------------------------------------------------------------------ */
/* About / GlobalSetup / ParamsSetup                                   */
/* ------------------------------------------------------------------ */
static PF_Err About(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
{
    PF_SPRINTF(out_data->return_msg, "%s v%d.%d\r%s", FG_NAME, FG_MAJOR_VERSION, FG_MINOR_VERSION, FG_DESCRIPTION);
    return PF_Err_NONE;
}

static PF_Err GlobalSetup(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
{
    out_data->my_version = PF_VERSION(FG_MAJOR_VERSION, FG_MINOR_VERSION, FG_BUG_VERSION, FG_STAGE_VERSION, FG_BUILD_VERSION);
    out_data->out_flags  = FG_OUT_FLAGS;
    out_data->out_flags2 = FG_OUT_FLAGS2;
    return PF_Err_NONE;
}

static PF_Err ParamsSetup(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
{
    PF_ParamDef def;
    const PF_ParamFlags collapsed = PF_ParamFlag_START_COLLAPSED;

    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("View", 2, 1, "Result|Grain Only", FG_ID(FG_VIEW));

    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Intensity", 0.0f, 100.0f, 0.0f, 5.0f, 1.0f, PF_Precision_HUNDREDTHS, 0, 0, FG_ID(FG_INTENSITY));
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Size", 0.25f, 100.0f, 0.5f, 10.0f, 1.0f, PF_Precision_HUNDREDTHS, 0, 0, FG_ID(FG_SIZE));
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Softness", 0.0f, 3.0f, 0.0f, 3.0f, 0.5f, PF_Precision_HUNDREDTHS, 0, 0, FG_ID(FG_SOFTNESS));
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Aspect Ratio", 0.1f, 10.0f, 0.25f, 4.0f, 1.0f, PF_Precision_HUNDREDTHS, 0, 0, FG_ID(FG_ASPECT));

    AEFX_CLR_STRUCT(def);
    def.flags = 0;
    PF_ADD_TOPIC("Color", FG_ID(FG_COLOR_TOPIC));
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Saturation", 0.0f, 100.0f, 0.0f, 100.0f, 100.0f, PF_Precision_INTEGER, PF_ValueDisplayFlag_PERCENT, 0, FG_ID(FG_SATURATION));
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Monochromatic", 0, 0, FG_ID(FG_MONO));
    AEFX_CLR_STRUCT(def);
    PF_ADD_COLOR("Tint Color", 255, 255, 255, FG_ID(FG_TINT_COLOR));
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Tint Amount", 0.0f, 100.0f, 0.0f, 100.0f, 0.0f, PF_Precision_INTEGER, PF_ValueDisplayFlag_PERCENT, 0, FG_ID(FG_TINT_AMOUNT));
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(FG_ID(FG_COLOR_END));

    AEFX_CLR_STRUCT(def);
    def.flags = collapsed;
    PF_ADD_TOPIC("Channel Intensity", FG_ID(FG_CHANINT_TOPIC));
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Red Intensity",   0.0f, 100.0f, 0.0f, 5.0f, 1.0f, PF_Precision_HUNDREDTHS, 0, 0, FG_ID(FG_INT_R));
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Green Intensity", 0.0f, 100.0f, 0.0f, 5.0f, 1.0f, PF_Precision_HUNDREDTHS, 0, 0, FG_ID(FG_INT_G));
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Blue Intensity",  0.0f, 100.0f, 0.0f, 5.0f, 1.0f, PF_Precision_HUNDREDTHS, 0, 0, FG_ID(FG_INT_B));
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(FG_ID(FG_CHANINT_END));

    AEFX_CLR_STRUCT(def);
    def.flags = collapsed;
    PF_ADD_TOPIC("Channel Size", FG_ID(FG_CHANSIZE_TOPIC));
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Red Size",   0.1f, 100.0f, 0.25f, 4.0f, 1.0f, PF_Precision_HUNDREDTHS, 0, 0, FG_ID(FG_SIZE_R));
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Green Size", 0.1f, 100.0f, 0.25f, 4.0f, 1.0f, PF_Precision_HUNDREDTHS, 0, 0, FG_ID(FG_SIZE_G));
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Blue Size",  0.1f, 100.0f, 0.25f, 4.0f, 1.0f, PF_Precision_HUNDREDTHS, 0, 0, FG_ID(FG_SIZE_B));
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(FG_ID(FG_CHANSIZE_END));

    AEFX_CLR_STRUCT(def);
    def.flags = 0;
    PF_ADD_TOPIC("Application", FG_ID(FG_APP_TOPIC));
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP("Blending Mode", 5, 1, "Film|Multiply|Add|Screen|Overlay", FG_ID(FG_BLEND_MODE));
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Shadows",    0.0f, 10.0f, 0.0f, 2.0f, 1.0f, PF_Precision_HUNDREDTHS, 0, 0, FG_ID(FG_SHADOWS));
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Midtones",   0.0f, 10.0f, 0.0f, 2.0f, 1.0f, PF_Precision_HUNDREDTHS, 0, 0, FG_ID(FG_MIDTONES));
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Highlights", 0.0f, 10.0f, 0.0f, 2.0f, 1.0f, PF_Precision_HUNDREDTHS, 0, 0, FG_ID(FG_HIGHLIGHTS));
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Midpoint",   0.01f, 0.99f, 0.05f, 0.95f, 0.5f, PF_Precision_HUNDREDTHS, 0, 0, FG_ID(FG_MIDPOINT));
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Blend with Original", 0.0f, 100.0f, 0.0f, 100.0f, 0.0f, PF_Precision_INTEGER, PF_ValueDisplayFlag_PERCENT, 0, FG_ID(FG_MIX));
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(FG_ID(FG_APP_END));

    AEFX_CLR_STRUCT(def);
    def.flags = collapsed;
    PF_ADD_TOPIC("Channel Balance", FG_ID(FG_BAL_TOPIC));
    {
        static const char* names[9] = {
            "Red Shadows",   "Red Midtones",   "Red Highlights",
            "Green Shadows", "Green Midtones", "Green Highlights",
            "Blue Shadows",  "Blue Midtones",  "Blue Highlights" };
        for (int i = 0; i < 9; ++i) {
            AEFX_CLR_STRUCT(def);
            PF_ADD_FLOAT_SLIDERX(names[i], 0.0f, 10.0f, 0.0f, 2.0f, 1.0f, PF_Precision_HUNDREDTHS, 0, 0, FG_ID(FG_R_SH + i));
        }
    }
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(FG_ID(FG_BAL_END));

    AEFX_CLR_STRUCT(def);
    def.flags = 0;
    PF_ADD_TOPIC("Animation", FG_ID(FG_ANIM_TOPIC));
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Random Seed", 0.0f, 100000.0f, 0.0f, 1000.0f, 0.0f, PF_Precision_INTEGER, 0, 0, FG_ID(FG_SEED));
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX("Animation Speed", 0.0f, 100.0f, 0.0f, 4.0f, 1.0f, PF_Precision_HUNDREDTHS, 0, 0, FG_ID(FG_SPEED));
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX("Animate Smoothly", 0, 0, FG_ID(FG_SMOOTH));
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(FG_ID(FG_ANIM_END));

    out_data->num_params = FG_NUM_PARAMS;
    return PF_Err_NONE;
}

/* ------------------------------------------------------------------ */
/* GPU device setup / setdown                                          */
/* ------------------------------------------------------------------ */
static PF_Err GPUDeviceSetup(PF_InData* in_data, PF_OutData* out_data, PF_GPUDeviceSetupExtra* extra)
{
    PF_Err err = PF_Err_NONE;
    const PF_GPU_Framework fw = extra->input->what_gpu;
    if (fw != PF_GPU_Framework_CUDA && fw != PF_GPU_Framework_DIRECTX) {
        return PF_Err_NONE;    /* OpenCL / Metal: not supported, AE falls back to the CPU path */
    }

    AEFX_SuiteScoper<PF_HandleSuite1>    handle_suite(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
    AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpu_suite(in_data, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, out_data);

    PF_GPUDeviceInfo info;
    AEFX_CLR_STRUCT(info);
    ERR(gpu_suite->GetDeviceInfo(in_data->effect_ref, extra->input->device_index, &info));
    if (err) return err;

    FGDX* dx = nullptr;
    if (fw == PF_GPU_Framework_DIRECTX) {
        dx = new (std::nothrow) FGDX();
        bool ok = dx != nullptr;
        if (ok) ok = dx->Init((ID3D12Device*)info.devicePV, (ID3D12CommandQueue*)info.command_queuePV);
        if (ok) ok = dx->LoadShader(FGDX::kGenLattice, fg_GenLattice_cso, fg_GenLattice_cso_len, fg_GenLattice_rs, fg_GenLattice_rs_len);
        if (ok) ok = dx->LoadShader(FGDX::kBlurH,      fg_BlurH_cso,      fg_BlurH_cso_len,      fg_BlurH_rs,      fg_BlurH_rs_len);
        if (ok) ok = dx->LoadShader(FGDX::kBlurV,      fg_BlurV_cso,      fg_BlurV_cso_len,      fg_BlurV_rs,      fg_BlurV_rs_len);
        if (ok) ok = dx->LoadShader(FGDX::kComposite,  fg_Composite_cso,  fg_Composite_cso_len,  fg_Composite_rs,  fg_Composite_rs_len);
        if (!ok) {
            delete dx;
            return PF_Err_NONE;    /* leave GPU flag unset -> CPU fallback */
        }
    }

    PF_Handle h = handle_suite->host_new_handle(sizeof(FGGPUData));
    if (!h) {
        delete dx;
        return PF_Err_OUT_OF_MEMORY;
    }
    FGGPUData* gd = reinterpret_cast<FGGPUData*>(*h);
    gd->dx = dx;
    extra->output->gpu_data = h;
    out_data->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
    return err;
}

static PF_Err GPUDeviceSetdown(PF_InData* in_data, PF_OutData* out_data, PF_GPUDeviceSetdownExtra* extra)
{
    PF_Handle h = (PF_Handle)extra->input->gpu_data;
    if (h) {
        FGGPUData* gd = reinterpret_cast<FGGPUData*>(*h);
        if (gd && gd->dx) {
            delete gd->dx;
            gd->dx = nullptr;
        }
        AEFX_SuiteScoper<PF_HandleSuite1> handle_suite(in_data, kPFHandleSuite, kPFHandleSuiteVersion1, out_data);
        handle_suite->host_dispose_handle(h);
    }
    return PF_Err_NONE;
}

/* ------------------------------------------------------------------ */
/* lattice layout                                                      */
/* ------------------------------------------------------------------ */
static void BuildLayout(FGParams& P, const PF_LRect& rect, const float sizeX[4], const float sizeY[4])
{
    const float Xmin = P.originX;
    const float Xmax = P.originX + (float)(rect.right - rect.left - 1) * P.dsx;
    const float Ymin = P.originY;
    const float Ymax = P.originY + (float)(rect.bottom - rect.top - 1) * P.dsy;
    const int   r    = P.blurRadius;

    int rowOff[4], dimsW[4], dimsH[4], baseI[4], baseJ[4];
    float invSx[4], invSy[4];
    int off = 0, maxW = 0;
    for (int p = 0; p < 4; ++p) {
        invSx[p] = 1.0f / sizeX[p];
        invSy[p] = 1.0f / sizeY[p];
        int Imin = (int)floorf(Xmin * invSx[p]) - 1 - r;
        int Imax = (int)floorf(Xmax * invSx[p]) + 2 + r;
        int Jmin = (int)floorf(Ymin * invSy[p]) - 1 - r;
        int Jmax = (int)floorf(Ymax * invSy[p]) + 2 + r;
        baseI[p]  = Imin;
        baseJ[p]  = Jmin;
        dimsW[p]  = Imax - Imin + 1;
        dimsH[p]  = Jmax - Jmin + 1;
        rowOff[p] = off;
        off  += dimsH[p];
        maxW  = std::max(maxW, dimsW[p]);
    }
    P.p0RowOff = rowOff[0]; P.p0DimsW = dimsW[0]; P.p0DimsH = dimsH[0]; P.p0BaseI = baseI[0]; P.p0BaseJ = baseJ[0]; P.p0InvSx = invSx[0]; P.p0InvSy = invSy[0];
    P.p1RowOff = rowOff[1]; P.p1DimsW = dimsW[1]; P.p1DimsH = dimsH[1]; P.p1BaseI = baseI[1]; P.p1BaseJ = baseJ[1]; P.p1InvSx = invSx[1]; P.p1InvSy = invSy[1];
    P.p2RowOff = rowOff[2]; P.p2DimsW = dimsW[2]; P.p2DimsH = dimsH[2]; P.p2BaseI = baseI[2]; P.p2BaseJ = baseJ[2]; P.p2InvSx = invSx[2]; P.p2InvSy = invSy[2];
    P.p3RowOff = rowOff[3]; P.p3DimsW = dimsW[3]; P.p3DimsH = dimsH[3]; P.p3BaseI = baseI[3]; P.p3BaseJ = baseJ[3]; P.p3InvSx = invSx[3]; P.p3InvSy = invSy[3];
    P.lattMaxW   = maxW;
    P.lattTotalH = off;
    P.lattPitch  = maxW;     /* CPU default; GPU overrides with the world pitch */
}

/* ------------------------------------------------------------------ */
/* PreRender: read parameters, derive everything render needs           */
/* ------------------------------------------------------------------ */
static void DisposePreRenderData(void* p)
{
    if (p) free(p);
}

static PF_Err PreRender(PF_InData* in_data, PF_OutData* out_data, PF_PreRenderExtra* extra)
{
    PF_Err err = PF_Err_NONE;
    PF_RenderRequest req = extra->input->output_request;
    PF_CheckoutResult in_result;
    AEFX_CLR_STRUCT(in_result);

    ERR(extra->cb->checkout_layer(in_data->effect_ref, FG_INPUT, FG_INPUT, &req,
                                  in_data->current_time, in_data->time_step, in_data->time_scale, &in_result));
    if (err) return err;

    /* pixel-independent, transparent stays transparent: output rects = input rects */
    extra->output->result_rect     = in_result.result_rect;
    extra->output->max_result_rect = in_result.max_result_rect;
    extra->output->solid           = in_result.solid;

    FGPreRenderData* pre = reinterpret_cast<FGPreRenderData*>(calloc(1, sizeof(FGPreRenderData)));
    if (!pre) return PF_Err_OUT_OF_MEMORY;
    extra->output->pre_render_data             = pre;
    extra->output->delete_pre_render_data_func = DisposePreRenderData;
    pre->rect = in_result.result_rect;

    if (RectEmpty(pre->rect)) {
        return PF_Err_NONE;          /* nothing to render (or AE only wants max_result_rect) */
    }

    /* ---- parameters ---- */
    int   view = 1, blend = 1;
    bool  mono = false, smooth = false;
    float intensity = 1, size = 1, softness = 0.5f, aspect = 1;
    float saturation = 100, tintAmount = 0;
    float intR = 1, intG = 1, intB = 1, sizeR = 1, sizeG = 1, sizeB = 1;
    float shadows = 1, midtones = 1, highlights = 1, midpoint = 0.5f, mix = 0;
    float bal[9];
    float seed = 0, speed = 1;
    PF_Pixel tint = { 255, 255, 255, 255 };

    ERR(CheckoutPopup(in_data, FG_VIEW, view));
    ERR(CheckoutFloat(in_data, FG_INTENSITY, intensity));
    ERR(CheckoutFloat(in_data, FG_SIZE, size));
    ERR(CheckoutFloat(in_data, FG_SOFTNESS, softness));
    ERR(CheckoutFloat(in_data, FG_ASPECT, aspect));
    ERR(CheckoutFloat(in_data, FG_SATURATION, saturation));
    ERR(CheckoutBool (in_data, FG_MONO, mono));
    ERR(CheckoutColor(in_data, FG_TINT_COLOR, tint));
    ERR(CheckoutFloat(in_data, FG_TINT_AMOUNT, tintAmount));
    ERR(CheckoutFloat(in_data, FG_INT_R, intR));
    ERR(CheckoutFloat(in_data, FG_INT_G, intG));
    ERR(CheckoutFloat(in_data, FG_INT_B, intB));
    ERR(CheckoutFloat(in_data, FG_SIZE_R, sizeR));
    ERR(CheckoutFloat(in_data, FG_SIZE_G, sizeG));
    ERR(CheckoutFloat(in_data, FG_SIZE_B, sizeB));
    ERR(CheckoutPopup(in_data, FG_BLEND_MODE, blend));
    ERR(CheckoutFloat(in_data, FG_SHADOWS, shadows));
    ERR(CheckoutFloat(in_data, FG_MIDTONES, midtones));
    ERR(CheckoutFloat(in_data, FG_HIGHLIGHTS, highlights));
    ERR(CheckoutFloat(in_data, FG_MIDPOINT, midpoint));
    ERR(CheckoutFloat(in_data, FG_MIX, mix));
    for (int i = 0; i < 9; ++i) {
        bal[i] = 1.0f;
        ERR(CheckoutFloat(in_data, FG_R_SH + i, bal[i]));
    }
    ERR(CheckoutFloat(in_data, FG_SEED, seed));
    ERR(CheckoutFloat(in_data, FG_SPEED, speed));
    ERR(CheckoutBool (in_data, FG_SMOOTH, smooth));
    if (err) return err;

    FGParams& P = pre->P;
    std::memset(&P, 0, sizeof(P));

    /* downsample factors: full-res pixels per output pixel */
    P.dsx = (in_data->downsample_x.num > 0) ? (float)in_data->downsample_x.den / (float)in_data->downsample_x.num : 1.0f;
    P.dsy = (in_data->downsample_y.num > 0) ? (float)in_data->downsample_y.den / (float)in_data->downsample_y.num : 1.0f;
    P.originX = (float)pre->rect.left * P.dsx;
    P.originY = (float)pre->rect.top  * P.dsy;

    /* softness -> lattice blur */
    softness = std::max(0.0f, softness);
    if (softness < 0.05f) {
        P.blurRadius = 0;
        P.blurSigma  = 1.0f;
        P.blurNorm   = 1.0f;
    } else {
        P.blurSigma  = softness;
        P.blurRadius = std::min((int)ceilf(3.0f * softness), FG_MAX_BLUR_RADIUS);
        P.blurNorm   = fg_blur_norm(P.blurRadius, P.blurSigma);
    }

    /* cell sizes in full-res pixels; never finer than one output pixel */
    {
        const float ar = sqrtf(std::max(aspect, 0.01f));
        const float base = std::max(size, 0.01f);
        const float chan[4] = { std::max(sizeR, 0.01f), std::max(sizeG, 0.01f), std::max(sizeB, 0.01f), 1.0f };
        for (int p = 0; p < 4; ++p) {
            pre->sizeX[p] = std::max(base * ar * chan[p], P.dsx);
            pre->sizeY[p] = std::max(base / ar * chan[p], P.dsy);
        }
    }
    BuildLayout(P, pre->rect, pre->sizeX, pre->sizeY);

    /* animation */
    {
        double frame = (in_data->time_step != 0) ? (double)in_data->current_time / (double)in_data->time_step : 0.0;
        double t     = frame * (double)std::max(speed, 0.0f);
        double fl    = floor(t);
        P.frameA     = (int)fl;
        P.frameB     = (int)fl + 1;
        P.frameBlend = smooth ? (float)(t - fl) : 0.0f;
        P.seed       = (int)seed;
    }

    /* colour */
    P.saturation = mono ? 0.0f : std::min(std::max(saturation / 100.0f, 0.0f), 1.0f);
    {
        float tr = tint.red / 255.0f, tg = tint.green / 255.0f, tb = tint.blue / 255.0f;
        float mx = std::max(std::max(tr, tg), std::max(tb, 1.0e-3f));
        float am = std::min(std::max(tintAmount / 100.0f, 0.0f), 1.0f);
        P.tintR = 1.0f + (tr / mx - 1.0f) * am;
        P.tintG = 1.0f + (tg / mx - 1.0f) * am;
        P.tintB = 1.0f + (tb / mx - 1.0f) * am;
    }
    P.ampR = FG_K_INTENSITY * intensity * intR;
    P.ampG = FG_K_INTENSITY * intensity * intG;
    P.ampB = FG_K_INTENSITY * intensity * intB;

    P.shR = shadows * bal[0]; P.miR = midtones * bal[1]; P.hiR = highlights * bal[2];
    P.shG = shadows * bal[3]; P.miG = midtones * bal[4]; P.hiG = highlights * bal[5];
    P.shB = shadows * bal[6]; P.miB = midtones * bal[7]; P.hiB = highlights * bal[8];
    P.midpoint = std::min(std::max(midpoint, 0.01f), 0.99f);

    P.blendMode   = std::min(std::max(blend - 1, 0), 4);
    P.mixOriginal = std::min(std::max(mix / 100.0f, 0.0f), 1.0f);
    P.viewMode    = (view == 2) ? FG_VIEW_GRAIN : FG_VIEW_RESULT;
    P.clampOutput = 0;

    /* GPU render only on the frameworks we implement */
    if (extra->input->what_gpu == PF_GPU_Framework_CUDA || extra->input->what_gpu == PF_GPU_Framework_DIRECTX) {
        extra->output->flags |= PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;
    }
    return err;
}

/* ------------------------------------------------------------------ */
/* CPU render                                                          */
/* ------------------------------------------------------------------ */
struct FGCPUContext
{
    FGParams P;
    float*   latt;
    float*   temp;
};

static PF_Err GenRow(void* refcon, A_long thread, A_long row, A_long n)
{
    FGCPUContext* c = reinterpret_cast<FGCPUContext*>(refcon);
    float* out = c->latt + (size_t)row * c->P.lattPitch;
    for (int i = 0; i < c->P.lattMaxW; ++i) out[i] = fg_gen_cell(i, (int)row, c->P);
    return PF_Err_NONE;
}

static PF_Err BlurHRow(void* refcon, A_long thread, A_long row, A_long n)
{
    FGCPUContext* c = reinterpret_cast<FGCPUContext*>(refcon);
    float* out = c->temp + (size_t)row * c->P.lattPitch;
    for (int i = 0; i < c->P.lattMaxW; ++i) out[i] = fg_blur_h_cell(c->latt, i, (int)row, c->P);
    return PF_Err_NONE;
}

static PF_Err BlurVRow(void* refcon, A_long thread, A_long row, A_long n)
{
    FGCPUContext* c = reinterpret_cast<FGCPUContext*>(refcon);
    float* out = c->latt + (size_t)row * c->P.lattPitch;
    for (int i = 0; i < c->P.lattMaxW; ++i) out[i] = fg_blur_v_cell(c->temp, i, (int)row, c->P);
    return PF_Err_NONE;
}

static PF_Err PixelFloat(void* refcon, A_long x, A_long y, PF_PixelFloat* inP, PF_PixelFloat* outP)
{
    FGCPUContext* c = reinterpret_cast<FGCPUContext*>(refcon);
    FGPixel s; s.r = inP->red; s.g = inP->green; s.b = inP->blue; s.a = inP->alpha;
    FGPixel r = fg_composite(c->latt, c->P, s, c->P.originX + (float)x * c->P.dsx, c->P.originY + (float)y * c->P.dsy);
    outP->red = r.r; outP->green = r.g; outP->blue = r.b; outP->alpha = r.a;
    return PF_Err_NONE;
}

static PF_Err Pixel16(void* refcon, A_long x, A_long y, PF_Pixel16* inP, PF_Pixel16* outP)
{
    FGCPUContext* c = reinterpret_cast<FGCPUContext*>(refcon);
    const float k = 1.0f / (float)PF_MAX_CHAN16;
    FGPixel s; s.r = inP->red * k; s.g = inP->green * k; s.b = inP->blue * k; s.a = inP->alpha * k;
    FGPixel r = fg_composite(c->latt, c->P, s, c->P.originX + (float)x * c->P.dsx, c->P.originY + (float)y * c->P.dsy);
    outP->red   = (A_u_short)std::min(std::max(r.r * (float)PF_MAX_CHAN16 + 0.5f, 0.0f), (float)PF_MAX_CHAN16);
    outP->green = (A_u_short)std::min(std::max(r.g * (float)PF_MAX_CHAN16 + 0.5f, 0.0f), (float)PF_MAX_CHAN16);
    outP->blue  = (A_u_short)std::min(std::max(r.b * (float)PF_MAX_CHAN16 + 0.5f, 0.0f), (float)PF_MAX_CHAN16);
    outP->alpha = (A_u_short)std::min(std::max(r.a * (float)PF_MAX_CHAN16 + 0.5f, 0.0f), (float)PF_MAX_CHAN16);
    return PF_Err_NONE;
}

static PF_Err Pixel8(void* refcon, A_long x, A_long y, PF_Pixel8* inP, PF_Pixel8* outP)
{
    FGCPUContext* c = reinterpret_cast<FGCPUContext*>(refcon);
    const float k = 1.0f / (float)PF_MAX_CHAN8;
    FGPixel s; s.r = inP->red * k; s.g = inP->green * k; s.b = inP->blue * k; s.a = inP->alpha * k;
    FGPixel r = fg_composite(c->latt, c->P, s, c->P.originX + (float)x * c->P.dsx, c->P.originY + (float)y * c->P.dsy);
    outP->red   = (A_u_char)std::min(std::max(r.r * (float)PF_MAX_CHAN8 + 0.5f, 0.0f), (float)PF_MAX_CHAN8);
    outP->green = (A_u_char)std::min(std::max(r.g * (float)PF_MAX_CHAN8 + 0.5f, 0.0f), (float)PF_MAX_CHAN8);
    outP->blue  = (A_u_char)std::min(std::max(r.b * (float)PF_MAX_CHAN8 + 0.5f, 0.0f), (float)PF_MAX_CHAN8);
    outP->alpha = (A_u_char)std::min(std::max(r.a * (float)PF_MAX_CHAN8 + 0.5f, 0.0f), (float)PF_MAX_CHAN8);
    return PF_Err_NONE;
}

static PF_Err RenderCPU(PF_InData* in_data, PF_OutData* out_data, PF_PixelFormat fmt,
                        PF_EffectWorld* input, PF_EffectWorld* output, FGPreRenderData* pre)
{
    PF_Err err = PF_Err_NONE;
    FGCPUContext ctx;
    ctx.P = pre->P;
    ctx.P.width       = std::min(input->width,  output->width);
    ctx.P.height      = std::min(input->height, output->height);
    ctx.P.lattPitch   = ctx.P.lattMaxW;
    ctx.P.clampOutput = (fmt == PF_PixelFormat_ARGB128) ? 0 : 1;

    const size_t cells = (size_t)ctx.P.lattPitch * (size_t)ctx.P.lattTotalH;
    std::vector<float> latt(cells), temp;
    if (ctx.P.blurRadius > 0) temp.resize(cells);
    ctx.latt = latt.data();
    ctx.temp = temp.empty() ? nullptr : temp.data();

    AEFX_SuiteScoper<PF_Iterate8Suite2> it8(in_data, kPFIterate8Suite, kPFIterate8SuiteVersion2, out_data);

    ERR(it8->iterate_generic(ctx.P.lattTotalH, &ctx, GenRow));
    if (!err && ctx.P.blurRadius > 0) {
        ERR(it8->iterate_generic(ctx.P.lattTotalH, &ctx, BlurHRow));
        ERR(it8->iterate_generic(ctx.P.lattTotalH, &ctx, BlurVRow));
    }
    if (err) return err;

    switch (fmt) {
        case PF_PixelFormat_ARGB128: {
            AEFX_SuiteScoper<PF_iterateFloatSuite2> it(in_data, kPFIterateFloatSuite, kPFIterateFloatSuiteVersion2, out_data);
            ERR(it->iterate(in_data, 0, output->height, input, NULL, &ctx, PixelFloat, output));
            break;
        }
        case PF_PixelFormat_ARGB64: {
            AEFX_SuiteScoper<PF_iterate16Suite2> it(in_data, kPFIterate16Suite, kPFIterate16SuiteVersion2, out_data);
            ERR(it->iterate(in_data, 0, output->height, input, NULL, &ctx, Pixel16, output));
            break;
        }
        case PF_PixelFormat_ARGB32: {
            ERR(it8->iterate(in_data, 0, output->height, input, NULL, &ctx, Pixel8, output));
            break;
        }
        default:
            err = PF_Err_BAD_CALLBACK_PARAM;
            break;
    }
    return err;
}

/* ------------------------------------------------------------------ */
/* GPU render                                                          */
/* ------------------------------------------------------------------ */
static PF_Err RenderGPU(PF_InData* in_data, PF_OutData* out_data, PF_PixelFormat fmt,
                        PF_EffectWorld* input, PF_EffectWorld* output,
                        PF_SmartRenderExtra* extra, FGPreRenderData* pre)
{
    PF_Err err = PF_Err_NONE;
    if (fmt != PF_PixelFormat_GPU_BGRA128) return PF_Err_UNRECOGNIZED_PARAM_TYPE;

    AEFX_SuiteScoper<PF_GPUDeviceSuite1> gpu(in_data, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, out_data);
    PF_GPUDeviceInfo info;
    AEFX_CLR_STRUCT(info);
    ERR(gpu->GetDeviceInfo(in_data->effect_ref, extra->input->device_index, &info));
    if (err) return err;

    FGParams P = pre->P;
    P.width       = std::min(input->width,  output->width);
    P.height      = std::min(input->height, output->height);
    P.srcPitch    = input->rowbytes / 16;
    P.dstPitch    = output->rowbytes / 16;
    P.clampOutput = 0;

    const A_long lattW4 = (P.lattMaxW + 3) / 4;      /* lattice stored in a BGRA128 world, 4 floats per pixel */
    PF_EffectWorld* latt = nullptr;
    PF_EffectWorld* temp = nullptr;
    ERR(gpu->CreateGPUWorld(in_data->effect_ref, extra->input->device_index, lattW4, P.lattTotalH,
                            input->pix_aspect_ratio, in_data->field, PF_PixelFormat_GPU_BGRA128, false, &latt));
    if (!err && P.blurRadius > 0) {
        ERR(gpu->CreateGPUWorld(in_data->effect_ref, extra->input->device_index, lattW4, P.lattTotalH,
                                input->pix_aspect_ratio, in_data->field, PF_PixelFormat_GPU_BGRA128, false, &temp));
        if (!err && temp->rowbytes != latt->rowbytes) err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }

    void *srcMem = nullptr, *dstMem = nullptr, *lattMem = nullptr, *tempMem = nullptr;
    if (!err) {
        P.lattPitch = latt->rowbytes / 4;
        ERR(gpu->GetGPUWorldData(in_data->effect_ref, input,  &srcMem));
        ERR(gpu->GetGPUWorldData(in_data->effect_ref, output, &dstMem));
        ERR(gpu->GetGPUWorldData(in_data->effect_ref, latt,   &lattMem));
        if (temp) ERR(gpu->GetGPUWorldData(in_data->effect_ref, temp, &tempMem));
    }

    if (!err) {
        if (extra->input->what_gpu == PF_GPU_Framework_CUDA) {
            int e = FG_CudaRender(P, srcMem, dstMem, lattMem, tempMem, info.command_queuePV);
            if (e != 0) err = PF_Err_INTERNAL_STRUCT_DAMAGED;
        } else if (extra->input->what_gpu == PF_GPU_Framework_DIRECTX) {
            PF_Handle h = (PF_Handle)extra->input->gpu_data;
            FGGPUData* gd = h ? reinterpret_cast<FGGPUData*>(*h) : nullptr;
            if (!gd || !gd->dx) {
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            } else {
                const uint64_t lattBytes = (uint64_t)latt->rowbytes * (uint64_t)latt->height;
                bool ok = gd->dx->Render(P,
                    (ID3D12Resource*)srcMem,  (uint64_t)input->rowbytes  * (uint64_t)input->height,
                    (ID3D12Resource*)dstMem,  (uint64_t)output->rowbytes * (uint64_t)output->height,
                    (ID3D12Resource*)lattMem, lattBytes,
                    (ID3D12Resource*)tempMem, temp ? lattBytes : 0);
                if (!ok) err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            }
        } else {
            err = PF_Err_UNRECOGNIZED_PARAM_TYPE;
        }
    }

    if (temp) gpu->DisposeGPUWorld(in_data->effect_ref, temp);
    if (latt) gpu->DisposeGPUWorld(in_data->effect_ref, latt);
    return err;
}

/* ------------------------------------------------------------------ */
/* SmartRender dispatch                                                */
/* ------------------------------------------------------------------ */
static PF_Err SmartRender(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra, bool isGPU)
{
    PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;
    FGPreRenderData* pre = reinterpret_cast<FGPreRenderData*>(extra->input->pre_render_data);
    if (!pre) return PF_Err_INTERNAL_STRUCT_DAMAGED;
    if (RectEmpty(pre->rect)) return PF_Err_NONE;

    PF_EffectWorld* input  = nullptr;
    PF_EffectWorld* output = nullptr;
    ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, FG_INPUT, &input));
    ERR(extra->cb->checkout_output(in_data->effect_ref, &output));

    if (!err && input && output) {
        AEFX_SuiteScoper<PF_WorldSuite2> ws(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);
        PF_PixelFormat fmt = PF_PixelFormat_INVALID;
        ERR(ws->PF_GetPixelFormat(input, &fmt));
        if (!err) {
            if (isGPU) ERR(RenderGPU(in_data, out_data, fmt, input, output, extra, pre));
            else       ERR(RenderCPU(in_data, out_data, fmt, input, output, pre));
        }
    }
    ERR2(extra->cb->checkin_layer_pixels(in_data->effect_ref, FG_INPUT));
    return err;
}

/* legacy (non-smart) render: only reached by hosts without SmartFX; pass through */
static PF_Err Render(PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output)
{
    AEFX_SuiteScoper<PF_WorldTransformSuite1> wt(in_data, kPFWorldTransformSuite, kPFWorldTransformSuiteVersion1, out_data);
    return wt->copy(in_data->effect_ref, &params[FG_INPUT]->u.ld, output, NULL, NULL);
}

/* ------------------------------------------------------------------ */
/* entry points                                                        */
/* ------------------------------------------------------------------ */
extern "C" DllExport
PF_Err PluginDataEntryFunction2(
    PF_PluginDataPtr   inPtr,
    PF_PluginDataCB2   inPluginDataCallBackPtr,
    SPBasicSuite*      inSPBasicSuitePtr,
    const char*        inHostName,
    const char*        inHostVersion)
{
    /* Same as PF_REGISTER_EFFECT_EXT2, but with the spec version pinned to FG_SPEC_VERSION/SUBVERS
       instead of the SDK's PF_AE_PLUG_IN_VERSION/SUBVERS (13.29), which this host rejects. */
    if (!inPluginDataCallBackPtr) return PF_Err_INVALID_CALLBACK;
    A_Err result = (*inPluginDataCallBackPtr)(
        inPtr,
        reinterpret_cast<const A_u_char*>(FG_NAME),
        reinterpret_cast<const A_u_char*>(FG_MATCH_NAME),
        reinterpret_cast<const A_u_char*>(FG_CATEGORY),
        reinterpret_cast<const A_u_char*>("EffectMain"),
        'eFKT',
        FG_SPEC_VERSION,
        FG_SPEC_SUBVERS,
        AE_RESERVED_INFO,
        reinterpret_cast<const A_u_char*>(FG_SUPPORT_URL));
    return (result == A_Err_NONE) ? PF_Err_NONE : PF_Err_INVALID_CALLBACK;
}

PF_Err EffectMain(PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* params[], PF_LayerDef* output, void* extra)
{
    PF_Err err = PF_Err_NONE;
    try {
        switch (cmd) {
            case PF_Cmd_ABOUT:              err = About(in_data, out_data, params, output); break;
            case PF_Cmd_GLOBAL_SETUP:       err = GlobalSetup(in_data, out_data, params, output); break;
            case PF_Cmd_PARAMS_SETUP:       err = ParamsSetup(in_data, out_data, params, output); break;
            case PF_Cmd_GPU_DEVICE_SETUP:   err = GPUDeviceSetup(in_data, out_data, (PF_GPUDeviceSetupExtra*)extra); break;
            case PF_Cmd_GPU_DEVICE_SETDOWN: err = GPUDeviceSetdown(in_data, out_data, (PF_GPUDeviceSetdownExtra*)extra); break;
            case PF_Cmd_RENDER:             err = Render(in_data, out_data, params, output); break;
            case PF_Cmd_SMART_PRE_RENDER:   err = PreRender(in_data, out_data, (PF_PreRenderExtra*)extra); break;
            case PF_Cmd_SMART_RENDER:       err = SmartRender(in_data, out_data, (PF_SmartRenderExtra*)extra, false); break;
            case PF_Cmd_SMART_RENDER_GPU:   err = SmartRender(in_data, out_data, (PF_SmartRenderExtra*)extra, true); break;
            default: break;
        }
    } catch (PF_Err& thrown) {
        err = thrown;
    } catch (...) {
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
    return err;
}
