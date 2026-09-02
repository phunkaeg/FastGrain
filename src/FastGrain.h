#pragma once
/*
    FastGrain.h - "Fast Grain": GPU (CUDA / DirectX 12) + multithreaded CPU film grain
    for After Effects, 8/16/32-bit aware, SmartFX, Multi-Frame Rendering safe.
*/

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "AEConfig.h"
#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_EffectCBSuites.h"
#include "AE_EffectGPUSuites.h"
#include "AE_EffectPixelFormat.h"
#include "AE_Macros.h"
#include "AEFX_SuiteHelper.h"
#include "AEFX_SuiteHandlerTemplate.h"
#include "Param_Utils.h"

#ifdef AE_OS_WIN
    #include <Windows.h>
#endif

#include "FastGrain_Shared.h"

#define FG_NAME         "Fast Grain"
#define FG_MATCH_NAME   "PHUNK FastGrain"
#define FG_CATEGORY     "Noise & Grain"
#define FG_DESCRIPTION  "GPU accelerated film grain with float precision.\rCUDA, DirectX 12 and multithreaded CPU."
#define FG_SUPPORT_URL  "https://github.com/phunkaeg"

#define FG_MAJOR_VERSION 1
#define FG_MINOR_VERSION 0
#define FG_BUG_VERSION   0
#define FG_STAGE_VERSION PF_Stage_DEVELOP
#define FG_BUILD_VERSION 1

/* Effect API spec version advertised to the host (PiPL + PluginDataEntryFunction2).
   Pinned to 13.28, the value shipping third-party effects (e.g. Deep Glow 2) use;
   the 25.6 SDK headers default to 13.29. Verified loading in AE 2025 with 13.28.
   Keep in sync with FastGrainPiPL.r. */
#define FG_SPEC_VERSION  13
#define FG_SPEC_SUBVERS  28

/* grain std-dev (in [0,1] pixel units) at Intensity = 1.0, response = 1 */
#define FG_K_INTENSITY   0.05f

/* Flags declared here must match the PiPL (FastGrainPiPL.r) exactly. */
#define FG_OUT_FLAGS  (PF_OutFlag_NON_PARAM_VARY | PF_OutFlag_PIX_INDEPENDENT | PF_OutFlag_DEEP_COLOR_AWARE)
#define FG_OUT_FLAGS2 (PF_OutFlag2_PARAM_GROUP_START_COLLAPSED_FLAG | \
                       PF_OutFlag2_SUPPORTS_SMART_RENDER | \
                       PF_OutFlag2_FLOAT_COLOR_AWARE | \
                       PF_OutFlag2_SUPPORTS_GPU_RENDER_F32 | \
                       PF_OutFlag2_SUPPORTS_THREADED_RENDERING | \
                       PF_OutFlag2_SUPPORTS_DIRECTX_RENDERING)

/* parameter indices (order = UI order) */
enum {
    FG_INPUT = 0,
    FG_VIEW,
    FG_INTENSITY,
    FG_SIZE,
    FG_SOFTNESS,
    FG_ASPECT,

    FG_COLOR_TOPIC,
    FG_SATURATION,
    FG_MONO,
    FG_TINT_COLOR,
    FG_TINT_AMOUNT,
    FG_COLOR_END,

    FG_CHANINT_TOPIC,
    FG_INT_R,
    FG_INT_G,
    FG_INT_B,
    FG_CHANINT_END,

    FG_CHANSIZE_TOPIC,
    FG_SIZE_R,
    FG_SIZE_G,
    FG_SIZE_B,
    FG_CHANSIZE_END,

    FG_APP_TOPIC,
    FG_BLEND_MODE,
    FG_SHADOWS,
    FG_MIDTONES,
    FG_HIGHLIGHTS,
    FG_MIDPOINT,
    FG_MIX,
    FG_APP_END,

    FG_BAL_TOPIC,
    FG_R_SH, FG_R_MI, FG_R_HI,
    FG_G_SH, FG_G_MI, FG_G_HI,
    FG_B_SH, FG_B_MI, FG_B_HI,
    FG_BAL_END,

    FG_ANIM_TOPIC,
    FG_SEED,
    FG_SPEED,
    FG_SMOOTH,
    FG_ANIM_END,

    FG_NUM_PARAMS
};

/* stable parameter IDs for project versioning: never renumber, only append */
#define FG_ID(idx) (1000 + (idx))

extern "C" {
    DllExport PF_Err EffectMain(
        PF_Cmd        cmd,
        PF_InData*    in_data,
        PF_OutData*   out_data,
        PF_ParamDef*  params[],
        PF_LayerDef*  output,
        void*         extra);
}
