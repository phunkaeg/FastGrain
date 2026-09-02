#include "AEConfig.h"
#include "AE_EffectVers.h"

#ifndef AE_OS_WIN
	#include "AE_General.r"
#endif

/*
	Flag values must match FG_OUT_FLAGS / FG_OUT_FLAGS2 in FastGrain.h:
	  out_flags  = NON_PARAM_VARY (1<<2) | PIX_INDEPENDENT (1<<10) | DEEP_COLOR_AWARE (1<<25)        = 0x02000404
	  out_flags2 = PARAM_GROUP_START_COLLAPSED (1<<3) | SUPPORTS_SMART_RENDER (1<<10)
	             | FLOAT_COLOR_AWARE (1<<12) | SUPPORTS_GPU_RENDER_F32 (1<<25)
	             | SUPPORTS_THREADED_RENDERING (1<<27) | SUPPORTS_DIRECTX_RENDERING (1<<29)          = 0x2A001408
*/
resource 'PiPL' (16000) {
	{
		Kind {
			AEEffect
		},
		Name {
			"Fast Grain"
		},
		Category {
			"Noise & Grain"
		},
#ifdef AE_OS_WIN
	#if defined(AE_PROC_INTELx64)
		CodeWin64X86 {"EffectMain"},
	#elif defined(AE_PROC_ARM64)
		CodeWinARM64 {"EffectMain"},
	#endif
#elif defined(AE_OS_MAC)
		CodeMacIntel64 {"EffectMain"},
		CodeMacARM64 {"EffectMain"},
#endif
		AE_PiPL_Version {
			2,
			0
		},
		AE_Effect_Spec_Version {
			13,		/* FG_SPEC_VERSION - keep in sync with FastGrain.h */
			28		/* FG_SPEC_SUBVERS - pinned to what shipping third-party effects use (SDK 25.6 default is 29) */
		},
		AE_Effect_Version {
			524289	/* 1.0.0 develop build 1 */
		},
		AE_Effect_Info_Flags {
			0
		},
		AE_Effect_Global_OutFlags {
			0x02000404
		},
		AE_Effect_Global_OutFlags_2 {
			0x2A001408
		},
		AE_Effect_Match_Name {
			"PHUNK FastGrain"
		},
		AE_Reserved_Info {
			0
		},
		AE_Effect_Support_URL {
			"https://github.com/phunkaeg"
		}
	}
};
