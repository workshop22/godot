#include "register_types.h"

#include "core/object/class_db.h"
#include "physx_gi_effect.h"
#include "pxtgi_3d.h"
#include "physx_lens_dirt_3d.h"
#include "physx_lens_dirt_effect.h"
#include "physx_lens_flare_3d.h"
#include "physx_lens_flare_effect.h"
#include "physx_lens_flare_settings.h"
#include "physx_light_shafts_3d.h"
#include "physx_light_shafts_effect.h"

void initialize_lightriders_pxtgi_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		GDREGISTER_CLASS(PXTGI3D);
		GDREGISTER_INTERNAL_CLASS(PhysXGIEffect);
		GDREGISTER_CLASS(PhysXLensDirt3D);
		GDREGISTER_CLASS(PhysXLensFlare3D);
		GDREGISTER_CLASS(PhysXLensFlareSettings);
		GDREGISTER_CLASS(PhysXLightShafts3D);
		GDREGISTER_INTERNAL_CLASS(PhysXLensDirtEffect);
		GDREGISTER_INTERNAL_CLASS(PhysXLensFlareEffect);
		GDREGISTER_INTERNAL_CLASS(PhysXLightShaftsEffect);
	}
}

void uninitialize_lightriders_pxtgi_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		PhysXLensDirtEffect::free_shared();
		PhysXLensFlareEffect::free_shared();
		PhysXLightShaftsEffect::free_shared();
		PhysXGIEffect::free_shared();
		PXTGI3D::free_shared();
	}
}
