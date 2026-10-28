#include "register_types.h"

#include "core/object/class_db.h"
#include "physx_gi_effect.h"
#include "pxtgi_3d.h"

void initialize_lightriders_pxtgi_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		GDREGISTER_CLASS(PXTGI3D);
		GDREGISTER_INTERNAL_CLASS(PhysXGIEffect);
	}
}

void uninitialize_lightriders_pxtgi_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		PhysXGIEffect::free_shared();
		PXTGI3D::free_shared();
	}
}
