/**************************************************************************/
/*  pxtgi_3d.cpp                                                          */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "pxtgi_3d.h"

#include "pxtgi_render_thread.h"
#include "physx_gi_effect.h"

#include "core/config/engine.h"
#include "core/config/project_settings.h"
#include "core/math/convex_hull.h"
#include "core/object/callable_mp.h"
#include "core/os/os.h"
#include "core/templates/hashfuncs.h"
#include "scene/3d/camera_3d.h"
#include "scene/3d/light_3d.h"
#include "scene/3d/mesh_instance_3d.h"
#include "scene/3d/physics/animatable_body_3d.h"
#include "scene/3d/physics/area_3d.h"
#include "scene/3d/physics/collision_object_3d.h"
#include "scene/3d/physics/collision_shape_3d.h"
#include "scene/3d/reflection_probe.h"
#include "scene/main/scene_tree.h"
#include "scene/main/viewport.h"
#include "scene/main/window.h"
#include "scene/resources/3d/box_shape_3d.h"
#include "scene/resources/3d/capsule_shape_3d.h"
#include "scene/resources/3d/concave_polygon_shape_3d.h"
#include "scene/resources/3d/convex_polygon_shape_3d.h"
#include "scene/resources/3d/cylinder_shape_3d.h"
#include "scene/resources/3d/height_map_shape_3d.h"
#include "scene/resources/3d/sphere_shape_3d.h"
#include "scene/resources/3d/world_3d.h"
#include "scene/resources/material.h"

#ifdef TOOLS_ENABLED
#include "main/main.h"
#endif
#include "servers/rendering/renderer_rd/environment/gi.h"
#include "servers/rendering/renderer_rd/renderer_scene_render_rd.h"
#include "servers/rendering/rendering_device_binds.h"
#include "servers/rendering/rendering_server.h"

#include <algorithm>

int PXTGI3D::users = 0;

namespace {

RID hook_shader;

// The binding layout of Godot's gi.glsl, declared exactly (every binding used, so none is stripped), so the uniform
// sets process_gi builds for its own shader bind to this one too. @VRS@ is replaced with the VRS image binding when the
// GPU supports VRS: Godot then builds its uniform set against the VRS variant.
const char *HOOK_SHADER = R"(
#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(constant_id = 0) const bool sc_half_res = false;
layout(constant_id = 1) const bool sc_use_full_projection_matrix = false;
layout(constant_id = 2) const bool sc_use_vrs = false;

layout(set = 0, binding = 1) uniform texture3D sdf_cascades[8];
layout(set = 0, binding = 2) uniform texture3D light_cascades[8];
layout(set = 0, binding = 3) uniform texture3D aniso0_cascades[8];
layout(set = 0, binding = 4) uniform texture3D aniso1_cascades[8];
layout(set = 0, binding = 5) uniform texture3D occlusion_texture;
layout(set = 0, binding = 6) uniform sampler linear_sampler;
layout(set = 0, binding = 7) uniform sampler linear_sampler_with_mipmaps;
layout(rgba16f, set = 0, binding = 9) uniform restrict writeonly image2D ambient_buffer;
layout(rgba16f, set = 0, binding = 10) uniform restrict writeonly image2D reflection_buffer;
layout(set = 0, binding = 11) uniform texture2DArray lightprobe_texture;
layout(set = 0, binding = 12) uniform texture2D depth_buffer;
layout(set = 0, binding = 13) uniform texture2D normal_roughness_buffer;
layout(set = 0, binding = 14) uniform utexture2D voxel_gi_buffer;
struct ProbeCascadeData {
	vec3 position;
	float to_probe;
	ivec3 probe_world_offset;
	float to_cell;
	vec3 pad;
	float exposure_normalization;
};
layout(set = 0, binding = 15, std140) uniform SDFGI {
	vec3 grid_size;
	uint max_cascades;
	bool use_occlusion;
	int probe_axis_size;
	float probe_to_uvw;
	float normal_bias;
	vec3 lightprobe_tex_pixel_size;
	float energy;
	vec3 lightprobe_uv_offset;
	float y_mult;
	vec3 occlusion_clamp;
	uint pad3;
	vec3 occlusion_renormalize;
	uint pad4;
	vec3 cascade_probe_size;
	uint pad5;
	ProbeCascadeData cascades[8];
}
sdfgi;
struct VoxelGIData {
	mat4 xform;
	vec3 bounds;
	float dynamic_range;
	float bias;
	float normal_bias;
	bool blend_ambient;
	uint mipmaps;
	vec3 pad;
	float exposure_normalization;
};
layout(set = 0, binding = 16, std140) uniform VoxelGIs {
	VoxelGIData data[8];
}
voxel_gi_instances;
layout(set = 0, binding = 17) uniform texture3D voxel_gi_textures[8];
layout(set = 0, binding = 18, std140) uniform SceneData {
	mat4x4 inv_projection[2];
	mat4x4 cam_transform;
	vec4 eye_offset[2];
	ivec2 screen_size;
	float pad1;
	float pad2;
}
scene_data;
@VRS@

layout(push_constant, std430) uniform Params {
	uint max_voxel_gi_instances;
	bool high_quality_vct;
	bool orthogonal;
	uint view_index;
	vec4 proj_info;
	float z_near;
	float z_far;
	float pad2;
	float pad3;
}
params;

vec3 view_position(ivec2 screen_pos) {
	float depth = texelFetch(sampler2D(depth_buffer, linear_sampler), screen_pos, 0).r;
	if (sc_use_full_projection_matrix) {
		vec4 pos = vec4((2.0 * vec2(screen_pos) / vec2(scene_data.screen_size)) - 1.0, depth, 1.0);
		pos = scene_data.inv_projection[params.view_index] * pos;
		return pos.xyz / pos.w;
	}
	vec3 pos;
	pos.z = depth * 2.0 - 1.0;
	if (params.orthogonal) {
		pos.z = -(pos.z * (params.z_far - params.z_near) - (params.z_far + params.z_near)) / 2.0;
	} else {
		pos.z = 2.0 * params.z_near * params.z_far / (params.z_far + params.z_near + pos.z * (params.z_far - params.z_near));
	}
	pos.z = -pos.z;
	pos.xy = vec2(screen_pos) * params.proj_info.xy + params.proj_info.zw;
	if (!params.orthogonal) {
		pos.xy *= pos.z;
	}
	pos.y = -pos.y;
	return pos;
}

// Never true (Godot's view index is 0 or 1): keeps every binding in the compiled shader so the layout matches.
void keep_bindings(ivec2 pos) {
	if (params.view_index == 0xFFFFFFFFu) {
		vec4 k = texture(sampler3D(sdf_cascades[0], linear_sampler), vec3(0.0)) + texture(sampler3D(light_cascades[0], linear_sampler), vec3(0.0)) +
				texture(sampler3D(aniso0_cascades[0], linear_sampler), vec3(0.0)) + texture(sampler3D(aniso1_cascades[0], linear_sampler), vec3(0.0)) +
				texture(sampler3D(occlusion_texture, linear_sampler_with_mipmaps), vec3(0.0)) + texture(sampler2DArray(lightprobe_texture, linear_sampler), vec3(0.0)) +
				texture(sampler2D(normal_roughness_buffer, linear_sampler), vec2(0.0)) + vec4(texelFetch(usampler2D(voxel_gi_buffer, linear_sampler), ivec2(0), 0)) +
				vec4(sdfgi.grid_size, sdfgi.energy) + voxel_gi_instances.data[0].xform[0] + texture(sampler3D(voxel_gi_textures[0], linear_sampler), vec3(0.0)) + scene_data.eye_offset[0] @VRS_USE@;
		imageStore(reflection_buffer, pos, k);
	}
}

void main() {
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
	if (sc_half_res) {
		pos <<= 1;
	}
	if (any(greaterThanEqual(pos, scene_data.screen_size))) {
		return;
	}
	keep_bindings(pos);
	// Nothing else: the module's PhysXGIEffect already wrote this frame's GI into these buffers, before the opaque
	// pass; this step only has to leave them alone.
}
)";

Vector<RD::PipelineSpecializationConstant> specialization(int p_variation) {
	Vector<RD::PipelineSpecializationConstant> constants;
	for (int i = 0; i < 3; i++) {
		RD::PipelineSpecializationConstant sc;
		sc.type = RD::PIPELINE_SPECIALIZATION_CONSTANT_TYPE_BOOL;
		sc.constant_id = i;
		sc.bool_value = (p_variation & (1 << i)) != 0;
		constants.push_back(sc);
	}
	return constants;
}

} // namespace

// On the render thread.
void PXTGI3D::_install() {
	RendererSceneRenderRD *scene_render = RendererSceneRenderRD::get_singleton();
	RenderingDevice *rd = RenderingDevice::get_singleton();
	ERR_FAIL_COND(scene_render == nullptr || rd == nullptr);
	RendererRD::GI *gi = scene_render->get_gi();
	if (hook_shader.is_null()) {
		const bool vrs = scene_render->is_vrs_supported();
		String source = String(HOOK_SHADER)
								.replace("@VRS@", vrs ? "layout(r8ui, set = 0, binding = 19) uniform restrict readonly uimage2D vrs_buffer;" : "")
								.replace("@VRS_USE@", vrs ? "+ vec4(imageLoad(vrs_buffer, ivec2(0)))" : "");
		Ref<RDShaderFile> file;
		file.instantiate();
		ERR_FAIL_COND_MSG(file->parse_versions_from_text(source) != OK, "PXTGI3D: the hook shader failed to parse.");
		Ref<RDShaderSPIRV> spirv = file->get_spirv();
		ERR_FAIL_COND(spirv.is_null());
		const String err = spirv->get_stage_compile_error(RD::SHADER_STAGE_COMPUTE);
		ERR_FAIL_COND_MSG(!err.is_empty(), "PXTGI3D: " + err);
		hook_shader = rd->shader_create_from_spirv(file->get_spirv_stages());
		ERR_FAIL_COND(hook_shader.is_null());
	}
	for (int v = 0; v < RendererRD::GI::SHADER_SPECIALIZATION_VARIATIONS; v++) {
		for (int mode : { RendererRD::GI::MODE_SDFGI, RendererRD::GI::MODE_COMBINED, RendererRD::GI::MODE_COMBINED_WITHOUT_SAMPLER }) {
			gi->pipelines[v][mode].create_compute_pipeline(hook_shader, specialization(v));
		}
	}
	print_verbose("PXTGI3D: GI step hooked.");
}

// On the render thread: Godot's own pipelines back, exactly as GI::init builds them.
void PXTGI3D::_uninstall() {
	RendererSceneRenderRD *scene_render = RendererSceneRenderRD::get_singleton();
	if (scene_render == nullptr) {
		return;
	}
	RendererRD::GI *gi = scene_render->get_gi();
	const int variant_base = scene_render->is_vrs_supported() ? RendererRD::GI::MODE_MAX : 0;
	for (int v = 0; v < RendererRD::GI::SHADER_SPECIALIZATION_VARIATIONS; v++) {
		for (int mode : { RendererRD::GI::MODE_SDFGI, RendererRD::GI::MODE_COMBINED, RendererRD::GI::MODE_COMBINED_WITHOUT_SAMPLER }) {
			gi->pipelines[v][mode].create_compute_pipeline(gi->shader.version_get_shader(gi->shader_version, variant_base + mode), specialization(v));
		}
	}
	print_verbose("PXTGI3D: GI step restored.");
}

// At shutdown: the do-nothing shader (Godot's own GI step is back by then).
void PXTGI3D::free_shared() {
	pxtgi_run_on_render_thread([]() {
		RenderingDevice *rd = RenderingDevice::get_singleton();
		if (hook_shader.is_valid() && rd) {
			rd->free_rid(hook_shader);
		}
		hook_shader = RID();
	});
}

void PXTGI3D::_attach() {
	if (attached_world.is_valid() || !is_inside_world()) {
		return;
	}
	World3D *world = get_world_3d().ptr();
	if (world == nullptr) {
		return;
	}
	PhysXGIEffect::attach_world(world);
	attached_world = world->get_instance_id();
}

// In the world and visible: deliver the GI (the first such node hooks Godot's GI step).
void PXTGI3D::_activate() {
	if (active) {
		return;
	}
	active = true;
	_attach();
	if (users++ == 0) {
		RenderingServer::get_singleton()->environment_set_sdfgi_ray_count(RSE::ENV_SDFGI_RAY_COUNT_4);
		RenderingServer::get_singleton()->environment_set_sdfgi_frames_to_update_light(RSE::ENV_SDFGI_UPDATE_LIGHT_IN_16_FRAMES);
		RenderingServer::get_singleton()->call_on_render_thread(callable_mp_static(&PXTGI3D::_install));
	}
	// What's tracked is found by one walk on the first update, then kept up to date from the tree's signals.
	tracked_root = ObjectID();
	get_tree()->connect("node_added", callable_mp(this, &PXTGI3D::_node_added));
	get_tree()->connect("node_removed", callable_mp(this, &PXTGI3D::_node_removed));
	set_process_internal(true);
	set_physics_process_internal(true);
}

// Out of the world or hidden: Godot's own GI back (the last such node restores its GI step).
void PXTGI3D::_deactivate() {
	if (!active) {
		return;
	}
	active = false;
	set_process_internal(false);
	set_physics_process_internal(false);
	moved_poses.clear();
	get_tree()->disconnect("node_added", callable_mp(this, &PXTGI3D::_node_added));
	get_tree()->disconnect("node_removed", callable_mp(this, &PXTGI3D::_node_removed));
	tracked.clear();
	looks.clear();
	pending_meshes.clear();
	tracked_root = ObjectID();
	_detach();
	if (--users == 0) {
		RenderingServer::get_singleton()->call_on_render_thread(callable_mp_static(&PXTGI3D::_uninstall));
		// SDFGI's probe ray count and light update rate are global: the project's own back (as Godot reads them at
		// start), or a scene using SDFGI after PXTGI would keep the cheapest ones set above.
		RenderingServer::get_singleton()->environment_set_sdfgi_ray_count(RSE::EnvironmentSDFGIRayCount(CLAMP(int(GLOBAL_GET("rendering/global_illumination/sdfgi/probe_ray_count")), 0, int(RSE::ENV_SDFGI_RAY_COUNT_MAX) - 1)));
		RenderingServer::get_singleton()->environment_set_sdfgi_frames_to_update_light(RSE::EnvironmentSDFGIFramesToUpdateLight(CLAMP(int(GLOBAL_GET("rendering/global_illumination/sdfgi/frames_to_update_lights")), 0, int(RSE::ENV_SDFGI_UPDATE_LIGHT_MAX) - 1)));
	}
}

void PXTGI3D::_detach() {
	PhysXGIEffect::remove_settings(get_instance_id());
	// The environment's own SDFGI settings back (setting a property re-sends them all).
	if (environment.is_valid()) {
		environment->set_sdfgi_enabled(environment->is_sdfgi_enabled());
		environment.unref();
	}
	if (attached_world.is_valid()) {
		PhysXGIEffect::detach_world(attached_world);
		attached_world = ObjectID();
	}
}

namespace {

// Textures' average colours, worked out once each (forgotten whenever the GI walks the scene again, so the map
// doesn't keep growing with textures long gone).
HashMap<ObjectID, Color> texture_averages;

// A texture's average colour (linear): a masked emission's overall glow.
Color texture_average(const Ref<Texture2D> &p_texture) {
	HashMap<ObjectID, Color> &known = texture_averages;
	if (const Color *c = known.getptr(p_texture->get_instance_id())) {
		return *c;
	}
	Color average(1, 1, 1);
	Ref<Image> image = p_texture->get_image();
	if (image.is_valid() && !image->is_empty()) {
		image = image->duplicate();
		if (image->is_compressed()) {
			image->decompress();
		}
		const int w = image->get_width();
		const int h = image->get_height();
		Color sum(0, 0, 0, 0);
		int n = 0;
		for (int y = 0; y < 32; y++) {
			for (int x = 0; x < 32; x++) {
				sum += image->get_pixel((x * 2 + 1) * w / 64, (y * 2 + 1) * h / 64).srgb_to_linear();
				n++;
			}
		}
		average = sum / float(n);
	}
	known.insert(p_texture->get_instance_id(), average);
	return average;
}

// The metadata names, made once: a StringName from a string looks the name up in the global table each time, and the
// looks are read again a slice every frame.
const StringName &meta_albedo() {
	static const StringName name = "pxtgi_albedo";
	return name;
}

const StringName &meta_emission() {
	static const StringName name = "pxtgi_emission";
	return name;
}

const StringName &meta_ignore() {
	static const StringName name = "pxtgi_ignore";
	return name;
}

const StringName &meta_sample_emission() {
	static const StringName name = "pxtgi_sample_emission";
	return name;
}

const StringName &meta_manual_updates() {
	static const StringName name = "pxtgi_manual_updates";
	return name;
}

// The light a material gives off (linear; emission times its energy, in the units Godot adds it to the frame; an
// emission texture by its average). A material Godot can't read the glow of (a ShaderMaterial) can say it with a
// "pxtgi_emission" Color in its metadata.
Color material_glow(const Ref<Material> &p_material) {
	if (p_material.is_null()) {
		return Color(0, 0, 0);
	}
	if (p_material->has_meta(meta_emission())) {
		Color glow = Color(p_material->get_meta(meta_emission())).srgb_to_linear();
		glow.a = 1.0f;
		return glow;
	}
	Ref<BaseMaterial3D> material = p_material;
	if (material.is_null() || !material->get_feature(BaseMaterial3D::FEATURE_EMISSION)) {
		return Color(0, 0, 0);
	}
	const Color base = material->get_emission().srgb_to_linear();
	Ref<Texture2D> texture = material->get_texture(BaseMaterial3D::TEXTURE_EMISSION);
	Color glow = base;
	if (texture.is_valid()) {
		const Color average = texture_average(texture);
		glow = material->get_emission_operator() == BaseMaterial3D::EMISSION_OP_ADD ? base + average : base * average;
	}
	glow.a = 1.0f;
	// With physical light units Godot also scales it by the material's intensity in nits.
	const bool physical = GLOBAL_GET_CACHED(bool, "rendering/lights_and_shadows/use_physical_light_units");
	return glow * material->get_emission_energy_multiplier() * (physical ? material->get_emission_intensity() : 1.0f);
}

// A light's colour times its strength, in the trace's units: Godot's light energy over pi (Godot scales a light by pi
// without physical light units). With them, Godot scales it by its intensity instead: lux for the sun, lumens over 4 pi
// for an omni light and over pi for a spot.
Color light_strength(const Light3D *p_light) {
	float strength = p_light->get_param(Light3D::PARAM_ENERGY);
	if (GLOBAL_GET_CACHED(bool, "rendering/lights_and_shadows/use_physical_light_units")) {
		const float intensity = p_light->get_param(Light3D::PARAM_INTENSITY);
		if (Object::cast_to<DirectionalLight3D>(p_light)) {
			strength *= intensity / Math::PI;
		} else if (Object::cast_to<OmniLight3D>(p_light)) {
			strength *= intensity / (4.0f * Math::PI * Math::PI);
		} else {
			strength *= intensity / (Math::PI * Math::PI);
		}
	}
	return p_light->get_color().srgb_to_linear() * strength;
}

// A mesh's albedo (its first surface Godot can read the colour of, or that says it with a "pxtgi_albedo" Color in its
// metadata) and the light it gives off (its brightest surface: a lamp's housing and bulb are often one mesh). A
// "pxtgi_emission" Color in the MeshInstance3D's own metadata stands for the whole mesh.
void mesh_look(MeshInstance3D *p_mesh, Color &r_albedo, Color &r_emission) {
	r_albedo = Color(0.5, 0.5, 0.5);
	r_emission = Color(0, 0, 0);
	bool have_albedo = false;
	const int surfaces = p_mesh->get_mesh().is_valid() ? p_mesh->get_mesh()->get_surface_count() : 0;
	for (int i = 0; i < MAX(surfaces, 1); i++) {
		const Ref<Material> material = p_mesh->get_active_material(i);
		const Ref<BaseMaterial3D> base = material;
		if (!have_albedo && material.is_valid() && material->has_meta(meta_albedo())) {
			r_albedo = Color(material->get_meta(meta_albedo())).srgb_to_linear();
			have_albedo = true;
		} else if (!have_albedo && base.is_valid()) {
			r_albedo = base->get_albedo().srgb_to_linear();
			have_albedo = true;
		}
		const Color glow = material_glow(material);
		if (glow.get_luminance() > r_emission.get_luminance()) {
			r_emission = glow;
		}
	}
	if (p_mesh->has_meta(meta_emission())) {
		r_emission = Color(p_mesh->get_meta(meta_emission())).srgb_to_linear();
		r_emission.a = 1.0f;
	}
}

// The first MeshInstance3D among p_node's children.
MeshInstance3D *first_mesh_child(const Node *p_node) {
	for (int i = 0; i < p_node->get_child_count(); i++) {
		if (MeshInstance3D *mesh = Object::cast_to<MeshInstance3D>(p_node->get_child(i))) {
			return mesh;
		}
	}
	return nullptr;
}

// The look of the mesh a body belongs to: the first mesh under it, else the mesh it sits under (a model with its
// collision body as a child), else the first mesh beside it (in a group of up to 8), else a neutral grey. A
// "pxtgi_albedo" or "pxtgi_emission" Color in the body's own metadata says it outright.
void body_look(Node *p_body, Color &r_albedo, Color &r_emission) {
	r_albedo = Color(0.5, 0.5, 0.5);
	r_emission = Color(0, 0, 0);
	MeshInstance3D *mesh = first_mesh_child(p_body);
	if (mesh == nullptr) {
		mesh = Object::cast_to<MeshInstance3D>(p_body->get_parent());
	}
	// A mesh beside it only in a small group (a model node holding its mesh and its body), not among the thousands of
	// bodies a spawner keeps under one node.
	if (mesh == nullptr && p_body->get_parent() && p_body->get_parent()->get_child_count() <= 8) {
		mesh = first_mesh_child(p_body->get_parent());
	}
	if (mesh) {
		mesh_look(mesh, r_albedo, r_emission);
	}
	if (p_body->has_meta(meta_albedo())) {
		r_albedo = Color(p_body->get_meta(meta_albedo())).srgb_to_linear();
	}
	if (p_body->has_meta(meta_emission())) {
		r_emission = Color(p_body->get_meta(meta_emission())).srgb_to_linear();
		r_emission.a = 1.0f;
	}
}

// Whether the GI keeps track of a node: a light (which ones it uses is chosen each frame, as lights are often hidden
// and shown rather than added and removed), a reflection probe, a collision shape of a body (an Area3D's are trigger
// volumes, not surfaces: they neither block nor bounce light) or a mesh outside any body (which gives off light as its
// bounding box when it's emissive).
bool is_tracked(const Node *p_node) {
	if (bool(p_node->get_meta(meta_ignore(), false))) {
		return false;
	}
	if (Object::cast_to<Light3D>(p_node) || Object::cast_to<ReflectionProbe>(p_node)) {
		return true;
	}
	if (Object::cast_to<CollisionShape3D>(p_node)) {
		return Object::cast_to<CollisionObject3D>(p_node->get_parent()) && !Object::cast_to<Area3D>(p_node->get_parent());
	}
	return Object::cast_to<MeshInstance3D>(p_node) && !Object::cast_to<CollisionObject3D>(p_node->get_parent());
}

void collect(Node *p_node, HashSet<ObjectID> &r_tracked) {
	if (is_tracked(p_node)) {
		r_tracked.insert(p_node->get_instance_id());
	}
	for (int i = 0; i < p_node->get_child_count(); i++) {
		collect(p_node->get_child(i), r_tracked);
	}
}

// How much of a light Godot draws for a camera at p_eye: none when hidden or off, and with distance fade on, fading
// out past its begin distance as Godot's renderer does.
float light_presence(const Light3D *p_light, const Vector3 &p_eye) {
	if (p_light == nullptr || !p_light->is_visible_in_tree() || p_light->get_param(Light3D::PARAM_ENERGY) <= 0.0f) {
		return 0.0f;
	}
	if (!p_light->is_distance_fade_enabled()) {
		return 1.0f;
	}
	const float distance = p_light->get_global_position().distance_to(p_eye);
	const float begin = p_light->get_distance_fade_begin();
	if (distance <= begin) {
		return 1.0f;
	}
	return Math::smoothstep(0.0f, 1.0f, 1.0f - (distance - begin) / MAX(float(p_light->get_distance_fade_length()), 0.001f));
}

// Up to p_max of the lights Godot draws, those whose reach (distance less range) comes nearest p_eye first, each with
// how much of it is drawn.
LocalVector<Pair<Light3D *, float>> nearest_lights(const LocalVector<ObjectID> &p_lights, const Vector3 &p_eye, int p_max) {
	LocalVector<Pair<float, Pair<Light3D *, float>>> found;
	for (const ObjectID &id : p_lights) {
		Light3D *light = ObjectDB::get_instance<Light3D>(id);
		const float presence = light_presence(light, p_eye);
		if (presence <= 0.0f) {
			continue;
		}
		const float gap = MAX(light->get_global_position().distance_to(p_eye) - float(light->get_param(Light3D::PARAM_RANGE)), 0.0f);
		found.push_back(Pair<float, Pair<Light3D *, float>>(gap, Pair<Light3D *, float>(light, presence)));
	}
	if (found.size() > uint32_t(p_max)) {
		found.sort_custom<PairSort<float, Pair<Light3D *, float>>>();
		found.resize(p_max);
	}
	LocalVector<Pair<Light3D *, float>> chosen;
	for (const Pair<float, Pair<Light3D *, float>> &f : found) {
		chosen.push_back(f.second);
	}
	return chosen;
}

struct BuildItem {
	AABB bounds;
	Vector3 center;
	int index = 0;
	Vector3 lo; // the bounds' corners, for the SAH binning's plain min / max
	Vector3 hi;
};

// The binned SAH split of p_items[p_from, p_to): 12 bins per axis over the centres, the split that minimises both
// sides' surface area times shape count; returns the split point (items partitioned), or p_from when the centres all
// fall in one bin.
uint32_t sah_split(LocalVector<BuildItem> &p_items, uint32_t p_from, uint32_t p_to, const AABB &p_centres) {
	constexpr int BINS = 12;
	auto area = [](const float *lo, const float *hi) {
		const float x = hi[0] - lo[0];
		const float y = hi[1] - lo[1];
		const float z = hi[2] - lo[2];
		return 2.0f * (x * y + y * z + z * x);
	};
	const float inf = Math::INF;
	float best_cost = inf;
	int best_axis = -1;
	int best_bin = 0;
	for (int axis = 0; axis < 3; axis++) {
		const float lo = p_centres.position[axis];
		const float extent = p_centres.size[axis];
		if (extent <= 1e-6f) {
			continue;
		}
		const float scale = BINS / extent;
		float bin_lo[BINS][3];
		float bin_hi[BINS][3];
		uint32_t bin_count[BINS] = {};
		for (int b = 0; b < BINS; b++) {
			for (int k = 0; k < 3; k++) {
				bin_lo[b][k] = inf;
				bin_hi[b][k] = -inf;
			}
		}
		for (uint32_t i = p_from; i < p_to; i++) {
			const BuildItem &it = p_items[i];
			const int b = MIN(int((it.center[axis] - lo) * scale), BINS - 1);
			bin_count[b]++;
			for (int k = 0; k < 3; k++) {
				bin_lo[b][k] = MIN(bin_lo[b][k], it.lo[k]);
				bin_hi[b][k] = MAX(bin_hi[b][k], it.hi[k]);
			}
		}
		// Sweep: the left side's area x count for each split, then the right side's.
		float left_cost[BINS - 1];
		float acc_lo[3] = { inf, inf, inf };
		float acc_hi[3] = { -inf, -inf, -inf };
		uint32_t n = 0;
		for (int b = 0; b < BINS - 1; b++) {
			if (bin_count[b]) {
				for (int k = 0; k < 3; k++) {
					acc_lo[k] = MIN(acc_lo[k], bin_lo[b][k]);
					acc_hi[k] = MAX(acc_hi[k], bin_hi[b][k]);
				}
				n += bin_count[b];
			}
			left_cost[b] = n ? area(acc_lo, acc_hi) * n : 0.0f;
		}
		for (int k = 0; k < 3; k++) {
			acc_lo[k] = inf;
			acc_hi[k] = -inf;
		}
		n = 0;
		for (int b = BINS - 1; b > 0; b--) {
			if (bin_count[b]) {
				for (int k = 0; k < 3; k++) {
					acc_lo[k] = MIN(acc_lo[k], bin_lo[b][k]);
					acc_hi[k] = MAX(acc_hi[k], bin_hi[b][k]);
				}
				n += bin_count[b];
			}
			const uint32_t left_n = (p_to - p_from) - n;
			if (n == 0 || left_n == 0) {
				continue;
			}
			const float cost = left_cost[b - 1] + area(acc_lo, acc_hi) * n;
			if (cost < best_cost) {
				best_cost = cost;
				best_axis = axis;
				best_bin = b;
			}
		}
	}
	if (best_axis < 0) {
		return p_from;
	}
	const float lo = p_centres.position[best_axis];
	const float scale = BINS / p_centres.size[best_axis];
	BuildItem *mid = std::partition(p_items.ptr() + p_from, p_items.ptr() + p_to, [&](const BuildItem &it) {
		return MIN(int((it.center[best_axis] - lo) * scale), BINS - 1) < best_bin;
	});
	return uint32_t(mid - p_items.ptr());
}

// The tree: binned SAH splits (median where SAH finds none), leaves of up to four, depth first (an inner node's left
// child right after it), the primitives in leaf order. Never deeper than 15 levels (the shader's walk keeps a 16-entry
// stack): a node that deep is a leaf.
int build_bvh(LocalVector<BuildItem> &p_items, uint32_t p_from, uint32_t p_to, const float *p_source, LocalVector<float> &r_nodes, LocalVector<float> &r_primitives, LocalVector<int> &r_order, int p_depth = 0) {
	// The node's bounds and its shapes' centres' bounds, by plain min / max.
	Vector3 lo = p_items[p_from].lo;
	Vector3 hi = p_items[p_from].hi;
	Vector3 c_lo = p_items[p_from].center;
	Vector3 c_hi = c_lo;
	for (uint32_t i = p_from + 1; i < p_to; i++) {
		const BuildItem &it = p_items[i];
		lo = lo.min(it.lo);
		hi = hi.max(it.hi);
		c_lo = c_lo.min(it.center);
		c_hi = c_hi.max(it.center);
	}
	const AABB centres(c_lo, c_hi - c_lo);
	const int node = r_nodes.size() / PhysXGIEffect::NODE_FLOATS;
	r_nodes.resize(r_nodes.size() + PhysXGIEffect::NODE_FLOATS);
	float *out = &r_nodes[node * PhysXGIEffect::NODE_FLOATS];
	out[0] = lo.x;
	out[1] = lo.y;
	out[2] = lo.z;
	out[4] = hi.x;
	out[5] = hi.y;
	out[6] = hi.z;
	const uint32_t count = p_to - p_from;
	uint32_t mid = p_from;
	if (count > 4 && p_depth < 15) {
		mid = sah_split(p_items, p_from, p_to, centres);
		if (mid == p_from || mid == p_to) {
			const int axis = centres.get_longest_axis_index();
			mid = p_from + count / 2;
			std::nth_element(p_items.ptr() + p_from, p_items.ptr() + mid, p_items.ptr() + p_to, [axis](const BuildItem &a, const BuildItem &b) {
				return a.center[axis] < b.center[axis];
			});
		}
	}
	if (mid == p_from) {
		const int first = r_primitives.size() / PhysXGIEffect::PRIMITIVE_FLOATS;
		for (uint32_t i = p_from; i < p_to; i++) {
			const float *src = p_source + p_items[i].index * PhysXGIEffect::PRIMITIVE_FLOATS;
			r_order.push_back(p_items[i].index);
			for (int k = 0; k < PhysXGIEffect::PRIMITIVE_FLOATS; k++) {
				r_primitives.push_back(src[k]);
			}
		}
		memcpy(&out[3], &first, sizeof(int));
		const int leaf_count = count;
		memcpy(&out[7], &leaf_count, sizeof(int));
		return node;
	}
	build_bvh(p_items, p_from, mid, p_source, r_nodes, r_primitives, r_order, p_depth + 1);
	const int right = build_bvh(p_items, mid, p_to, p_source, r_nodes, r_primitives, r_order, p_depth + 1);
	out = &r_nodes[node * PhysXGIEffect::NODE_FLOATS];
	memcpy(&out[3], &right, sizeof(int));
	const int zero = 0;
	memcpy(&out[7], &zero, sizeof(int));
	return node;
}

// A mesh's triangles are split like the scene's shapes (binned SAH).
using TriangleItem = BuildItem;

void store_int(float *r_at, int p_value) {
	memcpy(r_at, &p_value, sizeof(int));
}

// A mesh's tree over its triangles (binned SAH splits, median where SAH finds none, leaves of up to four), in the scene
// tree's node layout; the triangles reordered to its leaves. Never deeper than 23 levels (the shader's mesh walk keeps
// a 24-entry stack).
int build_triangle_tree(LocalVector<TriangleItem> &p_items, uint32_t p_from, uint32_t p_to, const Vector<Vector3> &p_corners, Vector<float> &r_nodes, Vector<float> &r_triangles, int p_depth = 0) {
	AABB bounds = p_items[p_from].bounds;
	AABB centres(p_items[p_from].center, Vector3());
	for (uint32_t i = p_from + 1; i < p_to; i++) {
		bounds.merge_with(p_items[i].bounds);
		centres.expand_to(p_items[i].center);
	}
	const int node = r_nodes.size() / 8;
	r_nodes.resize(r_nodes.size() + 8);
	float *out = r_nodes.ptrw() + node * 8;
	const Vector3 lo = bounds.position;
	const Vector3 hi = bounds.position + bounds.size;
	out[0] = lo.x;
	out[1] = lo.y;
	out[2] = lo.z;
	out[4] = hi.x;
	out[5] = hi.y;
	out[6] = hi.z;
	const uint32_t count = p_to - p_from;
	uint32_t mid = p_from;
	if (count > 4 && p_depth < 23) {
		mid = sah_split(p_items, p_from, p_to, centres);
		if (mid == p_from || mid == p_to) {
			const int axis = centres.get_longest_axis_index();
			mid = p_from + count / 2;
			std::nth_element(p_items.ptr() + p_from, p_items.ptr() + mid, p_items.ptr() + p_to, [axis](const TriangleItem &a, const TriangleItem &b) {
				return a.center[axis] < b.center[axis];
			});
		}
	}
	if (mid == p_from) {
		store_int(&out[3], r_triangles.size() / 12);
		store_int(&out[7], count);
		for (uint32_t i = p_from; i < p_to; i++) {
			for (int c = 0; c < 3; c++) {
				const Vector3 v = p_corners[p_items[i].index * 3 + c];
				r_triangles.push_back(v.x);
				r_triangles.push_back(v.y);
				r_triangles.push_back(v.z);
				r_triangles.push_back(0.0f);
			}
		}
		return node;
	}
	build_triangle_tree(p_items, p_from, mid, p_corners, r_nodes, r_triangles, p_depth + 1);
	const int right = build_triangle_tree(p_items, mid, p_to, p_corners, r_nodes, r_triangles, p_depth + 1);
	out = r_nodes.ptrw() + node * 8;
	store_int(&out[3], right);
	store_int(&out[7], 0);
	return node;
}

} // namespace

// The planes of a convex hull, or a triangle mesh's (or height map's) tree and triangles, in the shape's local space;
// built the first time the shape is seen.
const PXTGI3D::GeometryBlock *PXTGI3D::_geometry_for(const Ref<Shape3D> &p_shape) {
	const ObjectID id = p_shape->get_instance_id();
	if (const GeometryBlock *known = geometry_blocks.getptr(id)) {
		return known->type ? known : nullptr;
	}
	GeometryBlock block;
	Vector<Vector3> corners;
	if (Ref<ConvexPolygonShape3D> convex = p_shape; convex.is_valid()) {
		const Vector<Vector3> points = convex->get_points();
		Geometry3D::MeshData hull;
		if (points.size() >= 4 && ConvexHullComputer::convex_hull(points, hull) == OK && hull.faces.size() >= 4) {
			block.type = 4;
			for (const Geometry3D::MeshData::Face &face : hull.faces) {
				block.planes.push_back(face.plane.normal.x);
				block.planes.push_back(face.plane.normal.y);
				block.planes.push_back(face.plane.normal.z);
				block.planes.push_back(face.plane.d);
			}
			block.bounds = AABB(points[0], Vector3());
			for (const Vector3 &v : points) {
				block.bounds.expand_to(v);
			}
		}
	} else if (Ref<ConcavePolygonShape3D> concave = p_shape; concave.is_valid()) {
		corners = concave->get_faces();
	} else if (Ref<HeightMapShape3D> height = p_shape; height.is_valid()) {
		// Godot's height map: a grid one unit apart, centred on the shape's origin.
		const int w = height->get_map_width();
		const int d = height->get_map_depth();
		const Vector<real_t> heights = height->get_map_data();
		if (w >= 2 && d >= 2 && heights.size() >= w * d) {
			auto corner = [&](int x, int z) {
				return Vector3(x - (w - 1) * 0.5f, heights[z * w + x], z - (d - 1) * 0.5f);
			};
			corners.resize((w - 1) * (d - 1) * 6);
			Vector3 *c = corners.ptrw();
			for (int z = 0; z < d - 1; z++) {
				for (int x = 0; x < w - 1; x++) {
					*c++ = corner(x, z);
					*c++ = corner(x + 1, z);
					*c++ = corner(x, z + 1);
					*c++ = corner(x + 1, z);
					*c++ = corner(x + 1, z + 1);
					*c++ = corner(x, z + 1);
				}
			}
		}
	}
	if (corners.size() >= 3) {
		LocalVector<TriangleItem> items;
		items.resize(corners.size() / 3);
		for (uint32_t i = 0; i < items.size(); i++) {
			AABB b(corners[i * 3], Vector3());
			b.expand_to(corners[i * 3 + 1]);
			b.expand_to(corners[i * 3 + 2]);
			items[i] = { b, b.get_center(), int(i), b.position, b.position + b.size };
		}
		build_triangle_tree(items, 0, items.size(), corners, block.nodes, block.triangles);
		block.type = 5;
		block.bounds = AABB(corners[0], Vector3());
		for (const Vector3 &v : corners) {
			block.bounds.expand_to(v);
		}
	}
	geometry_blocks.insert(id, block);
	const GeometryBlock *stored = geometry_blocks.getptr(id);
	return stored->type ? stored : nullptr;
}

// What the GI looks at: the scene (in the editor, the scene being edited, not the editor's own nodes).
Node *PXTGI3D::_scan_root() const {
	if (Engine::get_singleton()->is_editor_hint() && get_tree()->get_edited_scene_root()) {
		return get_tree()->get_edited_scene_root();
	}
	return get_tree()->get_root();
}

void PXTGI3D::_track_all(Node *p_root) {
	tracked.clear();
	looks.clear();
	texture_averages.clear();
	collect(p_root, tracked);
	tracked_root = p_root->get_instance_id();
	tracked_changed = true;
}

void PXTGI3D::_node_added(Node *p_node) {
	if (tracked_root.is_null() || !is_tracked(p_node)) {
		return;
	}
	const Node *root = ObjectDB::get_instance<Node>(tracked_root);
	if (root == nullptr || (root != get_tree()->get_root() && !root->is_ancestor_of(p_node))) {
		return;
	}
	const ObjectID id = p_node->get_instance_id();
	tracked.insert(id);
	// Lights and probes join their lists (which are used are chosen each frame; a node moved to another parent comes
	// back, once); a mesh waits for its look. Only a collision shape means finding the shapes again straight away.
	if (Object::cast_to<DirectionalLight3D>(p_node)) {
		if (suns.find(id) < 0) {
			suns.push_back(id);
		}
	} else if (Object::cast_to<OmniLight3D>(p_node)) {
		if (omnis.find(id) < 0) {
			omnis.push_back(id);
		}
	} else if (Object::cast_to<SpotLight3D>(p_node)) {
		if (spots.find(id) < 0) {
			spots.push_back(id);
		}
	} else if (Object::cast_to<ReflectionProbe>(p_node)) {
		if (probes.find(id) < 0) {
			probes.push_back(id);
		}
	} else if (Object::cast_to<MeshInstance3D>(p_node)) {
		pending_meshes.push_back(id);
	} else if (!Object::cast_to<Light3D>(p_node)) {
		tracked_changed = true;
		last_change = "added " + String(p_node->get_path());
	}
}

// A light or probe gone is skipped where it's used (no longer an object) until the next rescan drops it; only a shape
// the tree holds means finding the shapes again.
void PXTGI3D::_node_removed(Node *p_node) {
	const ObjectID id = p_node->get_instance_id();
	if (tracked.erase(id)) {
		looks.erase(id);
		if (entry_of.has(id)) {
			tracked_changed = true;
			last_change = "removed " + String(p_node->get_path());
		}
	}
}

// A tracked shape's look (its body's mesh) or a mesh's own: albedo, emission.
Pair<Color, Color> PXTGI3D::_look_of(Node *p_node) const {
	Pair<Color, Color> look;
	if (MeshInstance3D *mesh = Object::cast_to<MeshInstance3D>(p_node)) {
		mesh_look(mesh, look.first, look.second);
	} else {
		body_look(p_node->get_parent(), look.first, look.second);
	}
	return look;
}

// The looks of a slice of the tracked shapes and meshes, worked out again: a shape whose look changed is rewritten
// (r_changed); one that starts or stops giving off light, or a collision shape given another Shape3D, means finding
// the shapes again.
void PXTGI3D::_restyle_slice(LocalVector<int> &r_changed) {
	const uint32_t count = look_order.size();
	// About a second for all of them, but never more than 48 a frame however many there are.
	for (uint32_t k = 0; k < MIN(count / 60 + 1, 48u) && count > 0; k++) {
		const ObjectID id = look_order[look_cursor++ % count];
		Node *node = ObjectDB::get_instance<Node>(id);
		Pair<Color, Color> *known = looks.getptr(id);
		if (node == nullptr || known == nullptr) {
			continue;
		}
		Pair<Color, Color> look = _look_of(node);
		const int *index = entry_of.getptr(id);
		if (index && shapes[*index].driven) {
			look.second = known->second; // its glow comes from update_emitter_group
		}
		if (index) {
			const CollisionShape3D *cs = Object::cast_to<CollisionShape3D>(node);
			if (cs && (cs->get_shape().is_valid() ? cs->get_shape()->get_instance_id() : ObjectID()) != shapes[*index].shape_resource) {
				tracked_changed = true;
				last_change = "new shape " + String(node->get_path());
			}
		}
		if (look.first == known->first && look.second == known->second) {
			continue;
		}
		// Starting to glow only puts a shape among the emitters (a mesh outside any body joins the tree); nothing is
		// found again. One that goes dark stays where it is, unsampled.
		const bool glows = look.second.get_luminance() > 0.0f;
		if (glows && known->second.get_luminance() <= 0.0f) {
			if (index == nullptr) {
				if (MeshInstance3D *mesh = Object::cast_to<MeshInstance3D>(node)) {
					*known = look;
					_append_mesh(mesh, look);
					continue;
				}
			} else if (emissive_entries.find(*index) < 0) {
				emissive_entries.push_back(*index);
			}
		}
		*known = look;
		if (index) {
			shapes[*index].albedo = look.first;
			shapes[*index].emission = look.second;
			r_changed.push_back(*index);
		}
	}
}

void PXTGI3D::_rescan() {
	Node *root = _scan_root();
	if (root->get_instance_id() != tracked_root) {
		_track_all(root);
	}
	tracked_changed = false;
	frames_since_rescan = 0;
	shapes.clear();
	suns.clear();
	omnis.clear();
	spots.clear();
	probes.clear();
	look_order.clear();
	entry_of.clear();
	for (const ObjectID &id : tracked) {
		Node *n = ObjectDB::get_instance<Node>(id);
		if (n == nullptr || !is_tracked(n)) {
			continue;
		}
		if (Object::cast_to<DirectionalLight3D>(n)) {
			suns.push_back(id);
			continue;
		} else if (Object::cast_to<OmniLight3D>(n)) {
			omnis.push_back(id);
			continue;
		} else if (Object::cast_to<SpotLight3D>(n)) {
			spots.push_back(id);
			continue;
		} else if (Object::cast_to<ReflectionProbe>(n)) {
			probes.push_back(id);
			continue;
		} else if (Object::cast_to<Light3D>(n)) {
			continue;
		}
		const Pair<Color, Color> *look = looks.getptr(id);
		if (look == nullptr) {
			look = &looks.insert(id, _look_of(n))->value;
		}
		look_order.push_back(id);
		ShapeEntry entry;
		entry.shape_node = id;
		entry.albedo = look->first;
		entry.emission = look->second;
		if (MeshInstance3D *mesh = Object::cast_to<MeshInstance3D>(n)) {
			// A mesh outside any body only counts while it gives off light (and not when it's kept out of GI, as
			// Godot's own GI leaves it out).
			if (mesh->get_mesh().is_null() || entry.emission.get_luminance() <= 0.0f || mesh->get_gi_mode() == GeometryInstance3D::GI_MODE_DISABLED) {
				continue;
			}
			entry.mesh_proxy = true;
			entry.moving = true;
		} else {
			CollisionShape3D *cs = Object::cast_to<CollisionShape3D>(n);
			entry.fixed = Object::cast_to<StaticBody3D>(cs->get_parent()) && !Object::cast_to<AnimatableBody3D>(cs->get_parent());
			entry.sample_emission = bool(cs->get_parent()->get_meta(meta_sample_emission(), true));
			entry.moving = !entry.fixed;
			if (cs->get_shape().is_valid()) {
				entry.shape_resource = cs->get_shape()->get_instance_id();
				if (const GeometryBlock *block = _geometry_for(cs->get_shape())) {
					entry.geometry_type = block->type;
					entry.local_bounds = block->bounds;
				}
			}
		}
		entry_of.insert(id, shapes.size());
		shapes.push_back(entry);
	}
	body_entries.clear();
	dynamic_entries.clear();
	proxy_entries.clear();
	emissive_entries.clear();
	for (uint32_t i = 0; i < shapes.size(); i++) {
		if (shapes[i].emission.get_luminance() > 0.0f) {
			emissive_entries.push_back(i);
		}
	}
	shape_signature = 0;
	for (uint32_t i = 0; i < shapes.size(); i++) {
		Object *node = ObjectDB::get_instance(shapes[i].shape_node);
		shape_signature = hash_murmur3_one_64(uint64_t(shapes[i].shape_node), shape_signature);
		if (shapes[i].mesh_proxy) {
			proxy_entries.push_back(i);
		} else if (CollisionShape3D *cs = Object::cast_to<CollisionShape3D>(node)) {
			body_entries[cs->get_parent()->get_instance_id()].push_back(i);
			if (!shapes[i].fixed && !bool(cs->get_parent()->get_meta(meta_manual_updates(), false))) {
				dynamic_entries.push_back(i);
			}
			shape_signature = hash_murmur3_one_64(cs->get_shape().is_valid() ? uint64_t(cs->get_shape()->get_instance_id()) : 0, shape_signature);
		}
	}
	// The geometry buffer: each hull's planes, each mesh's tree and triangles, laid end to end (in vec4s); rebuilt only
	// when the set of shapes with geometry changes.
	uint32_t signature = 0;
	for (const ShapeEntry &entry : shapes) {
		if (entry.geometry_type) {
			CollisionShape3D *cs = ObjectDB::get_instance<CollisionShape3D>(entry.shape_node);
			signature = hash_murmur3_one_64(cs ? uint64_t(cs->get_shape()->get_instance_id()) : 0, signature);
		}
	}
	const bool rebuild = signature != geometry_signature;
	geometry_signature = signature;
	if (rebuild) {
		geometry.clear();
	}
	HashMap<ObjectID, Vector2i> placed;
	int cursor = 0; // in vec4s: where the next block goes (the same places as when it was built)
	for (ShapeEntry &entry : shapes) {
		if (!entry.geometry_type) {
			continue;
		}
		CollisionShape3D *cs = ObjectDB::get_instance<CollisionShape3D>(entry.shape_node);
		const ObjectID shape_id = cs->get_shape()->get_instance_id();
		if (const Vector2i *at = placed.getptr(shape_id)) {
			entry.geometry_a = at->x;
			entry.geometry_b = at->y;
			continue;
		}
		const GeometryBlock &block = geometry_blocks[shape_id];
		const int base = cursor;
		if (block.type == 4) {
			entry.geometry_a = base;
			entry.geometry_b = block.planes.size() / 4;
			cursor += block.planes.size() / 4;
			if (rebuild) {
				geometry.append_array(block.planes);
			}
		} else {
			// The mesh's node and triangle indices are relative: the shader adds these bases.
			entry.geometry_a = base;
			entry.geometry_b = base + block.nodes.size() / 4;
			cursor += (block.nodes.size() + block.triangles.size()) / 4;
			if (rebuild) {
				geometry.append_array(block.nodes);
				geometry.append_array(block.triangles);
			}
		}
		placed.insert(shape_id, Vector2i(entry.geometry_a, entry.geometry_b));
	}
	if (rebuild) {
		geometry_version++;
	}
}

// Where shape p_index is and its local bounds (an emissive's box), from its record.
void PXTGI3D::_record_frame(int p_index, Transform3D &r_to_world, AABB &r_local) const {
	const float *record = &entry_floats[p_index * PhysXGIEffect::PRIMITIVE_FLOATS];
	// The record holds world-to-local rows: their inverse is the shape's frame.
	Transform3D to_local;
	for (int r = 0; r < 3; r++) {
		to_local.basis.rows[r] = Vector3(record[r * 4 + 0], record[r * 4 + 1], record[r * 4 + 2]);
		to_local.origin[r] = record[r * 4 + 3];
	}
	r_to_world = to_local.affine_inverse();
	const ShapeEntry &entry = shapes[p_index];
	if (entry.geometry_type) {
		r_local = entry.local_bounds;
	} else {
		const Vector3 e(record[13], record[14], record[15]);
		r_local = record[12] == 1.0f ? AABB(-Vector3(e.x, e.x, e.x), Vector3(2, 2, 2) * e.x) : AABB(-e, e * 2.0f);
	}
}

// Where shape p_index is and what it is (type, extents), from its node; false when it can't be traced (gone, hidden,
// disabled).
bool PXTGI3D::_shape_frame(int p_index, Transform3D &r_to_world, float &r_type, Vector3 &r_extents) const {
	const ShapeEntry &entry = shapes[p_index];
	float type = 0.0f;
	Vector3 extents;
	Transform3D to_world;
	if (entry.mesh_proxy) {
		MeshInstance3D *mesh = ObjectDB::get_instance<MeshInstance3D>(entry.shape_node);
		if (mesh == nullptr || !mesh->is_visible_in_tree() || mesh->get_mesh().is_null()) {
			return false;
		}
		const AABB bounds = mesh->get_aabb();
		extents = bounds.size * 0.5f;
		to_world = mesh->get_global_transform() * Transform3D(Basis(), bounds.get_center());
	} else {
		CollisionShape3D *cs = ObjectDB::get_instance<CollisionShape3D>(entry.shape_node);
		if (cs == nullptr || cs->is_disabled() || !cs->is_inside_tree() || !cs->is_visible_in_tree() || cs->get_shape().is_null()) {
			return false;
		}
		to_world = cs->get_global_transform();
		Ref<Shape3D> shape = cs->get_shape();
		if (entry.geometry_type) {
			type = float(entry.geometry_type);
		} else if (Ref<BoxShape3D> box = shape; box.is_valid()) {
			extents = box->get_size() * 0.5f;
		} else if (Ref<SphereShape3D> sphere = shape; sphere.is_valid()) {
			type = 1.0f;
			extents = Vector3(sphere->get_radius(), 0, 0);
		} else if (Ref<CapsuleShape3D> capsule = shape; capsule.is_valid()) {
			type = 2.0f;
			extents = Vector3(capsule->get_radius(), capsule->get_height() * 0.5f, capsule->get_radius());
		} else if (Ref<CylinderShape3D> cylinder = shape; cylinder.is_valid()) {
			type = 3.0f;
			extents = Vector3(cylinder->get_radius(), cylinder->get_height() * 0.5f, cylinder->get_radius());
		} else {
			return false;
		}
	}
	r_to_world = to_world;
	r_type = type;
	r_extents = extents;
	return true;
}

// Shape p_index's GPU record and world bounds, placed at p_to_world.
void PXTGI3D::_pack_entry(int p_index, const Transform3D &p_to_world, float p_type, const Vector3 &p_extents, float *r_out, AABB &r_bounds) const {
	const ShapeEntry &entry = shapes[p_index];
	const float type = p_type;
	const Vector3 extents = p_extents;
	const Transform3D to_local = p_to_world.affine_inverse();
	const AABB local_bounds = entry.geometry_type ? entry.local_bounds : (type == 1.0f ? AABB(-Vector3(extents.x, extents.x, extents.x), Vector3(2, 2, 2) * extents.x) : AABB(-extents, extents * 2.0f));
	r_bounds = p_to_world.xform(local_bounds);
	float *p = r_out;
	for (int r = 0; r < 3; r++) {
		p[r * 4 + 0] = to_local.basis.rows[r].x;
		p[r * 4 + 1] = to_local.basis.rows[r].y;
		p[r * 4 + 2] = to_local.basis.rows[r].z;
		p[r * 4 + 3] = to_local.origin[r];
	}
	p[12] = type;
	p[13] = extents.x;
	p[14] = extents.y;
	p[15] = extents.z;
	if (entry.geometry_type) {
		store_int(&p[13], entry.geometry_a);
		store_int(&p[14], entry.geometry_b);
	}
	p[16] = entry.albedo.r;
	p[17] = entry.albedo.g;
	p[18] = entry.albedo.b;
	p[19] = 1.0f;
	p[20] = entry.emission.r;
	p[21] = entry.emission.g;
	p[22] = entry.emission.b;
	p[23] = entry.fixed ? 1.0f : 0.0f;
}

// Shape p_index's GPU record and world bounds, from its node (false when it can't be traced), also noting where the
// shape sits on its body, for the frames its body moves.
bool PXTGI3D::_refresh_entry(int p_index, float *r_out, AABB &r_bounds) {
	ShapeEntry &entry = shapes[p_index];
	entry.posed = false;
	if (!_shape_frame(p_index, entry.in_body, entry.type, entry.extents)) {
		return false;
	}
	const Transform3D to_world = entry.in_body;
	_pack_entry(p_index, to_world, entry.type, entry.extents, r_out, r_bounds);
	if (!entry.mesh_proxy) {
		const Node3D *body = ObjectDB::get_instance<CollisionShape3D>(entry.shape_node)->get_parent_node_3d();
		if (body) {
			entry.in_body = body->get_global_transform().affine_inverse() * to_world;
			entry.posed = true;
		}
	}
	return true;
}

// After a rescan: each shape still there (the same Shape3D) keeps its record, pose and tree slot from before, with its
// look and geometry offsets rewritten; only new shapes are read from their nodes. The static part of the tree is kept
// when it holds the same shapes, so a ribbon or a streamed cell coming and going doesn't rebuild the whole scene.
void PXTGI3D::_keep_records(const LocalVector<ShapeEntry> &p_was, const HashMap<ObjectID, int> &p_was_index, const LocalVector<float> &p_was_floats, const LocalVector<AABB> &p_was_bounds, const LocalVector<uint8_t> &p_was_valid) {
	constexpr int PF = PhysXGIEffect::PRIMITIVE_FLOATS;
	const int n = shapes.size();
	entry_floats.resize(n * PF);
	entry_bounds.resize(n);
	entry_valid.resize(n);
	LocalVector<int> now_index; // a former shape's index now, or -1
	now_index.resize(p_was.size());
	for (uint32_t i = 0; i < p_was.size(); i++) {
		now_index[i] = -1;
	}
	for (int i = 0; i < n; i++) {
		ShapeEntry &entry = shapes[i];
		const int *was = p_was_index.getptr(entry.shape_node);
		if (was == nullptr || p_was[*was].shape_resource != entry.shape_resource || p_was[*was].mesh_proxy != entry.mesh_proxy) {
			entry_valid[i] = _refresh_entry(i, &entry_floats[i * PF], entry_bounds[i]);
			continue;
		}
		const ShapeEntry &before = p_was[*was];
		now_index[*was] = i;
		entry.posed = before.posed;
		entry.in_body = before.in_body;
		entry.type = before.type;
		entry.extents = before.extents;
		entry.moving = entry.moving || before.moving;
		entry.driven = before.driven;
		if (entry.driven) {
			entry.emission = before.emission;
		}
		float *record = &entry_floats[i * PF];
		memcpy(record, &p_was_floats[*was * PF], PF * sizeof(float));
		if (entry.geometry_type) {
			store_int(&record[13], entry.geometry_a);
			store_int(&record[14], entry.geometry_b);
		}
		record[16] = entry.albedo.r;
		record[17] = entry.albedo.g;
		record[18] = entry.albedo.b;
		record[20] = entry.emission.r;
		record[21] = entry.emission.g;
		record[22] = entry.emission.b;
		entry_bounds[i] = p_was_bounds[*was];
		entry_valid[i] = p_was_valid[*was];
	}
	// The static part stays when every static shape in it is still there and still static, and no other static shape
	// is waiting to join it; its leaves then point at the shapes' new indices and take their rewritten records.
	uint32_t static_count = 0;
	for (int i = 0; i < n; i++) {
		static_count += (!shapes[i].moving && entry_valid[i]) ? 1 : 0;
	}
	if (static_count != static_leaves) {
		static_dirty = true;
	}
	for (uint32_t k = 0; k < static_leaves && !static_dirty; k++) {
		const int i = now_index[leaf_order[k]];
		if (i < 0 || shapes[i].moving || !entry_valid[i]) {
			static_dirty = true;
			break;
		}
		leaf_order[k] = i;
		memcpy(&ordered_floats[k * PF], &entry_floats[i * PF], PF * sizeof(float));
	}
	leaf_order.resize(static_leaves);
	// Slots for the kept static leaves; the moving part's come with its rebuild this frame.
	entry_slot.resize(n);
	for (int i = 0; i < n; i++) {
		entry_slot[i] = -1;
	}
	for (uint32_t k = 0; k < static_leaves && !static_dirty; k++) {
		entry_slot[leaf_order[k]] = k;
	}
	moving_dirty = true;
}

// The tree for the shapes as they are: the static part rebuilt only when it changed (or the root joining it to the
// moving part comes or goes), the moving part every time.
void PXTGI3D::_build_tree() {
	constexpr int NF = PhysXGIEffect::NODE_FLOATS;
	const int n = shapes.size();
	LocalVector<BuildItem> moving_items;
	for (int i : moving_entries) {
		if (entry_valid[i]) {
			const AABB &b = entry_bounds[i];
			moving_items.push_back({ b, b.get_center(), i, b.position, b.position + b.size });
		}
	}
	LocalVector<BuildItem> static_items;
	auto collect_static = [&]() {
		for (int i = 0; i < n; i++) {
			if (!shapes[i].moving && entry_valid[i]) {
				const AABB &b = entry_bounds[i];
				static_items.push_back({ b, b.get_center(), i, b.position, b.position + b.size });
			}
		}
	};
	if (static_dirty) {
		collect_static();
	}
	const bool rooted = !moving_items.is_empty() && (static_dirty ? !static_items.is_empty() : static_leaves > 0);
	if (static_dirty || rooted != static_rooted) {
		const uint64_t since = OS::get_singleton()->get_ticks_usec();
		if (!static_dirty) {
			collect_static(); // the same shapes, laid out with or without the shared root
		}
		tree_nodes.clear();
		ordered_floats.clear();
		leaf_order.clear();
		if (rooted) {
			tree_nodes.resize(NF);
		}
		if (!static_items.is_empty()) {
			build_bvh(static_items, 0, static_items.size(), entry_floats.ptr(), tree_nodes, ordered_floats, leaf_order, rooted ? 1 : 0);
		}
		static_node_floats = tree_nodes.size();
		static_record_floats = ordered_floats.size();
		static_leaves = leaf_order.size();
		static_rooted = rooted;
		static_dirty = false;
		_stage(STAGE_STATIC_TREE, since);
	}
	const uint64_t since = OS::get_singleton()->get_ticks_usec();
	tree_nodes.resize(static_node_floats);
	ordered_floats.resize(static_record_floats);
	leaf_order.resize(static_leaves);
	if (!moving_items.is_empty()) {
		const int moving_root = build_bvh(moving_items, 0, moving_items.size(), entry_floats.ptr(), tree_nodes, ordered_floats, leaf_order, rooted ? 1 : 0);
		if (rooted) {
			// The root: both parts' bounds; its left child (the static part) follows it, its right is the moving part.
			float *root = tree_nodes.ptr();
			const float *a = &tree_nodes[NF];
			const float *b = &tree_nodes[moving_root * NF];
			for (int k = 0; k < 3; k++) {
				root[k] = MIN(a[k], b[k]);
				root[4 + k] = MAX(a[4 + k], b[4 + k]);
			}
			store_int(&root[3], moving_root);
			store_int(&root[7], 0);
		}
	}
	entry_slot.resize(n);
	for (int i = 0; i < n; i++) {
		entry_slot[i] = -1;
	}
	for (uint32_t k = 0; k < leaf_order.size(); k++) {
		entry_slot[leaf_order[k]] = k;
	}
	moving_dirty = false;
	_stage(STAGE_MOVING_TREE, since);
}

// A mesh outside any body that starts to glow: appended to the shapes and the moving part of the tree, without
// finding every shape again.
void PXTGI3D::_append_mesh(MeshInstance3D *p_mesh, const Pair<Color, Color> &p_look) {
	constexpr int PF = PhysXGIEffect::PRIMITIVE_FLOATS;
	const ObjectID id = p_mesh->get_instance_id();
	if (entry_of.has(id) || p_mesh->get_mesh().is_null() || p_mesh->get_gi_mode() == GeometryInstance3D::GI_MODE_DISABLED) {
		return;
	}
	ShapeEntry entry;
	entry.shape_node = id;
	entry.albedo = p_look.first;
	entry.emission = p_look.second;
	entry.mesh_proxy = true;
	entry.moving = true;
	const int i = shapes.size();
	entry_of.insert(id, i);
	shapes.push_back(entry);
	proxy_entries.push_back(i);
	emissive_entries.push_back(i);
	moving_entries.push_back(i);
	entry_floats.resize((i + 1) * PF);
	entry_bounds.resize(i + 1);
	entry_valid.resize(i + 1);
	entry_valid[i] = _refresh_entry(i, &entry_floats[i * PF], entry_bounds[i]);
	moving_dirty = true;
}

// Shape p_index's glow, written into its record (and its tree slot) in place: the tree doesn't change.
void PXTGI3D::_set_glow(int p_index, const Color &p_glow) {
	constexpr int PF = PhysXGIEffect::PRIMITIVE_FLOATS;
	ShapeEntry &entry = shapes[p_index];
	if (entry.emission == p_glow) {
		return;
	}
	entry.emission = p_glow;
	if (Pair<Color, Color> *look = looks.getptr(entry.shape_node)) {
		look->second = p_glow;
	}
	if (uint32_t(p_index) < entry_valid.size()) {
		float *record = &entry_floats[p_index * PF];
		record[20] = p_glow.r;
		record[21] = p_glow.g;
		record[22] = p_glow.b;
		if (uint32_t(p_index) < entry_slot.size() && entry_slot[p_index] >= 0) {
			memcpy(&ordered_floats[entry_slot[p_index] * PF + 20], &record[20], 3 * sizeof(float));
		}
		radiance_dirty = true;
	}
}

// The shapes' records and their tree, refreshed for what changed. Shapes found again (when tracked nodes came or
// went) keep their records; otherwise only the moving shapes, those whose look changed and a slice of every shape each
// frame (to catch shapes edited, hidden or disabled) are read from their nodes. A record whose pose or shape changed
// rebuilds its part of the tree (a static shape that moves joins the moving part); a look alone is rewritten in place.
void PXTGI3D::_update_primitives(PhysXGIEffect::Snapshot &r_snapshot) {
	constexpr int PF = PhysXGIEffect::PRIMITIVE_FLOATS;
	for (const ObjectID &id : pending_meshes) {
		MeshInstance3D *mesh = ObjectDB::get_instance<MeshInstance3D>(id);
		if (mesh && tracked.has(id) && mesh->get_mesh().is_valid() && mesh->get_gi_mode() != GeometryInstance3D::GI_MODE_DISABLED) {
			const Pair<Color, Color> look = _look_of(mesh);
			looks.insert(id, look);
			if (look.second.get_luminance() > 0.0f) {
				_append_mesh(mesh, look);
			}
		}
	}
	pending_meshes.clear();
	// The shapes found again when tracked nodes came or went, at most every 10 frames (a stream of spawned debris
	// doesn't rescan every frame).
	if (tracked_root.is_null() || (tracked_changed && ++frames_since_rescan >= 10)) {
		const uint64_t since = OS::get_singleton()->get_ticks_usec();
		const LocalVector<ShapeEntry> was(shapes);
		const HashMap<ObjectID, int> was_index(entry_of);
		const LocalVector<float> was_floats(entry_floats);
		const LocalVector<AABB> was_bounds(entry_bounds);
		const LocalVector<uint8_t> was_valid(entry_valid);
		_rescan();
		_keep_records(was, was_index, was_floats, was_bounds, was_valid);
		_stage(STAGE_RESCAN, since);
		if (!last_change.is_empty()) {
			rescan_causes.push_back(last_change);
			last_change = String();
		}
	}
	const int n = shapes.size();
	moving_entries.clear();
	for (int i = 0; i < n; i++) {
		if (shapes[i].moving) {
			moving_entries.push_back(i);
		}
	}

	const uint64_t records_since = OS::get_singleton()->get_ticks_usec();
	// Each candidate with the pose of the body that moved it (its shapes placed from it), or none (read from its node).
	LocalVector<Pair<int, const Transform3D *>> candidates;
	LocalVector<int> restyled;
	_restyle_slice(restyled);
	for (int i : restyled) {
		candidates.push_back({ i, nullptr });
	}
	// Presentation-driven movers are checked every frame. Static geometry stays cached; explicit notifications refresh
	// manually maintained shapes (ribbons), while an amortized verification catches authoring edits and parent
	// visibility changes.
	for (int i : dynamic_entries) {
		candidates.push_back({ i, nullptr });
	}
	for (int i : proxy_entries) {
		candidates.push_back({ i, nullptr });
	}
	for (const KeyValue<ObjectID, Transform3D> &moved : moved_poses) {
		if (const LocalVector<int> *entries = body_entries.getptr(moved.key)) {
			for (int i : *entries) {
				if (Node *node = ObjectDB::get_instance<Node>(shapes[i].shape_node)) {
					Pair<Color, Color> look = _look_of(node);
					if (shapes[i].driven) {
						look.second = shapes[i].emission;
					}
					shapes[i].albedo = look.first;
					shapes[i].emission = look.second;
					looks.insert(shapes[i].shape_node, look);
				}
				candidates.push_back({ i, nullptr });
			}
		}
	}
	for (int k = 0; k < MIN(n / 60 + 1, 48) && n > 0; k++) {
		candidates.push_back({ int(verify_cursor++ % n), nullptr });
	}
	float scratch[PF];
	AABB bounds;
	for (const Pair<int, const Transform3D *> &c : candidates) {
		const int i = c.first;
		ShapeEntry &entry = shapes[i];
		bool valid;
		if (c.second && entry.posed && entry_valid[i]) {
			_pack_entry(i, *c.second * entry.in_body, entry.type, entry.extents, scratch, bounds);
			valid = true;
		} else {
			valid = _refresh_entry(i, scratch, bounds);
		}
		float *record = &entry_floats[i * PF];
		if (valid != bool(entry_valid[i])) {
			entry_valid[i] = valid;
			if (valid) {
				memcpy(record, scratch, sizeof(scratch));
				entry_bounds[i] = bounds;
			}
			(entry.moving ? moving_dirty : static_dirty) = true;
			records_changed++;
		} else if (valid && memcmp(scratch, record, sizeof(scratch)) != 0) {
			// The first 16 floats are its pose and shape; the rest its look.
			const bool placed = memcmp(scratch, record, 16 * sizeof(float)) != 0;
			memcpy(record, scratch, sizeof(scratch));
			entry_bounds[i] = bounds;
			records_changed++;
			if (placed) {
				if (!entry.moving) {
					entry.moving = true;
					moving_entries.push_back(i);
					static_dirty = true;
				}
				moving_dirty = true;
			} else if (uint32_t(i) < entry_slot.size() && entry_slot[i] >= 0) {
				memcpy(&ordered_floats[entry_slot[i] * PF], record, sizeof(scratch));
				radiance_dirty = true;
			}
		}
	}
	_stage(STAGE_RECORDS, records_since);

	const bool rebuilt = static_dirty || moving_dirty || entry_slot.size() != uint32_t(n);
	if (rebuilt) {
		_build_tree();
	}
	if (rebuilt || radiance_dirty) {
		const uint64_t copy_since = OS::get_singleton()->get_ticks_usec();
		published_primitives.resize(ordered_floats.size());
		if (ordered_floats.size()) {
			memcpy(published_primitives.ptrw(), ordered_floats.ptr(), ordered_floats.size() * sizeof(float));
		}
		if (rebuilt) {
			published_nodes.resize(tree_nodes.size());
			if (tree_nodes.size()) {
				memcpy(published_nodes.ptrw(), tree_nodes.ptr(), tree_nodes.size() * sizeof(float));
			}
		}
		published_count = leaf_order.size();
		primitives_version++;
		radiance_dirty = false;
		_stage(STAGE_COPY, copy_since);
	}
	r_snapshot.primitives = published_primitives;
	r_snapshot.primitive_count = published_count;
	r_snapshot.nodes = published_nodes;
	r_snapshot.node_count = published_nodes.size() / PhysXGIEffect::NODE_FLOATS;
	r_snapshot.primitives_version = primitives_version;
	moved_poses.clear();
}

void PXTGI3D::notify_body_changed(Node3D *p_body) {
	ERR_FAIL_NULL(p_body);
	moved_poses.insert(p_body->get_instance_id(), p_body->get_global_transform());
}

// Batched bridge for the underground pulse material. Parameters are shared by a group; each body's glow is written
// into its shapes' records in place (geometry and the tree stay as they are). A body no scan has found yet takes it
// through its metadata, as any look.
void PXTGI3D::update_emitter_group(const PackedInt64Array &p_bodies, const PackedFloat32Array &p_stations, const Dictionary &p_values, double p_time) {
	ERR_FAIL_COND(p_bodies.size() != p_stations.size());
	const uint64_t since = OS::get_singleton()->get_ticks_usec();
	const Color colour = Color(p_values["colour"]).srgb_to_linear();
	const float energy = p_values["energy"];
	const float presence = 1.0f + float(p_values["presence_gain"]) * float(p_values["presence"]);
	const float age = p_values["wave_age"];
	const float wave_speed = p_values["wave_speed"];
	const float wave_width = p_values["wave_width"];
	const float wave_gain = p_values["wave_gain"];
	const float life = p_values["wave_life"];
	const Vector3 origin = p_values["wave_origin"];
	const float chase = p_values["chase"];
	const float chase_gain = p_values["chase_gain"];
	const float wavelength = p_values["wavelength"];
	const float width = p_values["width"];
	const float speed = p_values["speed"];
	const float fade = age >= 0.0f ? 1.0f - Math::smoothstep(life * 0.5f, life, age) : 0.0f;
	for (int i = 0; i < p_bodies.size(); i++) {
		const ObjectID id(uint64_t(p_bodies[i]));
		const LocalVector<int> *entries = body_entries.getptr(id);
		const bool found = entries != nullptr && !entries->is_empty() && uint32_t((*entries)[0]) < entry_bounds.size();
		Node3D *body = found ? nullptr : ObjectDB::get_instance<Node3D>(id);
		if (!found && body == nullptr) {
			continue;
		}
		float gain = presence;
		if (chase > 0.0f) {
			const float ahead = Math::fposmod(p_stations[i] - float(p_time) * speed, wavelength);
			const float d = MIN(ahead, wavelength - ahead);
			gain += chase_gain * chase * Math::exp(-d * d / (width * width));
		}
		if (fade > 0.0f) {
			// An emitter box is centred on its body.
			const Vector3 at = found ? entry_bounds[(*entries)[0]].get_center() : body->get_global_position();
			const float front = at.distance_to(origin) - age * wave_speed;
			gain += wave_gain * (front > 0.0f ? Math::exp(-front * front / (wave_width * wave_width)) : Math::exp(front / (wave_width * 4.0f)) * 0.5f) * fade;
		}
		Color glow = colour * (energy * gain);
		glow.a = 1.0f;
		if (!found) {
			body->set_meta(meta_emission(), glow.linear_to_srgb());
			continue;
		}
		for (int e : *entries) {
			shapes[e].driven = true;
			_set_glow(e, glow);
		}
	}
	_stage(STAGE_EMITTERS, since);
}

void PXTGI3D::_stage(Stage p_stage, uint64_t p_since) {
	const uint64_t spent = OS::get_singleton()->get_ticks_usec() - p_since;
	StageTime &t = stage_times[p_stage];
	t.total += spent;
	t.worst = MAX(t.worst, spent);
	t.calls++;
}

// Each stage's time since the last reset: total and worst microseconds, and calls (frames for the whole update); the
// shapes, emitters and moving records it keeps.
Dictionary PXTGI3D::get_stats(bool p_reset) {
	static const char *names[STAGE_MAX] = { "publish", "emitters", "rescan", "records", "static_tree", "moving_tree", "select", "copy" };
	Dictionary stats;
	for (int i = 0; i < STAGE_MAX; i++) {
		Dictionary stage;
		stage["total_us"] = int64_t(stage_times[i].total);
		stage["worst_us"] = int64_t(stage_times[i].worst);
		stage["calls"] = int64_t(stage_times[i].calls);
		stats[names[i]] = stage;
	}
	stats["records_changed"] = int64_t(records_changed);
	stats["shapes"] = int64_t(shapes.size());
	stats["emissive"] = int64_t(emissive_entries.size());
	stats["dynamic"] = int64_t(dynamic_entries.size());
	stats["moving"] = int64_t(moving_entries.size());
	stats["static_leaves"] = int64_t(static_leaves);
	stats["geometry_floats"] = int64_t(geometry.size());
	stats["rescan_causes"] = rescan_causes;
	stats["mesh_proxies"] = int64_t(proxy_entries.size());
	if (p_reset) {
		for (StageTime &t : stage_times) {
			t = StageTime();
		}
		records_changed = 0;
		rescan_causes.clear();
	}
	return stats;
}

void PXTGI3D::_publish() {
	if (!is_visible_in_tree()) {
		PhysXGIEffect::remove_settings(get_instance_id());
		return;
	}
	const uint64_t publish_since = OS::get_singleton()->get_ticks_usec();
	PhysXGIEffect::Snapshot s;
	s.intensity = intensity;
	s.max_distance = max_distance;
	s.world_distance = world_ray_distance;
	s.reflection_distance = reflection_distance;
	s.filtered_reflections = filtered_reflections;
	s.reflection_softness = reflection_softness;
	s.reflection_resolution_scale = reflection_resolution_scale;
	s.debug = debug;
	s.multi_bounce = multi_bounce;
	s.rays_per_texel = rays_per_texel;
	s.reflection_mode = reflection_mode;
	s.reflection_cutoff = reflection_cutoff;

	_update_primitives(s);
	const Camera3D *camera = get_viewport() ? get_viewport()->get_camera_3d() : nullptr;
	const Vector3 eye = camera ? camera->get_global_position() : get_global_position();
	{
		const uint64_t select_since = OS::get_singleton()->get_ticks_usec();
		// The nearest by their records' bounds (a box's world bounds share its centre): no node is read.
		LocalVector<Pair<float, int>> nearby_emitters;
		nearby_emitters.reserve(emissive_entries.size());
		for (int i : emissive_entries) {
			if (shapes[i].sample_emission && entry_valid[i] && shapes[i].emission.get_luminance() > 0.0f) {
				nearby_emitters.push_back(Pair<float, int>(entry_bounds[i].get_center().distance_squared_to(eye), i));
			}
		}
		const uint32_t chosen = MIN(nearby_emitters.size(), uint32_t(PhysXGIEffect::Snapshot::MAX_EMISSIVES));
		std::partial_sort(nearby_emitters.ptr(), nearby_emitters.ptr() + chosen, nearby_emitters.ptr() + nearby_emitters.size(), [](const Pair<float, int> &a, const Pair<float, int> &b) {
			return a.first < b.first;
		});
		s.emissive_count = 0;
		for (uint32_t c = 0; c < chosen; c++) {
			const Pair<float, int> &e = nearby_emitters[c];
			Transform3D frame;
			AABB local;
			_record_frame(e.second, frame, local);
			const int k = s.emissive_count++;
			s.emissive_primitive[k] = entry_slot[e.second];
			const Vector3 half = local.size * 0.5f;
			s.emissive_center[k] = frame.xform(local.get_center());
			for (int a = 0; a < 3; a++) {
				Vector3 axis;
				axis[a] = half[a];
				s.emissive_axis[k][a] = frame.basis.xform(axis);
			}
			s.emissive_color[k] = shapes[e.second].emission;
		}
		_stage(STAGE_SELECT, select_since);
	}
	s.geometry = geometry;
	s.geometry_version = geometry_version;

	// The sun: the brightest directional light Godot draws on surfaces (a hidden one, or a sky-only moon, isn't).
	s.sun_color = Color(0, 0, 0);
	for (const ObjectID &id : suns) {
		DirectionalLight3D *light = ObjectDB::get_instance<DirectionalLight3D>(id);
		if (light_presence(light, eye) <= 0.0f || light->get_sky_mode() == DirectionalLight3D::SKY_MODE_SKY_ONLY) {
			continue;
		}
		const Color color = light_strength(light);
		if (color.get_luminance() > s.sun_color.get_luminance()) {
			s.sun_direction = light->get_global_transform().basis.get_column(2).normalized();
			s.sun_color = color;
		}
	}
	s.omni_count = 0;
	for (const Pair<Light3D *, float> &l : nearest_lights(omnis, eye, PhysXGIEffect::Snapshot::MAX_OMNIS)) {
		const OmniLight3D *omni = static_cast<OmniLight3D *>(l.first);
		const int i = s.omni_count++;
		s.omni_position[i] = omni->get_global_position();
		s.omni_inv_range[i] = 1.0f / MAX(omni->get_param(Light3D::PARAM_RANGE), 0.001f);
		s.omni_color[i] = light_strength(omni) * l.second;
		s.omni_attenuation[i] = omni->get_param(Light3D::PARAM_ATTENUATION);
	}
	s.spot_count = 0;
	for (const Pair<Light3D *, float> &l : nearest_lights(spots, eye, PhysXGIEffect::Snapshot::MAX_SPOTS)) {
		const SpotLight3D *spot = static_cast<SpotLight3D *>(l.first);
		const int i = s.spot_count++;
		const Transform3D xf = spot->get_global_transform();
		s.spot_position[i] = xf.origin;
		s.spot_inv_range[i] = 1.0f / MAX(spot->get_param(Light3D::PARAM_RANGE), 0.001f);
		s.spot_direction[i] = -xf.basis.get_column(2).normalized();
		s.spot_cos_angle[i] = Math::cos(Math::deg_to_rad(spot->get_param(Light3D::PARAM_SPOT_ANGLE)));
		s.spot_color[i] = light_strength(spot) * l.second;
		s.spot_attenuation[i] = spot->get_param(Light3D::PARAM_ATTENUATION);
		s.spot_cone_attenuation[i] = 1.0f / MAX(spot->get_param(Light3D::PARAM_SPOT_ATTENUATION), 0.001f);
	}
	// Reflection probes: inside their box Godot's scene shader replaces the GI buffers' ambient (ambient mode other
	// than Disabled) and reflections (intensity above 0) with the probe's, so those pixels needn't be traced. Only
	// where the override is whole -- inside the blend distance, less a margin for the filters' neighbours -- and only
	// for probes reaching every layer (the trace can't tell a pixel's layers).
	s.probe_count = 0;
	for (const ObjectID &id : probes) {
		ReflectionProbe *probe = ObjectDB::get_instance<ReflectionProbe>(id);
		if (probe == nullptr || !probe->is_visible_in_tree() || (probe->get_reflection_mask() & 0xFFFFF) != 0xFFFFF) {
			continue;
		}
		const int overrides = (probe->get_ambient_mode() != ReflectionProbe::AMBIENT_DISABLED ? 1 : 0) | (probe->get_intensity() > 0.0f ? 2 : 0);
		const Vector3 core = probe->get_size() * 0.5f - Vector3(1, 1, 1) * (probe->get_blend_distance() + 0.25f);
		if (overrides == 0 || core.x <= 0.0f || core.y <= 0.0f || core.z <= 0.0f || s.probe_count == PhysXGIEffect::Snapshot::MAX_PROBES) {
			continue;
		}
		Transform3D xf = probe->get_global_transform();
		xf.basis.orthonormalize();
		const int i = s.probe_count++;
		s.probe_to_local[i] = xf.affine_inverse();
		s.probe_core[i] = core;
		s.probe_overrides[i] = overrides;
	}
	s.light_changed = fresh_frames > 0;
	PhysXGIEffect::set_settings(get_instance_id(), s);
	// SDFGI is only the switch that makes surfaces read the GI buffer: on, at its cheapest (one cascade of 8 m cells,
	// so a moving camera rarely makes it re-voxelize).
	Ref<Environment> env = camera && camera->get_environment().is_valid() ? camera->get_environment() : get_world_3d()->get_environment();
	if (environment.is_valid() && environment != env) {
		environment->set_sdfgi_enabled(environment->is_sdfgi_enabled());
	}
	environment = env;
	if (env.is_valid()) {
		RenderingServer::get_singleton()->environment_set_sdfgi(env->get_rid(), true, 1, 8.0f, RSE::ENV_SDFGI_Y_SCALE_100_PERCENT, false, 0.0f, false, 1.0f, 1.1f, 1.1f);
	}

	// Anything that changes the light (shapes moved, lights, settings) means converging again.
	uint32_t h = hash_murmur3_one_64(s.primitives_version);
	h = hash_murmur3_one_float(s.intensity, h);
	h = hash_murmur3_one_float(s.max_distance, h);
	h = hash_murmur3_one_float(s.world_distance, h);
	h = hash_murmur3_one_float(s.reflection_distance, h);
	h = hash_murmur3_one_32(s.filtered_reflections, h);
	h = hash_murmur3_one_float(s.reflection_softness, h);
	h = hash_murmur3_one_float(s.reflection_resolution_scale, h);
	h = hash_murmur3_one_32(s.debug | (s.reflection_mode << 4) | (s.multi_bounce ? 256 : 0) | (s.rays_per_texel << 10), h);
	h = hash_murmur3_one_float(s.sun_direction.x + s.sun_direction.y * 3.1f + s.sun_direction.z * 7.3f + s.sun_color.get_luminance(), h);
	for (int i = 0; i < s.spot_count; i++) {
		h = hash_murmur3_one_float(s.spot_position[i].x + s.spot_direction[i].y * 3.1f + s.spot_direction[i].z * 7.3f + s.spot_color[i].get_luminance(), h);
	}
	for (int i = 0; i < s.omni_count; i++) {
		h = hash_murmur3_one_float(s.omni_position[i].x + s.omni_position[i].y * 3.1f + s.omni_position[i].z * 7.3f + s.omni_color[i].get_luminance(), h);
	}
	if (h != last_hash) {
		last_hash = h;
		PhysXGIEffect::unsettle();
	}
	// The light sources themselves (which and how bright): a lamp switched off or an emitter gone shouldn't glow on in
	// the radiance cache for seconds. Published from the next frame, for 60 frames.
	uint32_t sources = hash_murmur3_one_32(s.omni_count | (s.spot_count << 8) | (s.emissive_count << 16));
	sources = hash_murmur3_one_float(s.sun_color.r + s.sun_color.g * 3.1f + s.sun_color.b * 7.3f, sources);
	for (int i = 0; i < s.omni_count; i++) {
		sources = hash_murmur3_one_float(s.omni_color[i].r + s.omni_color[i].g * 3.1f + s.omni_color[i].b * 7.3f, sources);
	}
	for (int i = 0; i < s.spot_count; i++) {
		sources = hash_murmur3_one_float(s.spot_color[i].r + s.spot_color[i].g * 3.1f + s.spot_color[i].b * 7.3f, sources);
	}
	for (int i = 0; i < s.emissive_count; i++) {
		sources = hash_murmur3_one_float(s.emissive_color[i].r + s.emissive_color[i].g * 3.1f + s.emissive_color[i].b * 7.3f, sources);
	}
	if (sources != last_sources_hash) {
		last_sources_hash = sources;
		fresh_frames = 60;
	} else if (fresh_frames > 0) {
		fresh_frames--;
	}
	_stage(STAGE_PUBLISH, publish_since);
}

void PXTGI3D::_notification(int p_what) {
	if (RenderingServer::get_singleton()->get_current_rendering_method() != "forward_plus") {
		return;
	}
	switch (p_what) {
		case NOTIFICATION_ENTER_WORLD: {
			if (is_visible_in_tree()) {
				_activate();
			}
		} break;
		case NOTIFICATION_EXIT_WORLD: {
			_deactivate();
		} break;
		case NOTIFICATION_VISIBILITY_CHANGED: {
			// Hidden, the node steps aside entirely, as if it had left the world: Godot's own GI is back.
			if (is_visible_in_tree() && is_inside_world()) {
				_activate();
			} else {
				_deactivate();
			}
		} break;
		case NOTIFICATION_INTERNAL_PROCESS: {
			if (!attached_world.is_valid()) {
				return;
			}
			_publish();
			PhysXGIEffect::install(get_world_3d().ptr(), get_viewport() ? get_viewport()->get_camera_3d() : nullptr);
#ifdef TOOLS_ENABLED
			// The editor draws only when something changes: keep it drawing until the GI has converged.
			if (Engine::get_singleton()->is_editor_hint() && PhysXGIEffect::settling()) {
				Main::force_redraw();
			}
#endif
		} break;
	}
}

PackedStringArray PXTGI3D::get_configuration_warnings() const {
	PackedStringArray warnings = Node3D::get_configuration_warnings();
	if (RenderingServer::get_singleton()->get_current_rendering_method() != "forward_plus") {
		warnings.push_back(RTR("PXTGI3D only works with the Forward+ renderer: with Mobile or Compatibility it does nothing."));
	}
	return warnings;
}

void PXTGI3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("notify_body_changed", "body"), &PXTGI3D::notify_body_changed);
	ClassDB::bind_method(D_METHOD("update_emitter_group", "bodies", "stations", "values", "time"), &PXTGI3D::update_emitter_group);
	ClassDB::bind_method(D_METHOD("get_stats", "reset"), &PXTGI3D::get_stats, DEFVAL(true));
	ClassDB::bind_method(D_METHOD("set_intensity", "intensity"), &PXTGI3D::set_intensity);
	ClassDB::bind_method(D_METHOD("get_intensity"), &PXTGI3D::get_intensity);
	ClassDB::bind_method(D_METHOD("set_max_distance", "distance"), &PXTGI3D::set_max_distance);
	ClassDB::bind_method(D_METHOD("get_max_distance"), &PXTGI3D::get_max_distance);
	ClassDB::bind_method(D_METHOD("set_world_ray_distance", "distance"), &PXTGI3D::set_world_ray_distance);
	ClassDB::bind_method(D_METHOD("get_world_ray_distance"), &PXTGI3D::get_world_ray_distance);
	ClassDB::bind_method(D_METHOD("set_reflection_distance", "distance"), &PXTGI3D::set_reflection_distance);
	ClassDB::bind_method(D_METHOD("get_reflection_distance"), &PXTGI3D::get_reflection_distance);
	ClassDB::bind_method(D_METHOD("set_filtered_reflections", "enabled"), &PXTGI3D::set_filtered_reflections);
	ClassDB::bind_method(D_METHOD("get_filtered_reflections"), &PXTGI3D::get_filtered_reflections);
	ClassDB::bind_method(D_METHOD("set_reflection_softness", "softness"), &PXTGI3D::set_reflection_softness);
	ClassDB::bind_method(D_METHOD("get_reflection_softness"), &PXTGI3D::get_reflection_softness);
	ClassDB::bind_method(D_METHOD("set_reflection_resolution_scale", "scale"), &PXTGI3D::set_reflection_resolution_scale);
	ClassDB::bind_method(D_METHOD("get_reflection_resolution_scale"), &PXTGI3D::get_reflection_resolution_scale);
	ClassDB::bind_method(D_METHOD("set_rays_per_texel", "rays"), &PXTGI3D::set_rays_per_texel);
	ClassDB::bind_method(D_METHOD("get_rays_per_texel"), &PXTGI3D::get_rays_per_texel);
	ClassDB::bind_method(D_METHOD("set_multi_bounce", "enabled"), &PXTGI3D::set_multi_bounce);
	ClassDB::bind_method(D_METHOD("get_multi_bounce"), &PXTGI3D::get_multi_bounce);
	ClassDB::bind_method(D_METHOD("set_reflection_mode", "mode"), &PXTGI3D::set_reflection_mode);
	ClassDB::bind_method(D_METHOD("get_reflection_mode"), &PXTGI3D::get_reflection_mode);
	ClassDB::bind_method(D_METHOD("set_reflection_roughness_cutoff", "roughness"), &PXTGI3D::set_reflection_roughness_cutoff);
	ClassDB::bind_method(D_METHOD("get_reflection_roughness_cutoff"), &PXTGI3D::get_reflection_roughness_cutoff);
	ClassDB::bind_method(D_METHOD("set_debug", "mode"), &PXTGI3D::set_debug);
	ClassDB::bind_method(D_METHOD("get_debug"), &PXTGI3D::get_debug);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "intensity", PROPERTY_HINT_RANGE, "0,4,0.01,or_greater"), "set_intensity", "get_intensity");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "max_distance", PROPERTY_HINT_RANGE, "0.1,20,0.1,or_greater"), "set_max_distance", "get_max_distance");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "world_ray_distance", PROPERTY_HINT_RANGE, "1,100,0.5,or_greater,suffix:m"), "set_world_ray_distance", "get_world_ray_distance");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "reflection_distance", PROPERTY_HINT_RANGE, "1,300,0.5,or_greater,suffix:m"), "set_reflection_distance", "get_reflection_distance");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "filtered_reflections"), "set_filtered_reflections", "get_filtered_reflections");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "reflection_softness", PROPERTY_HINT_RANGE, "1,8,0.1"), "set_reflection_softness", "get_reflection_softness");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "reflection_resolution_scale", PROPERTY_HINT_RANGE, "0.5,1,0.5"), "set_reflection_resolution_scale", "get_reflection_resolution_scale");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "rays_per_texel", PROPERTY_HINT_RANGE, "1,4,1"), "set_rays_per_texel", "get_rays_per_texel");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "multi_bounce"), "set_multi_bounce", "get_multi_bounce");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "reflection_mode", PROPERTY_HINT_ENUM, "Godot,Screen Space,Traced,Traced + Probes"), "set_reflection_mode", "get_reflection_mode");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "reflection_roughness_cutoff", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_reflection_roughness_cutoff", "get_reflection_roughness_cutoff");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "debug", PROPERTY_HINT_ENUM, "Off,Last Frame,Round Trip,World Only,World Hits,World Position,Screen Only"), "set_debug", "get_debug");
}
