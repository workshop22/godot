/**************************************************************************/
/*  pxtgi_3d.h                                                            */
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

#pragma once

#include "physx_gi_effect.h"

#include "core/templates/hash_map.h"
#include "core/templates/hash_set.h"
#include "core/templates/local_vector.h"
#include "scene/3d/node_3d.h"
#include "scene/resources/3d/concave_polygon_shape_3d.h"
#include "scene/resources/3d/shape_3d.h"
#include "scene/resources/environment.h"
#include "scene/resources/texture.h"

// PXTGI (experimental): the module's own per-pixel global illumination, traced on screen and then against the physics
// shapes, delivered through Forward+'s stock GI path. With SDFGI enabled, every surface shader reads Godot's per-pixel
// GI buffer and multiplies it by its albedo. PhysXGIEffect writes that buffer before the opaque pass; while a visible
// PXTGI3D is in the tree, Godot's own GI step (which would overwrite it) runs a do-nothing shader with the same binding
// layout instead, and SDFGI is kept to its cheapest settings (it is only the switch that makes surfaces read the
// buffer). Hidden or gone, Godot's pipelines and the environment's own SDFGI settings are restored.
class MeshInstance3D;
class CollisionShape3D;

class PXTGI3D : public Node3D {
	GDCLASS(PXTGI3D, Node3D);

	static int users;

	float intensity = 1.0f;
	float max_distance = 8.0f;
	float world_ray_distance = 30.0f;
	float reflection_distance = 60.0f;
	bool filtered_reflections = false;
	float reflection_softness = 1.0f;
	float reflection_resolution_scale = 1.0f;
	int debug = 0;
	bool multi_bounce = true;
	int rays_per_texel = 4;
	bool blue_noise = false;
	bool gi_history_clamp = false;
	bool filtered_hits = false;
	int reflection_mode = 2;
	float reflection_cutoff = 0.5f;
	ObjectID attached_world;

	// The live collision shapes, each with its body's albedo and emission;
	// and emissive meshes outside any body, which stand in as their bounding box (mesh_proxy).
	struct ShapeEntry {
		// External reflection surfaces share the scene tree, but own a collision-free
		// history identity and immutable geometry without creating physics nodes.
		Ref<RefCounted> item_identity;
		Ref<ConcavePolygonShape3D> item_geometry;
		Transform3D item_pose;
		int item_material = -1;
		ObjectID shape_node;
		ObjectID body_node;
		CollisionShape3D *collision_node = nullptr; // valid only while this ID is tracked in the tree
		LocalVector<int> *body_indices = nullptr; // stable HashMap values, reused until this owner retires
		LocalVector<int> *geometry_indices = nullptr;
		// Positions in the small owner lists; swap removal never searches retained city membership.
		int body_slot = -1;
		int geometry_slot = -1;
		int dynamic_slot = -1;
		int proxy_slot = -1;
		int emissive_slot = -1;
		bool manual_updates = false;
		uint64_t motion_generation = 0; // pooled proxy identity, independent of geometry revisions
		Color albedo;
		Color emission;
		bool mesh_proxy = false;
		bool fixed = false; // a static body's shape (not animatable): a reflection probe's capture of it stays true
		ObjectID shape_resource; // a collision shape's Shape3D when found: another one means finding the shapes again
		bool sample_emission = true; // its body lets the trace sample it as a box light (pxtgi_sample_emission)
		// In the moving part of the tree (rebuilt whenever one of its shapes changes), else the static part (built when
		// the static shapes change): bodies that can move, emissive meshes, and a static shape found to have moved.
		bool moving = false;
		bool driven = false; // its glow is set by update_emitter_group, not read from its look
		// Hulls and meshes: their type (4, 5), where their geometry starts in the geometry buffer, their local bounds.
		int geometry_type = 0;
		uint64_t geometry_key = 0;
		int geometry_a = 0;
		int geometry_b = 0;
		int geometry_attributes = 0;
		AABB local_bounds;
		// Where the shape sits on its body and what it is (type, extents), read from its node when last checked: the
		// shapes of a body PhysX moved are placed from the body's own pose, without the scene tree.
		bool posed = false;
		Transform3D in_body;
		float type = 0.0f;
		Vector3 extents;
	};
	// Hull planes, or a mesh's tree and triangles, rebuilt when the shape resource changes.
	// Inactive resources retain CPU blocks by weak ID; only allocated ranges reach the renderer.
	struct GeometryBlock {
		int type = 0;
		uint64_t key = 0;
		int offset = -1; // allocated range in the packed geometry buffer, in vec4s
		int words = 0;
		Vector<float> planes; // hull: normal, offset
		Vector<float> nodes; // mesh: its tree (8 floats a node; indices relative to the mesh)
		Vector<float> triangles; // mesh: three vec4 corners; lattice vertex brightness in w, otherwise padding
		Vector<float> attributes; // item: two vec4s per triangle, in the same SAH order
		AABB bounds;
		// COW input references distinguish geometry edits from debug-style changed
		// signals without copying source data or keeping the Shape3D alive.
		Vector<Vector3> source_vertices;
		Vector<real_t> source_heights;
		int source_width = 0;
		int source_depth = 0;
	};
	HashMap<ObjectID, GeometryBlock> geometry_blocks;
	HashSet<ObjectID> geometry_dirty;
	HashMap<ObjectID, LocalVector<int>> geometry_entries;
	LocalVector<Pair<int, int>> geometry_ranges; // live GPU spans, sorted by offset, in vec4s
	int geometry_floats = 0; // logical size; the immutable blocks are packed only on the GPU
	Vector<PhysXGIEffect::GeometryUpload> published_geometry;
	Vector<PhysXGIEffect::GeometryUpload> geometry_spares[2];
	uint32_t geometry_spare = 0;
	uint64_t published_geometry_version = UINT64_MAX;
	uint64_t geometry_version = 0;

	const GeometryBlock *_geometry_for(const Ref<Shape3D> &p_shape, const PackedVector2Array *p_uv = nullptr, const PackedFloat32Array *p_vertex_red = nullptr);
	void _geometry_changed(ObjectID p_shape);
	void _layout_geometry(const LocalVector<ObjectID> *p_changed = nullptr);
	void _refresh_geometry();
	LocalVector<ShapeEntry> shapes;
	HashMap<ObjectID, HashMap<int, ObjectID>> reflection_item_ids;
	HashMap<String, Ref<ConcavePolygonShape3D>> reflection_item_geometry;
	PackedFloat32Array reflection_item_materials;
	TypedArray<Texture2D> reflection_item_masks;
	String reflection_item_shader;
	float reflection_item_hero = 1.0f;
	uint64_t reflection_source_policy = 0;
	bool reflection_items_driven = false;
	LocalVector<ObjectID> suns; // every DirectionalLight3D: the brightest one drawn on surfaces is the sun, chosen each frame
	Ref<Environment> environment; // the one whose SDFGI is the switch, given its own settings back on leaving
	uint32_t last_hash = 0; // what was published last: a change means the GI has to converge again
	// What gives off light (which lights and emitters, how bright; not where, as moving lamps change it every frame):
	// a change lets the radiance cache follow quickly for a while.
	uint32_t last_sources_hash = 0;
	int fresh_frames = 0;
	LocalVector<ObjectID> omnis;
	LocalVector<ObjectID> spots;
	LocalVector<ObjectID> probes; // ReflectionProbes: where they override the GI, it isn't traced

	// The nodes the GI keeps track of (lights, reflection probes, the collision shapes of bodies, meshes outside any
	// body): found by one walk of the scene when PXTGI starts, then kept up to date from the scene tree's node_added and
	// node_removed. Only pending membership deltas are classified after initial discovery.
	HashSet<ObjectID> tracked;
	ObjectID tracked_root; // the node they were found under (the edited scene in the editor); none: walk again
	HashSet<ObjectID> pending_membership; // only nodes whose classification or lifetime changed
	bool membership_immediate = false; // explicit notifications are applied at the next publication
	// Meshes outside any body that came since the last update: only one that glows means finding the shapes again
	// (effects add and drop plain meshes all the time).
	LocalVector<ObjectID> pending_meshes;
	int frames_since_rescan = 0;
	// Each tracked shape's and mesh's look (albedo, emission), with a bounded slice checked
	// each frame for unnotified authoring edits. Explicit notifications apply at the next publication.
	HashMap<ObjectID, Pair<Color, Color>> looks;
	LocalVector<ObjectID> look_order;
	HashMap<ObjectID, int> look_slots;
	uint32_t look_cursor = 0;
	HashMap<ObjectID, int> entry_of; // a shape node's index in shapes
	uint32_t verify_cursor = 0;

	void _track_all(Node *p_root);
	void _node_added(Node *p_node);
	void _node_removed(Node *p_node);
	Node *_scan_root() const;
	Pair<Color, Color> _look_of(Node *p_node) const;
	void _restyle_slice(LocalVector<int> &r_changed);

	// Each shape's GPU record and world bounds (in shape order), kept between frames: only what moved is rewritten,
	// and the tree is refit rather than rebuilt.
	LocalVector<float> entry_floats;
	LocalVector<AABB> entry_bounds;
	LocalVector<uint8_t> entry_valid;
	HashMap<ObjectID, LocalVector<int>> body_entries; // a physics body's shapes, by its node
	// The poses of the PhysX bodies awake in the physics steps since the last frame, by their nodes: read during
	// physics process, where the space may be read even with physics on its own thread.
	HashMap<ObjectID, Transform3D> moved_poses;
	LocalVector<int> dynamic_entries; // moving presentation bodies, excluding explicitly notified ribbons
	LocalVector<int> proxy_entries; // emissive meshes outside any body: checked every frame
	LocalVector<int> emissive_entries; // shapes that give off light: sampled directly
	LocalVector<int> moving_entries; // the shapes in the moving part of the tree
	// The tree's records in leaf order and its nodes: the static part first, then the moving part, joined under one
	// root node when both have shapes. Rebuilding the moving part rewrites only what follows the static part.
	LocalVector<float> ordered_floats;
	LocalVector<float> tree_nodes;
	LocalVector<int> leaf_order; // leaf slot -> shape
	LocalVector<int> entry_slot; // shape -> leaf slot (-1: not in the tree)
	uint32_t static_node_floats = 0;
	uint32_t static_record_floats = 0;
	uint32_t static_leaves = 0;
	bool static_rooted = false; // the static part was built under a shared root
	uint64_t static_version = 0; // the static part's layout: bumped whenever it is rebuilt (hardware trace spike)
	int moving_root_node = -1; // the moving part's root node, -1 with no moving part
	int trace_mode = 0; // 0 software, 1 world rays find nothing (their share of the cost), 2 hardware, 3 hardware where it is safe (NVIDIA, no earlier session lost with it)
	bool hardware_layout = false; // hardware requested and the shared effect has not permanently fallen back
	bool static_dirty = true;
	bool moving_dirty = true;
	bool radiance_dirty = false; // only glow changed: the records in place, not the tree
	AABB opaque_world_bounds;
	bool have_opaque_world_bounds = false;
	Vector<float> published_primitives;
	Vector<Pair<uint64_t, uint64_t>> published_ids;
	Vector<float> published_nodes;
	int published_count = 0;
	// Hardware tracing: each leaf's mesh (its Shape3D's id; 0 for a box), and every mesh's triangles by that id.
	Vector<uint64_t> published_keys;
	// The current publication and two rotating spares retain at most three buffers
	// per field. A renderer still holding a spare is protected by Vector's COW.
	Vector<float> primitive_spares[2];
	Vector<Pair<uint64_t, uint64_t>> id_spares[2];
	Vector<float> node_spares[2];
	Vector<uint64_t> key_spares[2];
	uint32_t publication_spare = 0;
	Vector<uint64_t> published_block_keys;
	Vector<Vector<float>> published_block_triangles;
	Vector<uint64_t> block_key_spares[2];
	Vector<Vector<float>> block_triangle_spares[2];
	uint32_t block_spare = 0;
	uint64_t published_blocks_version = UINT64_MAX;
	uint64_t primitives_version = 0;

	void _rescan();
	bool _sync_member(ObjectID p_id);
	bool _membership_changed(Node *p_node, const ShapeEntry *p_entry) const;
	void _observe(ObjectID p_id);
	void _unobserve(ObjectID p_id);
	void _register_entry(ShapeEntry &r_entry, int p_index);
	void _unregister_entry(const ShapeEntry &p_entry);
	void _remove_entry(int p_index);
	bool _shape_frame(int p_index, Transform3D &r_to_world, float &r_type, Vector3 &r_extents) const;
	void _pack_entry(int p_index, const Transform3D &p_to_world, float p_type, const Vector3 &p_extents, float *r_out, AABB &r_bounds) const;
	bool _refresh_entry(int p_index, float *r_out, AABB &r_bounds);
	void _record_frame(int p_index, Transform3D &r_to_world, AABB &r_local) const;
	void _set_glow(int p_index, const Color &p_glow);
	void _set_body_look(Node3D *p_body, const Color &p_albedo, const Color &p_emission, const Pair<Color, Color> &p_linear);
	void _append_mesh(MeshInstance3D *p_mesh, const Pair<Color, Color> &p_look);
	void _build_tree();
	void _update_primitives(PhysXGIEffect::Snapshot &r_snapshot);
	void _publish_geometry(PhysXGIEffect::Snapshot &r_snapshot);

	static void _install();
	static void _uninstall();

	// Main-thread cost of keeping the GI's scene up to date, by stage (get_stats): microseconds in all, the worst
	// call, and how many calls.
	enum Stage {
		STAGE_PUBLISH, // the whole frame's update
		STAGE_EMITTERS, // update_emitter_group
		STAGE_RESCAN,
		STAGE_RECORDS, // shapes' records read again from their nodes
		STAGE_STATIC_TREE, // the static part of the tree built
		STAGE_MOVING_TREE, // the moving part rebuilt
		STAGE_SELECT, // the directly sampled emitters chosen
		STAGE_COPY, // records and tree handed to the renderer
		STAGE_SCAN_MEMBERS,
		STAGE_SCAN_LAYOUT,
		STAGE_SCAN_KEEP,
		STAGE_SCAN_PREPARE,
		STAGE_SCAN_WALK,
		STAGE_SCAN_PRUNE,
		STAGE_SCAN_CHECK,
		STAGE_SCAN_LOOK,
		STAGE_SCAN_GEOMETRY,
		STAGE_SCAN_REGISTER,
		STAGE_GEOMETRY_FIRST,
		STAGE_GEOMETRY_RETIRED,
		STAGE_GEOMETRY_EDITED,
		STAGE_HARDWARE_PUBLISH,
		STAGE_GEOMETRY_PUBLISH,
		STAGE_MAX,
	};
	struct StageTime {
		uint64_t total = 0;
		uint64_t worst = 0;
		uint32_t calls = 0;
	};
	StageTime stage_times[STAGE_MAX];
	uint64_t records_changed = 0;
	uint64_t scan_retained = 0;
	uint64_t scan_fresh = 0;
	String last_change; // what last made the shapes need finding again
	PackedStringArray rescan_causes; // since the stats were last read
	void _stage(Stage p_stage, uint64_t p_since);

public:
	static void free_shared();
	void set_reflection_items_driven(bool p_enabled) { reflection_items_driven = p_enabled; }
	void set_reflection_items(const Array &p_records, const PackedFloat32Array &p_materials, const TypedArray<Texture2D> &p_masks, const String &p_shader_source, float p_hero, int64_t p_source_policy);
	void prepare_geometry(const Ref<Shape3D> &p_shape);
	void notify_body_changed(Node3D *p_body);
	void notify_bodies_changed(const Array &p_bodies);
	// False is for controlled pools that notify every body after re-enabling it.
	void disable_bodies(const Array &p_bodies, bool p_check_membership = true);
	void notify_subtree_retired(Node *p_root);
	void set_body_look(Node3D *p_body, const Color &p_albedo, const Color &p_emission);
	void set_bodies_look(const Array &p_bodies, const Color &p_albedo, const Color &p_emission);
	void set_body_group_looks(const Dictionary &p_groups, const Array &p_keys, const Color &p_albedo, const Array &p_emissions);
	void update_emitter_group(const PackedInt64Array &p_bodies, const PackedFloat32Array &p_stations, const Dictionary &p_values, double p_time);
	Dictionary get_stats(bool p_reset = true);

private:
	bool active = false; // in the world and visible: delivering the GI
	void _activate();
	void _deactivate();
	void _attach();
	void _detach();
	void _publish();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	PackedStringArray get_configuration_warnings() const override;

	void set_intensity(float p_value) { intensity = MAX(p_value, 0.0f); }
	float get_intensity() const { return intensity; }
	void set_max_distance(float p_value) { max_distance = MAX(p_value, 0.1f); }
	float get_max_distance() const { return max_distance; }
	void set_world_ray_distance(float p_value) { world_ray_distance = MAX(p_value, 0.1f); }
	float get_world_ray_distance() const { return world_ray_distance; }
	void set_reflection_distance(float p_value) { reflection_distance = MAX(p_value, 0.1f); }
	float get_reflection_distance() const { return reflection_distance; }
	void set_filtered_reflections(bool p_value) { filtered_reflections = p_value; }
	bool get_filtered_reflections() const { return filtered_reflections; }
	void set_reflection_softness(float p_value) { reflection_softness = CLAMP(p_value, 1.0f, 8.0f); }
	float get_reflection_softness() const { return reflection_softness; }
	void set_reflection_resolution_scale(float p_value) { reflection_resolution_scale = CLAMP(p_value, 0.5f, 1.0f); }
	float get_reflection_resolution_scale() const { return reflection_resolution_scale; }
	void set_rays_per_texel(int p_value) { rays_per_texel = CLAMP(p_value, 1, 4); }
	int get_rays_per_texel() const { return rays_per_texel; }
	void set_blue_noise(bool p_value) { blue_noise = p_value; }
	bool get_blue_noise() const { return blue_noise; }
	void set_gi_history_clamp(bool p_value) { gi_history_clamp = p_value; }
	bool get_gi_history_clamp() const { return gi_history_clamp; }
	void set_filtered_hits(bool p_value) { filtered_hits = p_value; }
	bool get_filtered_hits() const { return filtered_hits; }
	void set_multi_bounce(bool p_value) { multi_bounce = p_value; }
	bool get_multi_bounce() const { return multi_bounce; }
	void set_reflection_mode(int p_value) { reflection_mode = CLAMP(p_value, 0, 3); }
	int get_reflection_mode() const { return reflection_mode; }
	void set_reflection_roughness_cutoff(float p_value) { reflection_cutoff = CLAMP(p_value, 0.0f, 1.0f); }
	float get_reflection_roughness_cutoff() const { return reflection_cutoff; }
	void set_trace_mode(int p_mode) { trace_mode = CLAMP(p_mode, 0, 3); }
	int get_trace_mode() const { return trace_mode; }
	void set_debug(int p_value) { debug = p_value; }
	int get_debug() const { return debug; }
};
