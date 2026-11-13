# PXTGI rendering trial

Rendering sources from Wild Ox Studios' MIT-licensed `godot_physx` module:
https://github.com/uno1982/godot/tree/e2b3875bf68975aaac9154b55d05b14685c31857/modules/godot_physx

The source file headers and module LICENSE are retained. This module contains the
PXTGI node, GPU renderer and compositor attachment helper. It does not link the
PhysX SDK or supply a physics backend. The upstream non-PhysX scene-proxy update
path is retained; the optional awake-PhysX-body shortcut is removed. The render
thread call helper carries its callable locally instead of including the PhysX
server header. Engine base: Lightriders' pinned Godot 4.7.2 patch series.

This is a visual comparison candidate. The normal project does not instantiate
PXTGI; the game trial opts in with `--pxtgi` on `route.first`.

Local scene-integration changes expose reflection reach independently of GI reach,
allow a proxy to opt out of box-light sampling with `pxtgi_sample_emission`, and
retain traced emission for sources outside the direct-light selection. Reflections
run at full viewport resolution. The optional `filtered_reflections` mode uses
coherent mirror rays and the existing roughness resolve instead of stochastic GGX
directions; this is an artistic approximation for the wet-road study. Camera cuts
invalidate previous-colour reuse. These changes are experimental, not upstream
quality or performance claims.

The standalone module also carries the upstream lens dirt, lens flare/settings and
light shafts nodes/effects from the same source revision. They use the local render
thread callable and compositor helper, without an SDK dependency. Flare ghost scales
use aligned vec4 records on both sides of the GPU buffer; explicit UV rejection
replaces border-sampler reads. Lens extraction bounds non-finite HDR inputs. A local
`reflection_softness` setting adds a separable depth/normal-aware reflection-only blur
without feeding blurred output back into temporal history. These effects remain
opt-in through the game experiment.

Local changes of 2026-11-13, for the game's ray traced Lighting:

- `reflection_jitter`: a reflection ray starts at a point drawn in its texel (blue noise, anew each frame or
  fixed) instead of the texel's middle, on the pixel's own surface: a reflected line thinner than a texel is
  met as often as it covers it instead of as rows of beads.
- `PXTGI3D.hardware_fallback()` (static): hardware ray queries were asked for and the renderer fell back to the
  software walk for good. The game reads it to leave tracing rather than run the walk unannounced.
- Main-thread cost in a streamed city: `update_terrain_visibility` decides every record each frame from a copy
  whose inputs it reads again an eighth a frame; the nearest lamps are chosen from a shortlist between looks at
  every lamp (each eighth publication); emissives are scored every eighth and the light sources hashed every
  sixteenth publication; the verification slices are 24 records. `get_stats` adds the `items`, `lights`,
  `settings` and `sources` clocks and the render thread's time in the frame callback (`render_thread`).

## Upstream audit

Port basis `e2b3875` (2026-10-28). Audited through `7d46568e4e` (2026-11-09): the PXTGI
sources (`modules/godot_physx/nodes/pxtgi_3d.*`, `physx_gi_effect.*`) last changed upstream at
`fe01d53` (2026-11-04). Our shaders and scene bookkeeping have diverged, so each upstream change
is ported by hand, one commit each, with its local differences in the commit message.

| Upstream | Change | Here |
| --- | --- | --- |
| `de31a18` | Less sparkle and noise on glossy reflections (brightness-weighted history and gather, a-trous filter pass, cache light for glossy off-screen hits) | Ported. The filter serves stochastic glossy rays (`filtered_reflections` off); coherent mirror rays keep their on-screen softening. |
| `0b92bba` | Emissives light the scene evenly: 64 chosen by contribution, shadow rays stop at the emitter's box, the cache samples emissives, light-change watch on the scene | Ported. One-sided emission comes from an authored `pxtgi_emission_facing`, not a glowing mesh's normals; only `pxtgi_sample_emission` emitters count. |
| `27644cf`, `fe01d53` | Emissive noise hashed by texel; `emissive_candidates` (1 to 16, default 6) | Ported. |
| `9587db4` | Scenes with triangle meshes walk the shapes and meshes in one loop (software trace speed) | Not ported. A cost-phase candidate for the software walk: our static and moving trees changed this code. |
| `dbd0db9`, `567cc68` | An emissive shines from its glowing mesh surface; a glowing mesh with collision no longer shadows by its bounds | Not ported: our emitters are authored boxes and our proxies are explicit. |
| `8bad468`, `0d7c8d4`, `c6db02e`, `7e4845f` | Reflection probe warning, editor light deletion, hidden collision shapes, upstream GCC warnings | Not ported: editor or upstream-file concerns that do not reach the game's proxies. |
