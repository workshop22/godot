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
