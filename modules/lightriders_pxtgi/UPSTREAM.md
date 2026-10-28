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
