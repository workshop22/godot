/**************************************************************************/
/*  sdfgi_scroll.h                                                        */
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

#ifndef SDFGI_SCROLL_H
#define SDFGI_SCROLL_H

#include "core/math/vector3i.h"

namespace RendererRD::SDFGIScroll {

inline bool redraw_all(const Vector3i &p_dirty, uint32_t p_size) {
	uint32_t axes = 0;
	uint32_t safe_volume = 1;
	for (int i = 0; i < 3; i++) {
		const uint32_t moved = Math::abs(p_dirty[i]);
		if (moved >= p_size) {
			return true;
		}
		axes += moved != 0;
		safe_volume *= p_size - moved;
	}
	// One incoming slab always takes one voxel draw, just like a full rebuild.
	// Retain the overlap instead of drawing its geometry again. For several slabs,
	// keep the existing volume heuristic which trades repeated draws for one.
	return axes > 1 && p_size * p_size * p_size - safe_volume > safe_volume / 2;
}

} // namespace RendererRD::SDFGIScroll

#endif // SDFGI_SCROLL_H
