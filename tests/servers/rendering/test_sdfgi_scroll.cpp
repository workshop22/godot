/**************************************************************************/
/*  test_sdfgi_scroll.cpp                                                 */
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

#include "tests/test_macros.h"

TEST_FORCE_LINK(test_sdfgi_scroll)

#include "servers/rendering/renderer_rd/environment/sdfgi_scroll.h"

namespace TestSDFGIScroll {

TEST_CASE("[Rendering][SDFGI] A retained single-axis overlap stays one incoming slab") {
	using RendererRD::SDFGIScroll::redraw_all;
	CHECK_FALSE(redraw_all(Vector3i(), 128));
	for (int axis = 0; axis < 3; axis++) {
		for (int distance : {8, 40, 48, 56, 64, 104, 120, 127}) {
			Vector3i dirty;
			dirty[axis] = distance;
			CHECK_FALSE(redraw_all(dirty, 128));
			dirty[axis] = -distance;
			CHECK_FALSE(redraw_all(dirty, 128));
		}
		for (int distance : {128, 136, 1024}) {
			Vector3i dirty;
			dirty[axis] = distance;
			CHECK(redraw_all(dirty, 128));
			dirty[axis] = -distance;
			CHECK(redraw_all(dirty, 128));
		}
	}
}

TEST_CASE("[Rendering][SDFGI] Multi-axis moves keep the existing rebuild threshold") {
	using RendererRD::SDFGIScroll::redraw_all;
	CHECK_FALSE(redraw_all(Vector3i(8, -8, 0), 128));
	CHECK_FALSE(redraw_all(Vector3i(8, -8, 8), 128));
	CHECK(redraw_all(Vector3i(48, 8, 0), 128));
	CHECK(redraw_all(Vector3i(-48, 0, -8), 128));
	CHECK(redraw_all(Vector3i(32, 32, 32), 128));
}

} // namespace TestSDFGIScroll
