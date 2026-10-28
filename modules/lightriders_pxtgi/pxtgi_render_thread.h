/**************************************************************************/
/*  godot_physx_render_thread.h                                           */
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

#include "core/templates/hashfuncs.h"
#include "core/variant/callable.h"

template <typename F>
class PXTGIQueuedCall : public CallableCustom {
	mutable F func;

	static bool _compare_equal(const CallableCustom *p_a, const CallableCustom *p_b) { return p_a == p_b; }
	static bool _compare_less(const CallableCustom *p_a, const CallableCustom *p_b) { return p_a < p_b; }

public:
	virtual uint32_t hash() const override { return hash_murmur3_one_64((uint64_t)this); }
	virtual String get_as_text() const override { return "PXTGIQueuedCall"; }
	virtual CompareEqualFunc get_compare_equal_func() const override { return _compare_equal; }
	virtual CompareLessFunc get_compare_less_func() const override { return _compare_less; }
	virtual bool is_valid() const override { return true; }
	virtual ObjectID get_object() const override { return ObjectID(); }
	virtual void call(const Variant **p_arguments, int p_argcount, Variant &r_return_value, Callable::CallError &r_call_error) const override {
		func();
		r_call_error.error = Callable::CallError::CALL_OK;
	}

	PXTGIQueuedCall(const F &p_func) :
			func(p_func) {}
};

#include "servers/rendering/rendering_server.h"

// Runs p_func on the render thread and waits for it. A CompositorEffect is
// released on whichever thread lets go of it last (the main thread when a
// node or Compositor drops it), but the RenderingDevice frees its resources
// only on the render thread. Single-threaded rendering runs it right here.
template <typename F>
void pxtgi_run_on_render_thread(const F &p_func) {
	RenderingServer *rs = RenderingServer::get_singleton();
	if (rs == nullptr || rs->is_on_render_thread()) {
		p_func();
		return;
	}
	rs->call_on_render_thread(Callable(memnew(PXTGIQueuedCall<F>(p_func))));
	rs->sync();
}
