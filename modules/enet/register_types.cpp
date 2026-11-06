/**************************************************************************/
/*  register_types.cpp                                                    */
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

#include "register_types.h"

#include "enet_connection.h"
#include "enet_multiplayer_peer.h"
#include "enet_packet_peer.h"

#include "core/config/project_settings.h"
#include "core/error/error_macros.h"
#include "core/object/class_db.h"

static bool enet_ok = false;

void initialize_enet_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	if (enet_initialize() != 0) {
		ERR_PRINT("ENet initialization failure");
	} else {
		enet_ok = true;
	}

	// OS socket buffers for every ENet host (bytes, 0 = OS default). A client that stalls longer
	// than its receive buffer holds loses datagrams; Windows' UDP default is 64 KiB.
	GLOBAL_DEF(PropertyInfo(Variant::INT, "network/enet/receive_buffer_size", PROPERTY_HINT_RANGE, "0,67108864,1,suffix:B"), 4 * 1024 * 1024);
	GLOBAL_DEF(PropertyInfo(Variant::INT, "network/enet/send_buffer_size", PROPERTY_HINT_RANGE, "0,67108864,1,suffix:B"), 1024 * 1024);

	GDREGISTER_CLASS(ENetMultiplayerPeer);
	GDREGISTER_ABSTRACT_CLASS(ENetPacketPeer);
	GDREGISTER_CLASS(ENetConnection);
}

void uninitialize_enet_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	if (enet_ok) {
		enet_deinitialize();
	}
}
