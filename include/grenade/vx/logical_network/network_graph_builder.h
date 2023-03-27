#pragma once
#include "grenade/vx/genpybind.h"
#include "grenade/vx/logical_network/network.h"
#include "grenade/vx/logical_network/network_graph.h"
#include "hate/visibility.h"
#include <memory>

namespace grenade::vx GENPYBIND_TAG_GRENADE_VX {

namespace logical_network GENPYBIND_MODULE {

/**
 * Build a hardware network representation for a given network.
 * @param network Network for which to build hardware network representation
 */
NetworkGraph GENPYBIND(visible)
    build_network_graph(std::shared_ptr<Network> const& network) SYMBOL_VISIBLE;

} // namespace logical_network

} // namespace grenade::vx
