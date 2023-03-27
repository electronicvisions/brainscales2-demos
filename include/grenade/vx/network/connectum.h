#pragma once
#include "grenade/vx/graph.h"
#include "grenade/vx/network/population.h"
#include "grenade/vx/network/projection.h"
#include "haldls/vx/v3/event.h"
#include "hate/visibility.h"
#include <iosfwd>
#include <variant>
#include <vector>

namespace grenade::vx::network {

struct ConnectumConnection
{
	typedef std::pair<PopulationDescriptor, size_t> Source;
	Source source;

	halco::hicann_dls::vx::v3::AtomicNeuronOnDLS target;

	Projection::ReceptorType receptor_type;
	Projection::Connection::Weight weight;

	bool operator==(ConnectumConnection const& other) const SYMBOL_VISIBLE;
	bool operator!=(ConnectumConnection const& other) const SYMBOL_VISIBLE;

	friend std::ostream& operator<<(std::ostream& os, ConnectumConnection const& config)
	    SYMBOL_VISIBLE;
};

typedef std::vector<ConnectumConnection> Connectum;

struct Network;
struct NetworkGraph;

Connectum generate_connectum_from_abstract_network(NetworkGraph const& network_graph)
    SYMBOL_VISIBLE;
Connectum generate_connectum_from_hardware_network(NetworkGraph const& network_graph)
    SYMBOL_VISIBLE;

} // namespace grenade::vx::network
