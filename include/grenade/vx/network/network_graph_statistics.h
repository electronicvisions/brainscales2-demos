#pragma once
#include "grenade/vx/genpybind.h"
#include "hate/visibility.h"
#include <chrono>
#include <cstddef>
#include <iosfwd>

#if defined(__GENPYBIND__) || defined(__GENPYBIND_GENERATED__)
#include <pybind11/chrono.h>
#endif

namespace grenade::vx GENPYBIND_TAG_GRENADE_VX {

namespace network {

struct NetworkGraphStatistics;
struct NetworkGraph;

/**
 * Extract statistics from network graph.
 */
NetworkGraphStatistics GENPYBIND(visible)
    extract_statistics(NetworkGraph const& network_graph) SYMBOL_VISIBLE;

/**
 * Statistics of network graph.
 */
struct GENPYBIND(visible) NetworkGraphStatistics
{
	NetworkGraphStatistics() = default;

	/**
	 * Get number of populations in network graph.
	 */
	size_t get_num_populations() const SYMBOL_VISIBLE;
	/**
	 * Get number of projections in network graph.
	 */
	size_t get_num_projections() const SYMBOL_VISIBLE;
	/**
	 * Get number of neurons in network graph.
	 * This is the same amount as hardware circuits.
	 */
	size_t get_num_neurons() const SYMBOL_VISIBLE;
	/**
	 * Get number of synapses in network graph.
	 * This is the same amount as hardware circuits.
	 */
	size_t get_num_synapses() const SYMBOL_VISIBLE;
	/**
	 * Get number of synapse drivers in network graph.
	 * This is the same amount as hardware circuits.
	 */
	size_t get_num_synapse_drivers() const SYMBOL_VISIBLE;

	/**
	 * Get used fraction of neurons vs. all hardware circuits.
	 */
	double get_neuron_usage() const SYMBOL_VISIBLE;
	/**
	 * Get used fraction of synapses vs. all hardware circuits.
	 */
	double get_synapse_usage() const SYMBOL_VISIBLE;
	/**
	 * Get used fraction of synapse drivers vs. all hardware circuits.
	 */
	double get_synapse_driver_usage() const SYMBOL_VISIBLE;

	/**
	 * Get duration spent constructing abstract network (Network).
	 */
	std::chrono::microseconds get_abstract_network_construction_duration() const SYMBOL_VISIBLE;
	/**
	 * Get duration spent constructing hardware network (NetworkGraph) given a routing result and
	 * abstract network.
	 */
	std::chrono::microseconds get_hardware_network_construction_duration() const SYMBOL_VISIBLE;
	/**
	 * Get duration spent verifying hardware network (NetworkGraph) for correct routing.
	 */
	std::chrono::microseconds get_verification_duration() const SYMBOL_VISIBLE;
	/**
	 * Get duration spent routing the abstract network (Network).
	 */
	std::chrono::microseconds get_routing_duration() const SYMBOL_VISIBLE;

	GENPYBIND(stringstream)
	friend std::ostream& operator<<(std::ostream& os, NetworkGraphStatistics const& value)
	    SYMBOL_VISIBLE;

private:
	size_t m_num_populations{0};
	size_t m_num_projections{0};
	size_t m_num_neurons{0};
	size_t m_num_synapses{0};
	size_t m_num_synapse_drivers{0};

	double m_neuron_usage{0.};
	double m_synapse_usage{0.};
	double m_synapse_driver_usage{0.};

	std::chrono::microseconds m_abstract_network_construction_duration{0};
	std::chrono::microseconds m_hardware_network_construction_duration{0};
	std::chrono::microseconds m_verification_duration{0};
	std::chrono::microseconds m_routing_duration{0};

	friend NetworkGraphStatistics extract_statistics(NetworkGraph const& network_graph);
};

} // namespace network

} // namespace grenade::vx
