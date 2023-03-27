#pragma once
#include "grenade/vx/genpybind.h"
#include "grenade/vx/graph.h"
#include "grenade/vx/network/network.h"
#include "grenade/vx/network/population.h"
#include "halco/hicann-dls/vx/v3/event.h"
#include "hate/visibility.h"
#include <chrono>
#include <optional>
#include <vector>

namespace grenade::vx GENPYBIND_TAG_GRENADE_VX {

namespace network {

struct RoutingResult;
struct NetworkGraph;
struct NetworkGraphStatistics;

NetworkGraph build_network_graph(
    std::shared_ptr<Network> const& network, RoutingResult const& routing_result);

void update_network_graph(NetworkGraph& network_graph, std::shared_ptr<Network> const& network);

/**
 * Hardware graph representation of a placed and routed network.
 */
struct GENPYBIND(visible) NetworkGraph
{
	NetworkGraph() = default;

	/** Underlying network. */
	GENPYBIND(getter_for(network))
	std::shared_ptr<Network> const& get_network() const SYMBOL_VISIBLE;

	/** Graph representing the network. */
	GENPYBIND(getter_for(graph))
	Graph const& get_graph() const SYMBOL_VISIBLE;

	/** Vertex descriptor at which to insert external spike data. */
	GENPYBIND(getter_for(event_input_vertex))
	std::optional<Graph::vertex_descriptor> get_event_input_vertex() const SYMBOL_VISIBLE;

	/** Vertex descriptor from which to extract recorded spike data. */
	GENPYBIND(getter_for(event_output_vertex))
	std::optional<Graph::vertex_descriptor> get_event_output_vertex() const SYMBOL_VISIBLE;

	/** Vertex descriptor from which to extract recorded madc sample data. */
	GENPYBIND(getter_for(madc_sample_output_vertex))
	std::optional<Graph::vertex_descriptor> get_madc_sample_output_vertex() const SYMBOL_VISIBLE;

	/** Vertex descriptor from which to extract recorded cadc sample data. */
	GENPYBIND(getter_for(cadc_sample_output_vertex))
	std::vector<Graph::vertex_descriptor> get_cadc_sample_output_vertex() const SYMBOL_VISIBLE;

	/** Vertex descriptors of synapse views. */
	GENPYBIND(getter_for(synapse_vertices))
	std::map<
	    ProjectionDescriptor,
	    std::map<halco::hicann_dls::vx::HemisphereOnDLS, Graph::vertex_descriptor>> const&
	get_synapse_vertices() const SYMBOL_VISIBLE;

	/** Vertex descriptors of neuron views. */
	GENPYBIND(getter_for(neuron_vertices))
	std::map<
	    PopulationDescriptor,
	    std::map<halco::hicann_dls::vx::HemisphereOnDLS, Graph::vertex_descriptor>> const&
	get_neuron_vertices() const SYMBOL_VISIBLE;

	/** Vertex descriptor from which to extract recorded plasticity rule scratchpad memory. */
	GENPYBIND(getter_for(plasticity_rule_output_vertices))
	std::map<PlasticityRuleDescriptor, Graph::vertex_descriptor> const&
	get_plasticity_rule_output_vertices() const SYMBOL_VISIBLE;

	/**
	 * Spike labels corresponding to each neuron in a population.
	 * For external populations these are the input spike labels, for internal population this is
	 * only given for populations with enabled recording.
	 */
	typedef std::map<
	    PopulationDescriptor,
	    std::vector<std::vector<std::optional<halco::hicann_dls::vx::v3::SpikeLabel>>>>
	    SpikeLabels;
	GENPYBIND(getter_for(spike_labels))
	SpikeLabels const& get_spike_labels() const SYMBOL_VISIBLE;

	/**
	 * Checks validity of hardware graph representation in relation to the abstract network.
	 * This ensures all required elements and information being present as well as a functionally
	 * correct mapping and routing.
	 */
	bool valid() const SYMBOL_VISIBLE;

	/*
	 * Placed connection in synapse matrix.
	 */
	struct PlacedConnection
	{
		/** Weight of connection. */
		lola::vx::v3::SynapseMatrix::Weight weight;
		/** Vertical location. */
		halco::hicann_dls::vx::v3::SynapseRowOnDLS synapse_row;
		/** Horizontal location. */
		halco::hicann_dls::vx::v3::SynapseOnSynapseRow synapse_on_row;

		GENPYBIND(stringstream)
		friend std::ostream& operator<<(std::ostream&, PlacedConnection const&) SYMBOL_VISIBLE;
	};

	typedef std::vector<PlacedConnection> PlacedConnections;

	PlacedConnection get_placed_connection(ProjectionDescriptor descriptor, size_t index) const
	    SYMBOL_VISIBLE;
	PlacedConnections get_placed_connections(ProjectionDescriptor descriptor) const SYMBOL_VISIBLE;

private:
	std::shared_ptr<Network> m_network;
	Graph m_graph;
	std::optional<Graph::vertex_descriptor> m_event_input_vertex;
	std::optional<Graph::vertex_descriptor> m_event_output_vertex;
	std::optional<Graph::vertex_descriptor> m_madc_sample_output_vertex;
	std::vector<Graph::vertex_descriptor> m_cadc_sample_output_vertex;
	std::map<
	    ProjectionDescriptor,
	    std::map<halco::hicann_dls::vx::HemisphereOnDLS, Graph::vertex_descriptor>>
	    m_synapse_vertices;
	std::map<
	    PopulationDescriptor,
	    std::map<halco::hicann_dls::vx::HemisphereOnDLS, Graph::vertex_descriptor>>
	    m_neuron_vertices;
	std::map<
	    PopulationDescriptor,
	    std::map<halco::hicann_dls::vx::HemisphereOnDLS, Graph::vertex_descriptor>>
	    m_background_spike_source_vertices;
	std::map<PlasticityRuleDescriptor, Graph::vertex_descriptor> m_plasticity_rule_vertices;
	std::map<PlasticityRuleDescriptor, Graph::vertex_descriptor> m_plasticity_rule_output_vertices;
	SpikeLabels m_spike_labels;

	std::chrono::microseconds m_construction_duration;
	std::chrono::microseconds m_verification_duration;
	std::chrono::microseconds m_routing_duration;

	friend NetworkGraph build_network_graph(
	    std::shared_ptr<Network> const& network, RoutingResult const& routing_result);
	friend void update_network_graph(
	    NetworkGraph& network_graph, std::shared_ptr<Network> const& network);
	friend NetworkGraphStatistics extract_statistics(NetworkGraph const& network_graph);
};

} // namespace network

} // namespace grenade::vx
