#pragma once
#include "grenade/vx/genpybind.h"
#include "grenade/vx/graph.h"
#include "grenade/vx/network/cadc_recording.h"
#include "grenade/vx/network/madc_recording.h"
#include "grenade/vx/network/network.h"
#include "grenade/vx/network/network_graph.h"
#include "grenade/vx/network/plasticity_rule.h"
#include "grenade/vx/network/population.h"
#include "grenade/vx/network/projection.h"
#include "grenade/vx/network/routing_result.h"
#include "halco/hicann-dls/vx/v3/background.h"
#include "halco/hicann-dls/vx/v3/neuron.h"
#include "halco/hicann-dls/vx/v3/padi.h"
#include "halco/hicann-dls/vx/v3/synapse.h"
#include "halco/hicann-dls/vx/v3/synapse_driver.h"
#include "hate/visibility.h"
#include <map>
#include <memory>
#include <optional>
#include <vector>

namespace log4cxx {
class Logger;
typedef std::shared_ptr<Logger> LoggerPtr;
} // namespace log4cxx

namespace grenade::vx GENPYBIND_TAG_GRENADE_VX {

namespace network {

/**
 * Build a hardware graph representation for a given network with the given routing result.
 * @param network Network for which to build hardware graph representation
 * @param routing_result Routing result to use matching the given network
 */
NetworkGraph GENPYBIND(visible) build_network_graph(
    std::shared_ptr<Network> const& network, RoutingResult const& routing_result) SYMBOL_VISIBLE;

/**
 * Update an exisiting hardware graph representation.
 * For this to work, no new routing has to have been required.
 * @param network_graph Existing hardware graph representation to update or fill with newly built
 * instance
 * @param network New network for which to update or build
 */
void GENPYBIND(visible) update_network_graph(
    NetworkGraph& network_graph, std::shared_ptr<Network> const& network) SYMBOL_VISIBLE;

/**
 * Network graph builder wrapper for network.
 */
class NetworkGraphBuilder
{
public:
	NetworkGraphBuilder(Network const& network) SYMBOL_VISIBLE;

	struct Resources
	{
		struct PlacedPopulation
		{
			std::map<halco::hicann_dls::vx::HemisphereOnDLS, Graph::vertex_descriptor> neurons;
		};

		struct PlacedProjection
		{
			std::map<halco::hicann_dls::vx::HemisphereOnDLS, Graph::vertex_descriptor> synapses;
		};

		std::optional<Graph::vertex_descriptor> external_input;
		std::map<
		    PopulationDescriptor,
		    std::map<halco::hicann_dls::vx::v3::HemisphereOnDLS, Graph::vertex_descriptor>>
		    background_spike_sources;
		std::optional<Graph::vertex_descriptor> crossbar_l2_input;
		std::map<halco::hicann_dls::vx::v3::CrossbarNodeOnDLS, Graph::vertex_descriptor>
		    crossbar_nodes;
		std::map<halco::hicann_dls::vx::v3::PADIBusOnDLS, Graph::vertex_descriptor> padi_busses;
		std::map<halco::hicann_dls::vx::v3::SynapseDriverOnDLS, Graph::vertex_descriptor>
		    synapse_drivers;
		std::map<halco::hicann_dls::vx::v3::NeuronEventOutputOnDLS, Graph::vertex_descriptor>
		    neuron_event_outputs;
		std::map<PopulationDescriptor, PlacedPopulation> populations;
		std::map<ProjectionDescriptor, PlacedProjection> projections;
		std::optional<Graph::vertex_descriptor> external_output;
		std::optional<Graph::vertex_descriptor> madc_output;
		std::vector<Graph::vertex_descriptor> cadc_output;
		std::map<PlasticityRuleDescriptor, Graph::vertex_descriptor> plasticity_rules;
		std::map<PlasticityRuleDescriptor, Graph::vertex_descriptor> plasticity_rules_output;
	};

	static std::vector<Input> get_inputs(Graph const& graph, Graph::vertex_descriptor descriptor);

	void add_external_input(
	    Graph& graph, Resources& resources, coordinate::ExecutionInstance const& instance) const;

	void add_background_spike_sources(
	    Graph& graph,
	    Resources& resources,
	    coordinate::ExecutionInstance const& instance,
	    RoutingResult const& routing_result) const;

	void add_padi_bus(
	    Graph& graph,
	    Resources& resources,
	    halco::hicann_dls::vx::PADIBusOnDLS const& coordinate,
	    coordinate::ExecutionInstance const& instance) const;

	void add_crossbar_node(
	    Graph& graph,
	    Resources& resources,
	    halco::hicann_dls::vx::CrossbarNodeOnDLS const& coordinate,
	    RoutingResult const& connection_result,
	    coordinate::ExecutionInstance const& instance) const;

	void add_synapse_driver(
	    Graph& graph,
	    Resources& resources,
	    halco::hicann_dls::vx::SynapseDriverOnDLS const& coordinate,
	    RoutingResult const& connection_result,
	    coordinate::ExecutionInstance const& instance) const;

	void add_neuron_event_output(
	    Graph& graph,
	    Resources& resources,
	    halco::hicann_dls::vx::NeuronEventOutputOnDLS const& coordinate,
	    coordinate::ExecutionInstance const& instance) const;

	void add_synapse_array_view_sparse(
	    Graph& graph,
	    Resources& resources,
	    ProjectionDescriptor descriptor,
	    RoutingResult const& connection_result,
	    coordinate::ExecutionInstance const& instance) const;

	void add_population(
	    Graph& graph,
	    Resources& resources,
	    std::map<halco::hicann_dls::vx::v3::HemisphereOnDLS, std::vector<Input>> const& input,
	    PopulationDescriptor const& descriptor,
	    RoutingResult const& connection_result,
	    coordinate::ExecutionInstance const& instance) const;

	std::map<halco::hicann_dls::vx::v3::HemisphereOnDLS, Input> add_projection_from_external_input(
	    Graph& graph,
	    Resources& resources,
	    ProjectionDescriptor const& descriptor,
	    RoutingResult const& connection_result,
	    coordinate::ExecutionInstance const& instance) const;

	std::map<halco::hicann_dls::vx::v3::HemisphereOnDLS, Input>
	add_projection_from_background_spike_source(
	    Graph& graph,
	    Resources& resources,
	    ProjectionDescriptor const& descriptor,
	    RoutingResult const& connection_result,
	    coordinate::ExecutionInstance const& instance) const;

	std::map<halco::hicann_dls::vx::v3::HemisphereOnDLS, Input> add_projection_from_internal_input(
	    Graph& graph,
	    Resources& resources,
	    ProjectionDescriptor const& descriptor,
	    RoutingResult const& connection_result,
	    coordinate::ExecutionInstance const& instance) const;

	void add_populations(
	    Graph& graph,
	    Resources& resources,
	    RoutingResult const& connection_result,
	    coordinate::ExecutionInstance const& instance) const;

	void add_neuron_event_outputs(
	    Graph& graph, Resources& resources, coordinate::ExecutionInstance const& instance) const;

	void add_external_output(
	    Graph& graph,
	    Resources& resources,
	    RoutingResult const& connection_result,
	    coordinate::ExecutionInstance const& instance) const;

	void add_madc_recording(
	    Graph& graph,
	    Resources& resources,
	    MADCRecording const& madc_recording,
	    coordinate::ExecutionInstance const& instance) const;

	void add_cadc_recording(
	    Graph& graph,
	    Resources& resources,
	    CADCRecording const& cadc_recording,
	    coordinate::ExecutionInstance const& instance) const;

	void add_plasticity_rules(
	    Graph& graph, Resources& resources, coordinate::ExecutionInstance const& instance) const;

	NetworkGraph::SpikeLabels get_spike_labels(RoutingResult const& connection_result);

private:
	Network const& m_network;
	log4cxx::LoggerPtr m_logger;
};

} // namespace network

} // namespace grenade::vx
