#pragma once
#include "grenade/vx/genpybind.h"
#include "grenade/vx/graph.h"
#include "grenade/vx/logical_network/network.h"
#include "grenade/vx/logical_network/population.h"
#include "grenade/vx/network/network.h"
#include "hate/visibility.h"
#include <map>
#include <optional>

namespace grenade::vx GENPYBIND_TAG_GRENADE_VX {

namespace logical_network GENPYBIND_MODULE {

/**
 * Logical network representation.
 */
struct GENPYBIND(visible) NetworkGraph
{
	NetworkGraph() = default;

	/** Underlying network. */
	GENPYBIND(getter_for(network))
	std::shared_ptr<Network> const& get_network() const SYMBOL_VISIBLE;

	/** Hardware network. */
	GENPYBIND(getter_for(hardware_network))
	std::shared_ptr<network::Network> const& get_hardware_network() const SYMBOL_VISIBLE;

	/** Translation between logical and hardware populations. */
	typedef std::map<PopulationDescriptor, network::PopulationDescriptor> PopulationTranslation;
	GENPYBIND(getter_for(population_translation))
	PopulationTranslation const& get_population_translation() const SYMBOL_VISIBLE;

	/** Translation between logical and hardware neurons in populations. */
	typedef std::map<
	    PopulationDescriptor,
	    std::vector<
	        std::map<halco::hicann_dls::vx::v3::CompartmentOnLogicalNeuron, std::vector<size_t>>>>
	    NeuronTranslation;
	GENPYBIND(getter_for(neuron_translation))
	NeuronTranslation const& get_neuron_translation() const SYMBOL_VISIBLE;

	/** Translation between logical and hardware projections. */
	typedef std::multimap<
	    std::pair<ProjectionDescriptor, size_t>,
	    std::pair<network::ProjectionDescriptor, size_t>>
	    ProjectionTranslation;
	GENPYBIND(getter_for(projection_translation))
	ProjectionTranslation const& get_projection_translation() const SYMBOL_VISIBLE;

	/** Translation between logical and hardware populations. */
	typedef std::map<PlasticityRuleDescriptor, network::PlasticityRuleDescriptor>
	    PlasticityRuleTranslation;
	GENPYBIND(getter_for(plasticity_rule_translation))
	PlasticityRuleTranslation const& get_plasticity_rule_translation() const SYMBOL_VISIBLE;

private:
	std::shared_ptr<Network> m_network;
	std::shared_ptr<network::Network> m_hardware_network;
	PopulationTranslation m_population_translation;
	NeuronTranslation m_neuron_translation;
	ProjectionTranslation m_projection_translation;
	PlasticityRuleTranslation m_plasticity_rule_translation;

	friend NetworkGraph build_network_graph(std::shared_ptr<Network> const& network);
};

} // namespace network

} // namespace grenade::vx
