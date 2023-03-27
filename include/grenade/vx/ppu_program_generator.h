#pragma once
#include "grenade/vx/graph.h"
#include "grenade/vx/ppu/neuron_view_handle.h"
#include "grenade/vx/ppu/synapse_array_view_handle.h"
#include "grenade/vx/vertex/plasticity_rule.h"
#include "halco/hicann-dls/vx/v3/synapse.h"
#include "halco/hicann-dls/vx/v3/synram.h"
#include "hate/visibility.h"
#include <tuple>
#include <vector>

namespace grenade::vx {

class PPUProgramGenerator
{
public:
	PPUProgramGenerator() = default;

	void add(
	    Graph::vertex_descriptor descriptor,
	    vertex::PlasticityRule const& rule,
	    std::vector<
	        std::pair<halco::hicann_dls::vx::v3::SynramOnDLS, ppu::SynapseArrayViewHandle>> const&
	        synapses,
	    std::vector<
	        std::pair<halco::hicann_dls::vx::v3::NeuronRowOnDLS, ppu::NeuronViewHandle>> const&
	        neurons) SYMBOL_VISIBLE;

	std::vector<std::string> done() SYMBOL_VISIBLE;

	bool has_periodic_cadc_readout = false;

private:
	std::vector<std::tuple<
	    Graph::vertex_descriptor,
	    vertex::PlasticityRule,
	    std::vector<std::pair<halco::hicann_dls::vx::v3::SynramOnDLS, ppu::SynapseArrayViewHandle>>,
	    std::vector<std::pair<halco::hicann_dls::vx::v3::NeuronRowOnDLS, ppu::NeuronViewHandle>>>>
	    m_plasticity_rules;
};

} // namespace grenade::vx
