#pragma once
#include "grenade/vx/execution_instance.h"
#include "grenade/vx/graph.h"
#include "grenade/vx/ppu/neuron_view_handle.h"
#include "grenade/vx/ppu/synapse_array_view_handle.h"
#include "grenade/vx/vertex/plasticity_rule.h"
#include "halco/hicann-dls/vx/v3/chip.h"
#include "halco/hicann-dls/vx/v3/neuron.h"
#include "halco/hicann-dls/vx/v3/synapse.h"
#include "hate/visibility.h"
#include "lola/vx/v3/chip.h"
#include "stadls/vx/v3/playback_program_builder.h"
#include <optional>
#include <tuple>
#include <vector>

namespace grenade::vx {

/**
 * Visitor of graph vertices of a single execution instance for construction of the initial
 * configuration.
 * The result is applied to a given configuration object.
 */
class ExecutionInstanceConfigVisitor
{
public:
	/**
	 * Construct visitor.
	 * @param graph Graph to use for locality and property lookup
	 * @param execution_instance Local execution instance to visit
	 * @param config Configuration to alter
	 */
	ExecutionInstanceConfigVisitor(
	    Graph const& graph,
	    coordinate::ExecutionInstance const& execution_instance,
	    lola::vx::v3::Chip& config) SYMBOL_VISIBLE;

	/**
	 * Perform visit operation and generate initial configuration.
	 * @return Reference to altered chip object and optional PPU program
	 * symbols
	 */
	std::tuple<lola::vx::v3::Chip&, std::optional<lola::vx::v3::PPUElfFile::symbols_type>>
	operator()() SYMBOL_VISIBLE;

private:
	Graph const& m_graph;
	coordinate::ExecutionInstance m_execution_instance;

	lola::vx::v3::Chip& m_config;

	halco::common::typed_array<bool, halco::hicann_dls::vx::v3::NeuronResetOnDLS>
	    m_enabled_neuron_resets;
	bool m_requires_ppu;
	bool m_has_periodic_cadc_readout;
	bool m_used_madc;

	std::vector<std::tuple<
	    Graph::vertex_descriptor,
	    vertex::PlasticityRule,
	    std::vector<std::pair<halco::hicann_dls::vx::v3::SynramOnDLS, ppu::SynapseArrayViewHandle>>,
	    std::vector<std::pair<halco::hicann_dls::vx::v3::NeuronRowOnDLS, ppu::NeuronViewHandle>>>>
	    m_plasticity_rules;

	/**
	 * Process single vertex.
	 * This function is called in preprocess.
	 * @param vertex Vertex descriptor
	 * @param data Data associated with vertex
	 */
	template <typename Vertex>
	void process(Graph::vertex_descriptor const vertex, Vertex const& data);

	/**
	 * Preprocess by single visit of all local vertices.
	 */
	void pre_process() SYMBOL_VISIBLE;
};

} // namespace grenade::vx
