#pragma once
#include "grenade/vx/network/source_on_padi_bus_manager.h"
#include "grenade/vx/network/synapse_driver_on_dls_manager.h"
#include "halco/hicann-dls/vx/v3/neuron.h"
#include "halco/hicann-dls/vx/v3/padi.h"
#include "hate/visibility.h"
#include <optional>
#include <set>
#include <vector>

namespace grenade::vx::network::detail {

struct SourceOnPADIBusManager
{
	typedef grenade::vx::network::SourceOnPADIBusManager::Label Label;
	typedef grenade::vx::network::SourceOnPADIBusManager::SynapseDriver SynapseDriver;
	typedef grenade::vx::network::SourceOnPADIBusManager::InternalSource InternalSource;
	typedef grenade::vx::network::SourceOnPADIBusManager::BackgroundSource BackgroundSource;
	typedef grenade::vx::network::SourceOnPADIBusManager::ExternalSource ExternalSource;
	typedef grenade::vx::network::SourceOnPADIBusManager::Partition Partition;

	/**
	 * Split list of indices into source vector into chunks of synapse label size.
	 * Split is performed as [0, Label::size), [Label::size, ...), ... .
	 * @param filter List of indices to split
	 * @return Chunks of lists of indices
	 */
	static std::vector<std::vector<size_t>> split_linear(std::vector<size_t> const& filter)
	    SYMBOL_VISIBLE;

	/**
	 * Get number of synapse drivers required to place the synapses for the filtered sources.
	 * @param sources All sources containing their out-degree to neurons
	 * @param filter List of indices in sources to filter for calculation
	 * @return Number of required synapse drivers per PADI-bus
	 */
	template <typename S>
	static halco::common::typed_array<size_t, halco::hicann_dls::vx::v3::PADIBusBlockOnDLS>
	get_num_synapse_drivers(std::vector<S> const& sources, std::vector<size_t> const& filter);

	/**
	 * Distribute external sources onto PADI-busses given the distribution of internal and
	 * background sources. The algorithm fills synapse drivers on PADI-busses in Enum-order.
	 * @param sources All external sources containing their out-degree to neurons
	 * @param used_synapse_drivers Number of used synapse drivers per PADI-bus
	 * @return Indices into external source collection distributed onto PADI-busses.
	 *         If no distribution is found, returns none.
	 */
	static std::optional<halco::common::typed_array<
	    std::vector<std::vector<size_t>>,
	    halco::hicann_dls::vx::v3::PADIBusOnDLS>>
	distribute_external_sources_linear(
	    std::vector<ExternalSource> const& sources,
	    halco::common::typed_array<size_t, halco::hicann_dls::vx::v3::PADIBusOnDLS> const&
	        used_num_synapse_drivers) SYMBOL_VISIBLE;

	/**
	 * Get allocation requests for filtered internal sources.
	 * @param filter List of indices into sources, split into chunks
	 * @param padi_bus PADI-bus location
	 * @param num_synapse_drivers Number of synapse driver required for each chunk of sources
	 * @return Allocation requests
	 */
	static std::vector<SynapseDriverOnDLSManager::AllocationRequest>
	get_allocation_requests_internal(
	    std::vector<std::vector<size_t>> const& filter,
	    halco::hicann_dls::vx::v3::PADIBusOnPADIBusBlock const& padi_bus,
	    halco::hicann_dls::vx::v3::NeuronBackendConfigBlockOnDLS const& backend_block,
	    halco::common::typed_array<
	        std::vector<size_t>,
	        halco::hicann_dls::vx::v3::PADIBusOnDLS> const& num_synapse_drivers) SYMBOL_VISIBLE;

	/**
	 * Get allocation requests for filtered background sources.
	 * @param filter List of indices into sources, split into chunks
	 * @param padi_bus PADI-bus location
	 * @param num_synapse_drivers Number of synapse driver required for each chunk of sources
	 * @return Allocation requests for synapse driver allocation algorithm
	 */
	static std::vector<SynapseDriverOnDLSManager::AllocationRequest>
	get_allocation_requests_background(
	    std::vector<std::vector<size_t>> const& filter,
	    halco::hicann_dls::vx::v3::PADIBusOnDLS const& padi_bus,
	    std::vector<size_t> const& num_synapse_drivers) SYMBOL_VISIBLE;

	/**
	 * Get allocation requests for external sources.
	 * @param padi_bus PADI-bus location
	 * @param num_synapse_drivers Number of synapse driver required for each chunk of sources
	 * @return Allocation requests for synapse driver allocation algorithm
	 */
	static SynapseDriverOnDLSManager::AllocationRequest get_allocation_requests_external(
	    halco::hicann_dls::vx::v3::PADIBusOnDLS const& padi_bus,
	    size_t num_synapse_drivers) SYMBOL_VISIBLE;
};

} // namespace grenade::vx::network::detail
