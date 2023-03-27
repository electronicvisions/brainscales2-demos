#pragma once
#include "grenade/vx/network/population.h"
#include "grenade/vx/network/projection.h"
#include "grenade/vx/network/synapse_driver_on_dls_manager.h"
#include "halco/hicann-dls/vx/v3/neuron.h"
#include "halco/hicann-dls/vx/v3/padi.h"
#include "halco/hicann-dls/vx/v3/synapse_driver.h"
#include "haldls/vx/v3/synapse_driver.h"
#include "hate/visibility.h"
#include <memory>
#include <optional>
#include <set>
#include <vector>

namespace log4cxx {
class Logger;
typedef std::shared_ptr<Logger> LoggerPtr;
} // namespace log4cxx

namespace grenade::vx::network {

/**
 * Partitioning manager for sources projecting onto internal neurons via events propagated by
 * PADI-busses. Given all sources and their destination(s), distribute their event paths over
 * PADI-busses. Depending on the type of source, the constraints for possible PADI-busses vary.
 * Internal sources project at one fixed PADI-bus per hemisphere and background sources project at
 * one fixed PADI-bus per chip while external sources are freely distributable over all PADI-busses
 * of a chip.
 */
struct SourceOnPADIBusManager
{
	/**
	 * Label to identify events at synapse driver(s).
	 */
	typedef SynapseDriverOnDLSManager::Label Label;
	/**
	 * Synapse driver location.
	 */
	typedef SynapseDriverOnDLSManager::SynapseDriver SynapseDriver;

	/**
	 * Properties of an internal source.
	 */
	struct InternalSource
	{
		/**
		 * Location of source.
		 */
		halco::hicann_dls::vx::v3::AtomicNeuronOnDLS neuron;
		/**
		 * Number of required synapses per atomic neuron.
		 */
		std::map<
		    Projection::ReceptorType,
		    halco::common::typed_array<size_t, halco::hicann_dls::vx::v3::AtomicNeuronOnDLS>>
		    out_degree;
		/**
		 * Get PADI-bus on block which carries outgoing events of source.
		 * @return PADI-bus on block
		 */
		halco::hicann_dls::vx::v3::PADIBusOnPADIBusBlock toPADIBusOnPADIBusBlock() const
		    SYMBOL_VISIBLE;
	};

	/**
	 * Properties of a background source.
	 */
	struct BackgroundSource
	{
		/**
		 * Number of required synapses per atomic neuron.
		 */
		std::map<
		    Projection::ReceptorType,
		    halco::common::typed_array<size_t, halco::hicann_dls::vx::v3::AtomicNeuronOnDLS>>
		    out_degree;
		/**
		 * PADI-bus which carries outgoing events of source.
		 */
		halco::hicann_dls::vx::v3::PADIBusOnDLS padi_bus;
	};

	/**
	 * Properties of an external source.
	 */
	struct ExternalSource
	{
		/**
		 * Number of required synapses per atomic neuron.
		 */
		std::map<
		    Projection::ReceptorType,
		    halco::common::typed_array<size_t, halco::hicann_dls::vx::v3::AtomicNeuronOnDLS>>
		    out_degree;
	};

	/**
	 * Partitioning of sources in label space and onto PADI-busses.
	 */
	struct Partition
	{
		/**
		 * Group of sources projecting onto a single allocation request for synapse drivers.
		 */
		struct Group
		{
			/**
			 * Collection of source indices into the respective vector of sources supplied to
			 * solve().
			 */
			std::vector<size_t> sources;
			/**
			 * Synapse driver allocation request.
			 */
			SynapseDriverOnDLSManager::AllocationRequest allocation_request;

			friend std::ostream& operator<<(std::ostream& os, Group const& config) SYMBOL_VISIBLE;

			/**
			 * Check validity of group.
			 * A group is valid exactly if the sources are unique and their number allows unique
			 * addressing in the synapse matrix.
			 */
			bool valid() const SYMBOL_VISIBLE;
		};
		/**
		 * Groups of internal sources with their synapse driver allocation requests.
		 */
		std::vector<Group> internal;
		/**
		 * Groups of background sources with their synapse driver allocation requests.
		 */
		std::vector<Group> background;
		/**
		 * Groups of external sources with their synapse driver allocation requests.
		 */
		std::vector<Group> external;

		/**
		 * Check validity of partition.
		 * A partition is valid exactly if its groups are valid.
		 */
		bool valid() const SYMBOL_VISIBLE;

		friend std::ostream& operator<<(std::ostream& os, Partition const& config) SYMBOL_VISIBLE;
	};

	typedef std::map<
	    halco::hicann_dls::vx::v3::NeuronEventOutputOnDLS,
	    std::set<halco::hicann_dls::vx::v3::HemisphereOnDLS>>
	    DisabledInternalRoutes;

	/**
	 * Construct manager.
	 * @param disabled_internal_routes Disabled routes between internal sources and targets, which
	 * should never be used and required
	 */
	SourceOnPADIBusManager(DisabledInternalRoutes const& disabled_internal_routes = {})
	    SYMBOL_VISIBLE;

	/**
	 * Partition sources into allocation requests for projected-on synapse drivers.
	 * @param internal_sources Collection of internal sources
	 * @param background_sources Collection of background sources
	 * @param external_sources Collection of external sources
	 * @return Partitioning of sources into synapse driver allocation requests.
	 *         If unsuccessful, returns none.
	 */
	std::optional<Partition> solve(
	    std::vector<InternalSource> const& internal_sources,
	    std::vector<BackgroundSource> const& background_sources,
	    std::vector<ExternalSource> const& external_sources) const SYMBOL_VISIBLE;

private:
	log4cxx::LoggerPtr m_logger;

	DisabledInternalRoutes m_disabled_internal_routes;
};

} // namespace grenade::vx::network
