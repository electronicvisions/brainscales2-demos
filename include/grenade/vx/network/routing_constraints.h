#pragma once
#include "grenade/vx/network/population.h"
#include "grenade/vx/network/projection.h"
#include "halco/common/typed_array.h"
#include "halco/hicann-dls/vx/v3/neuron.h"
#include "halco/hicann-dls/vx/v3/padi.h"
#include "hate/visibility.h"
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace grenade::vx::network {

struct Network;

/**
 * Accessors for routing constraints for a given abstract network.
 */
struct RoutingConstraints
{
	/**
	 * Construct constraints from and for an abstract network.
	 * @param network Abstract network
	 */
	RoutingConstraints(Network const& network) SYMBOL_VISIBLE;

	/**
	 * Check if routing in possible in principle given the hardware limitations.
	 * The supported in-degree per neuron is checked as well as the required number of synapse rows
	 * per PADI-bus.
	 * @throws std::runtime_error On unsatisfiable constraints
	 */
	void check() const SYMBOL_VISIBLE;

	/**
	 * Synaptic connection between a pair of on-chip neurons.
	 */
	struct InternalConnection
	{
		/** Source neuron. */
		halco::hicann_dls::vx::v3::AtomicNeuronOnDLS source;
		/** Target neuron. */
		halco::hicann_dls::vx::v3::AtomicNeuronOnDLS target;
		/** Descriptor of connection in abstract network. */
		std::pair<ProjectionDescriptor, size_t> descriptor;
		/** Receptor type of connection. */
		Projection::ReceptorType receptor_type;

		/** Get PADI-bus onto which the synapse is to be placed. */
		halco::hicann_dls::vx::v3::PADIBusOnDLS toPADIBusOnDLS() const SYMBOL_VISIBLE;

		bool operator==(InternalConnection const& other) const SYMBOL_VISIBLE;
		bool operator!=(InternalConnection const& other) const SYMBOL_VISIBLE;
	};

	/**
	 * Get internal connections of network.
	 * @return Internal connections
	 */
	std::vector<InternalConnection> get_internal_connections() const SYMBOL_VISIBLE;

	/**
	 * Synaptic connection from a background spike source to an on-chip neuron.
	 */
	struct BackgroundConnection
	{
		/** Source circuit. */
		halco::hicann_dls::vx::v3::BackgroundSpikeSourceOnDLS source;
		/** Target neuron. */
		halco::hicann_dls::vx::v3::AtomicNeuronOnDLS target;
		/** Descriptor of connection in abstract network. */
		std::pair<ProjectionDescriptor, size_t> descriptor;
		/** Receptor type of connection. */
		Projection::ReceptorType receptor_type;

		/** Get PADI-bus onto which the synapse is to be placed. */
		halco::hicann_dls::vx::v3::PADIBusOnDLS toPADIBusOnDLS() const SYMBOL_VISIBLE;

		bool operator==(BackgroundConnection const& other) const SYMBOL_VISIBLE;
		bool operator!=(BackgroundConnection const& other) const SYMBOL_VISIBLE;
	};

	/**
	 * Get background connections of network.
	 * @return Background connections
	 */
	std::vector<BackgroundConnection> get_background_connections() const SYMBOL_VISIBLE;

	/**
	 * Synaptic connection from an external source to an on-chip neuron.
	 */
	struct ExternalConnection
	{
		/** Target neuron. */
		halco::hicann_dls::vx::v3::AtomicNeuronOnDLS target;
		/** Descriptor of connection in abstract network. */
		std::pair<ProjectionDescriptor, size_t> descriptor;
		/** Receptor type of connection. */
		Projection::ReceptorType receptor_type;

		/** Get PADI-bus block onto which the synapse is to be placed. */
		halco::hicann_dls::vx::v3::PADIBusBlockOnDLS toPADIBusBlockOnDLS() const SYMBOL_VISIBLE;

		bool operator==(ExternalConnection const& other) const SYMBOL_VISIBLE;
		bool operator!=(ExternalConnection const& other) const SYMBOL_VISIBLE;
	};

	/**
	 * Get external connections of network.
	 * @return External connections
	 */
	std::vector<ExternalConnection> get_external_connections() const SYMBOL_VISIBLE;

	/**
	 * Get number of external connections placed onto each chip hemisphere per receptor type.
	 */
	halco::common::typed_array<
	    std::map<Projection::ReceptorType, std::vector<std::pair<ProjectionDescriptor, size_t>>>,
	    halco::hicann_dls::vx::v3::HemisphereOnDLS>
	get_external_connections_per_hemisphere() const SYMBOL_VISIBLE;

	/**
	 * Get number of external sources projecting onto each chip hemisphere per receptor type.
	 */
	halco::common::typed_array<
	    std::map<Projection::ReceptorType, std::set<std::pair<PopulationDescriptor, size_t>>>,
	    halco::hicann_dls::vx::v3::HemisphereOnDLS>
	get_external_sources_to_hemisphere() const SYMBOL_VISIBLE;

	/**
	 * Get in-degree per on-chip neuron.
	 */
	halco::common::typed_array<size_t, halco::hicann_dls::vx::v3::AtomicNeuronOnDLS>
	get_neuron_in_degree() const SYMBOL_VISIBLE;

	/**
	 * Get in-degree per on-chip neuron for each incoming PADI-bus.
	 */
	halco::common::typed_array<
	    halco::common::typed_array<size_t, halco::hicann_dls::vx::v3::PADIBusOnPADIBusBlock>,
	    halco::hicann_dls::vx::v3::AtomicNeuronOnDLS>
	get_neuron_in_degree_per_padi_bus() const SYMBOL_VISIBLE;

	/**
	 * Get in-degree per on-chip neuron for each receptor type.
	 */
	halco::common::typed_array<
	    std::map<Projection::ReceptorType, size_t>,
	    halco::hicann_dls::vx::v3::AtomicNeuronOnDLS>
	get_neuron_in_degree_per_receptor_type() const SYMBOL_VISIBLE;

	/**
	 * Get in-degree per on-chip neuron for each incoming PADI-bus and receptor type.
	 */
	halco::common::typed_array<
	    halco::common::typed_array<
	        std::map<Projection::ReceptorType, size_t>,
	        halco::hicann_dls::vx::v3::PADIBusOnPADIBusBlock>,
	    halco::hicann_dls::vx::v3::AtomicNeuronOnDLS>
	get_neuron_in_degree_per_receptor_type_per_padi_bus() const SYMBOL_VISIBLE;

	/**
	 * Get required number of synapse rows for each PADI-bus and receptor type combination.
	 */
	halco::common::typed_array<
	    std::map<Projection::ReceptorType, size_t>,
	    halco::hicann_dls::vx::v3::PADIBusOnDLS>
	get_num_synapse_rows_per_padi_bus_per_receptor_type() const SYMBOL_VISIBLE;

	/**
	 * Get required number of synapse rows for each PADI-bus.
	 * This is the accumulation of all numbers of synapse rows required for different receptor types
	 * on each PADI-bus.
	 */
	halco::common::typed_array<size_t, halco::hicann_dls::vx::v3::PADIBusOnDLS>
	get_num_synapse_rows_per_padi_bus() const SYMBOL_VISIBLE;

	/**
	 * Get neurons which are neither recorded nor serve as source of (a) connection(s).
	 */
	std::set<halco::hicann_dls::vx::v3::AtomicNeuronOnDLS> get_neither_recorded_nor_source_neurons()
	    const SYMBOL_VISIBLE;

	/**
	 * Get on-chip neurons forwarding their events per neuron event output.
	 */
	std::map<
	    halco::hicann_dls::vx::v3::NeuronEventOutputOnDLS,
	    std::vector<halco::hicann_dls::vx::v3::AtomicNeuronOnDLS>>
	get_neurons_on_event_output() const SYMBOL_VISIBLE;

	/**
	 * Get on-chip neurons forwarding their events onto each PADI-bus.
	 * This function assumes the crossbar to forward all events at every node.
	 */
	std::map<
	    halco::hicann_dls::vx::v3::PADIBusOnDLS,
	    std::set<halco::hicann_dls::vx::v3::AtomicNeuronOnDLS>>
	get_neurons_on_padi_bus() const SYMBOL_VISIBLE;

	/**
	 * Get neuron event outputs projecting onto each PADI-bus.
	 * This function assumes the crossbar to forward all events at every node.
	 */
	std::map<
	    halco::hicann_dls::vx::v3::PADIBusOnDLS,
	    std::set<halco::hicann_dls::vx::v3::NeuronEventOutputOnDLS>>
	get_neuron_event_outputs_on_padi_bus() const SYMBOL_VISIBLE;

	/**
	 * Get number of background spike sources projecting onto each PADI-bus.
	 */
	halco::common::typed_array<size_t, halco::hicann_dls::vx::v3::PADIBusOnDLS>
	get_num_background_sources_on_padi_bus() const SYMBOL_VISIBLE;

	/**
	 * Collection of constraints for a single PADI-bus.
	 */
	struct PADIBusConstraints
	{
		/**
		 * Background spike source number.
		 */
		size_t num_background_spike_sources;

		/**
		 * Neuron sources.
		 */
		std::set<halco::hicann_dls::vx::v3::AtomicNeuronOnDLS> neuron_sources;

		/**
		 * Neuron circuits, which don't serve as source for any other target neuron but are still
		 * recorded and therefore elicit spike events.
		 */
		std::set<halco::hicann_dls::vx::v3::AtomicNeuronOnDLS> only_recorded_neurons;

		/**
		 * Internal connections.
		 */
		std::vector<InternalConnection> internal_connections;

		/**
		 * Background connections.
		 */
		std::vector<BackgroundConnection> background_connections;
	};

	/**
	 * Get constraints for each PADI-bus.
	 */
	halco::common::typed_array<PADIBusConstraints, halco::hicann_dls::vx::v3::PADIBusOnDLS>
	get_padi_bus_constraints() const SYMBOL_VISIBLE;

private:
	Network const& m_network;
};

} // namespace grenade::vx::network
