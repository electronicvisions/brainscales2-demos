#pragma once
#include "grenade/vx/genpybind.h"
#include "grenade/vx/logical_network/projection.h"
#include "grenade/vx/network/plasticity_rule.h"
#include "halco/common/geometry.h"
#include "hate/visibility.h"
#include <map>
#include <string>
#include <vector>

namespace grenade::vx GENPYBIND_TAG_GRENADE_VX {

namespace logical_network GENPYBIND_MODULE {

/**
 * Plasticity rule.
 */
struct GENPYBIND(visible) PlasticityRule
{
	/**
	 * Descriptor to projections this rule has access to.
	 * All projections are required to be dense and in order.
	 */
	std::vector<ProjectionDescriptor> projections{};

	/**
	 * Population handle parameters.
	 */
	struct PopulationHandle
	{
		typedef lola::vx::v3::AtomicNeuron::Readout::Source NeuronReadoutSource;

		/**
		 * Descriptor of population.
		 */
		PopulationDescriptor descriptor;
		/**
		 * Readout source specification per neuron circuit used for static configuration such that
		 * the plasticity rule can read the specified signal.
		 */
		std::vector<std::map<
		    halco::hicann_dls::vx::v3::CompartmentOnLogicalNeuron,
		    std::vector<std::optional<NeuronReadoutSource>>>>
		    neuron_readout_sources;

		bool operator==(PopulationHandle const& other) const SYMBOL_VISIBLE;
		bool operator!=(PopulationHandle const& other) const SYMBOL_VISIBLE;

		GENPYBIND(stringstream)
		friend std::ostream& operator<<(std::ostream& os, PopulationHandle const& population)
		    SYMBOL_VISIBLE;
	};
	std::vector<PopulationHandle> populations;

	/**
	 * Plasticity rule kernel to be compiled into the PPU program.
	 */
	std::string kernel{};

	/**
	 * Timing information for execution of the rule.
	 */
	typedef network::PlasticityRule::Timer Timer;
	Timer timer;

	/**
	 * Enable whether this plasticity rule requires all projections to have one source per row and
	 * them being in order.
	 */
	bool enable_requires_one_source_per_row_in_order{false};

	/*
	 * Recording information for execution of the rule.
	 * Raw recording of one scratchpad memory region for all timed invocations of the
	 * rule. No automated recording of time is performed.
	 */
	typedef network::PlasticityRule::RawRecording RawRecording;

	/**
	 * Recording information for execution of the rule.
	 * Recording of exclusive scratchpad memory per rule invocation with
	 * time recording and returned data as time-annotated events.
	 */
	typedef network::PlasticityRule::TimedRecording TimedRecording;

	/**
	 * Recording memory provided to plasticity rule kernel and recorded after
	 * execution.
	 */
	typedef network::PlasticityRule::Recording Recording;
	std::optional<Recording> recording;

	/**
	 * Recording data corresponding to a raw recording.
	 */
	typedef network::PlasticityRule::RawRecordingData RawRecordingData;

	/**
	 * Extracted recorded data of observables corresponding to timed recording.
	 */
	struct TimedRecordingData
	{
		/**
		 * Recording per logical synapse, where for each sample the outer dimension of the data are
		 * the logical synapses of the projection and the inner dimension are the performed
		 * recordings of the corresponding hardware synapse(s).
		 */
		typedef std::variant<
		    std::vector<TimedDataSequence<std::vector<std::vector<int8_t>>>>,
		    std::vector<TimedDataSequence<std::vector<std::vector<uint8_t>>>>,
		    std::vector<TimedDataSequence<std::vector<std::vector<int16_t>>>>,
		    std::vector<TimedDataSequence<std::vector<std::vector<uint16_t>>>>>
		    EntryPerSynapse;

		/**
		 * Recording per logical neuron, where for each sample the outer dimension of the data are
		 * the logical neurons of the population and the inner dimensions are the performed
		 * recordings of the corresponding hardware neuron(s) per compartment.
		 */
		typedef std::variant<
		    std::vector<TimedDataSequence<std::vector<std::map<
		        halco::hicann_dls::vx::v3::CompartmentOnLogicalNeuron,
		        std::vector<int8_t>>>>>,
		    std::vector<TimedDataSequence<std::vector<std::map<
		        halco::hicann_dls::vx::v3::CompartmentOnLogicalNeuron,
		        std::vector<uint8_t>>>>>,
		    std::vector<TimedDataSequence<std::vector<std::map<
		        halco::hicann_dls::vx::v3::CompartmentOnLogicalNeuron,
		        std::vector<int16_t>>>>>,
		    std::vector<TimedDataSequence<std::vector<std::map<
		        halco::hicann_dls::vx::v3::CompartmentOnLogicalNeuron,
		        std::vector<uint16_t>>>>>>
		    EntryPerNeuron;

		typedef network::PlasticityRule::TimedRecordingData::Entry EntryArray;

		std::map<std::string, std::map<ProjectionDescriptor, EntryPerSynapse>> data_per_synapse;
		std::map<std::string, std::map<PopulationDescriptor, EntryPerNeuron>> data_per_neuron;
		std::map<std::string, EntryArray> data_array;
	};

	/**
	 * Recorded data.
	 */
	typedef std::variant<RawRecordingData, TimedRecordingData> RecordingData;

	PlasticityRule() = default;

	bool operator==(PlasticityRule const& other) const SYMBOL_VISIBLE;
	bool operator!=(PlasticityRule const& other) const SYMBOL_VISIBLE;

	GENPYBIND(stringstream)
	friend std::ostream& operator<<(std::ostream& os, PlasticityRule const& plasticity_rule)
	    SYMBOL_VISIBLE;
};


/** Descriptor to be used to identify a plasticity rule. */
struct GENPYBIND(inline_base("*")) PlasticityRuleDescriptor
    : public halco::common::detail::BaseType<PlasticityRuleDescriptor, size_t>
{
	constexpr explicit PlasticityRuleDescriptor(value_type const value = 0) : base_t(value) {}
};

} // namespace logical_network

GENPYBIND(postamble, tag(grenade_vx))
GENPYBIND_MANUAL({
	parent.attr("logical_network").attr("PlasticityRule").attr("Timer") =
	    parent.attr("PlasticityRule").attr("Timer");
	parent.attr("logical_network").attr("PlasticityRule").attr("RawRecording") =
	    parent.attr("PlasticityRule").attr("RawRecording");
	parent.attr("logical_network").attr("PlasticityRule").attr("TimedRecording") =
	    parent.attr("PlasticityRule").attr("TimedRecording");
	parent.attr("logical_network").attr("PlasticityRule").attr("RawRecordingData") =
	    parent.attr("PlasticityRule").attr("RawRecordingData");
})

typedef TimedData<std::vector<std::vector<int8_t>>> _SingleEntryPerSynapseInt8
    GENPYBIND(opaque(false));
typedef TimedData<std::vector<std::vector<uint8_t>>> _SingleEntryPerSynapseUInt8
    GENPYBIND(opaque(false));
typedef TimedData<std::vector<std::vector<int16_t>>> _SingleEntryPerSynapseInt16
    GENPYBIND(opaque(false));
typedef TimedData<std::vector<std::vector<uint16_t>>> _SingleEntryPerSynapseUInt16
    GENPYBIND(opaque(false));
typedef TimedData<std::vector<int8_t>> _ArrayEntryInt8 GENPYBIND(opaque(false));
typedef TimedData<std::vector<uint8_t>> _ArrayEntryUInt8 GENPYBIND(opaque(false));
typedef TimedData<std::vector<int16_t>> _ArrayEntryInt16 GENPYBIND(opaque(false));
typedef TimedData<std::vector<uint16_t>> _ArrayEntryUInt16 GENPYBIND(opaque(false));

typedef TimedData<std::vector<
    std::map<halco::hicann_dls::vx::v3::CompartmentOnLogicalNeuron, std::vector<int8_t>>>>
    _SingleEntryPerNeuronInt8 GENPYBIND(opaque(false));
typedef TimedData<std::vector<
    std::map<halco::hicann_dls::vx::v3::CompartmentOnLogicalNeuron, std::vector<uint8_t>>>>
    _SingleEntryPerNeuronUInt8 GENPYBIND(opaque(false));
typedef TimedData<std::vector<
    std::map<halco::hicann_dls::vx::v3::CompartmentOnLogicalNeuron, std::vector<int16_t>>>>
    _SingleEntryPerNeuronInt16 GENPYBIND(opaque(false));
typedef TimedData<std::vector<
    std::map<halco::hicann_dls::vx::v3::CompartmentOnLogicalNeuron, std::vector<uint16_t>>>>
    _SingleEntryPerNeuronUInt16 GENPYBIND(opaque(false));

} // namespace grenade::vx

namespace std {

HALCO_GEOMETRY_HASH_CLASS(grenade::vx::logical_network::PlasticityRuleDescriptor)

} // namespace std
