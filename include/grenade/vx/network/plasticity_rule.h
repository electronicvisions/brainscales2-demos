#pragma once
#include "grenade/vx/genpybind.h"
#include "grenade/vx/network/projection.h"
#include "grenade/vx/vertex/plasticity_rule.h"
#include "halco/common/geometry.h"
#include "hate/visibility.h"
#include <optional>
#include <string>
#include <vector>

namespace grenade::vx GENPYBIND_TAG_GRENADE_VX {

namespace network {

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
		 * Readout source specification per neuron used for static configuration such that the
		 * plasticity rule can read the specified signal.
		 */
		std::vector<std::optional<NeuronReadoutSource>> neuron_readout_sources;

		bool operator==(PopulationHandle const& other) const SYMBOL_VISIBLE;
		bool operator!=(PopulationHandle const& other) const SYMBOL_VISIBLE;

		GENPYBIND(stringstream)
		friend std::ostream& operator<<(std::ostream& os, PopulationHandle const& population)
		    SYMBOL_VISIBLE;
	};
	/**
	 * List of populations this rule has access to.
	 */
	std::vector<PopulationHandle> populations;

	/**
	 * Plasticity rule kernel to be compiled into the PPU program.
	 */
	std::string kernel{};

	/**
	 * Timing information for execution of the rule.
	 */
	struct Timer
	{
		/** PPU clock cycles. */
		struct GENPYBIND(inline_base("*")) Value
		    : public halco::common::detail::RantWrapper<Value, uintmax_t, 0xffffffff, 0>
		{
			constexpr explicit Value(uintmax_t const value = 0) : rant_t(value) {}
		};

		Value start;
		Value period;
		size_t num_periods;

		bool operator==(Timer const& other) const SYMBOL_VISIBLE;
		bool operator!=(Timer const& other) const SYMBOL_VISIBLE;

		GENPYBIND(stringstream)
		friend std::ostream& operator<<(std::ostream& os, Timer const& timer) SYMBOL_VISIBLE;
	} timer;

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
	typedef vertex::PlasticityRule::RawRecording RawRecording GENPYBIND(opaque(false));

	/**
	 * Recording information for execution of the rule.
	 * Recording of exclusive scratchpad memory per rule invocation with
	 * time recording and returned data as time-annotated events.
	 */
	typedef vertex::PlasticityRule::TimedRecording TimedRecording GENPYBIND(opaque(false));

	/**
	 * Recording memory provided to plasticity rule kernel and recorded after
	 * execution.
	 */
	typedef vertex::PlasticityRule::Recording Recording;
	std::optional<Recording> recording;

	/**
	 * Recording data corresponding to a raw recording.
	 */
	typedef vertex::PlasticityRule::RawRecordingData RawRecordingData GENPYBIND(opaque(false));

	/**
	 * Extracted recorded data of observables corresponding to timed recording.
	 */
	struct TimedRecordingData
	{
		typedef vertex::PlasticityRule::TimedRecordingData::Entry Entry;

		std::map<std::string, std::map<ProjectionDescriptor, Entry>> data_per_synapse;
		std::map<std::string, std::map<PopulationDescriptor, Entry>> data_per_neuron;
		std::map<std::string, Entry> data_array;
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

} // namespace network

} // namespace grenade::vx

namespace std {

HALCO_GEOMETRY_HASH_CLASS(grenade::vx::network::PlasticityRuleDescriptor)

} // namespace std
