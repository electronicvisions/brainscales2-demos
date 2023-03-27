#pragma once
#include "grenade/vx/genpybind.h"
#include "grenade/vx/logical_network/population.h"
#include "hate/visibility.h"
#include "lola/vx/v3/neuron.h"

namespace grenade::vx GENPYBIND_TAG_GRENADE_VX {

namespace logical_network GENPYBIND_MODULE {

/**
 * MADC recording of a single neuron.
 */
struct GENPYBIND(visible) MADCRecording
{
	PopulationDescriptor population{};

	size_t neuron_on_population{0};
	halco::hicann_dls::vx::v3::CompartmentOnLogicalNeuron compartment_on_neuron{};
	size_t atomic_neuron_on_compartment{0};

	lola::vx::v3::AtomicNeuron::Readout::Source source{
	    lola::vx::v3::AtomicNeuron::Readout::Source::membrane};

	MADCRecording() = default;

	bool operator==(MADCRecording const& other) const SYMBOL_VISIBLE;
	bool operator!=(MADCRecording const& other) const SYMBOL_VISIBLE;
};

} // namespace logical_network

} // namespace grenade::vx
