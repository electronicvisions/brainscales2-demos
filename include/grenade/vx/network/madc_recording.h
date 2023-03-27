#pragma once
#include "grenade/vx/genpybind.h"
#include "grenade/vx/network/population.h"
#include "hate/visibility.h"
#include "lola/vx/v3/neuron.h"

namespace grenade::vx GENPYBIND_TAG_GRENADE_VX {

namespace network {

/**
 * MADC recording of a single neuron.
 */
struct GENPYBIND(visible) MADCRecording
{
	PopulationDescriptor population{};
	size_t index{0};
	typedef lola::vx::v3::AtomicNeuron::Readout::Source Source;
	Source source{Source::membrane};

	MADCRecording() = default;
	MADCRecording(PopulationDescriptor population, size_t index, Source source) SYMBOL_VISIBLE;

	bool operator==(MADCRecording const& other) const SYMBOL_VISIBLE;
	bool operator!=(MADCRecording const& other) const SYMBOL_VISIBLE;
};

} // namespace network

} // namespace grenade::vx
