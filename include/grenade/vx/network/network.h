#pragma once
#include "grenade/vx/genpybind.h"
#include "grenade/vx/network/cadc_recording.h"
#include "grenade/vx/network/madc_recording.h"
#include "grenade/vx/network/plasticity_rule.h"
#include "grenade/vx/network/population.h"
#include "grenade/vx/network/projection.h"
#include "hate/visibility.h"
#include <chrono>
#include <iosfwd>
#include <map>
#include <memory>
#include <variant>

#if defined(__GENPYBIND__) || defined(__GENPYBIND_GENERATED__)
#include <pybind11/chrono.h>
#endif

namespace grenade::vx GENPYBIND_TAG_GRENADE_VX {

namespace network {

/**
 * Placed but not routed network consisting of populations and projections.
 */
struct GENPYBIND(visible, holder_type("std::shared_ptr<grenade::vx::network::Network>")) Network
{
	std::map<
	    PopulationDescriptor,
	    std::variant<Population, ExternalPopulation, BackgroundSpikeSourcePopulation>> const
	    populations;
	std::map<ProjectionDescriptor, Projection> const projections;
	std::optional<MADCRecording> const madc_recording;
	std::optional<CADCRecording> const cadc_recording;
	std::map<PlasticityRuleDescriptor, PlasticityRule> const plasticity_rules;

	/**
	 * Duration spent during construction of network.
	 * This value is not compared in operator{==,!=}.
	 */
	std::chrono::microseconds const construction_duration;

	bool operator==(Network const& other) const SYMBOL_VISIBLE;
	bool operator!=(Network const& other) const SYMBOL_VISIBLE;

	GENPYBIND(stringstream)
	friend std::ostream& operator<<(std::ostream& os, Network const& network) SYMBOL_VISIBLE;
};

} // namespace network

} // namespace grenade::vx
