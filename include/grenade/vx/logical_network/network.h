#pragma once
#include "grenade/vx/genpybind.h"
#include "grenade/vx/logical_network/cadc_recording.h"
#include "grenade/vx/logical_network/madc_recording.h"
#include "grenade/vx/logical_network/plasticity_rule.h"
#include "grenade/vx/logical_network/population.h"
#include "grenade/vx/logical_network/projection.h"
#include "hate/visibility.h"
#include <iosfwd>
#include <map>
#include <memory>
#include <variant>

namespace grenade::vx GENPYBIND_TAG_GRENADE_VX {

namespace logical_network GENPYBIND_MODULE {

/**
 * Placed but not routed network consisting of populations and projections.
 */
struct GENPYBIND(
    visible, holder_type("std::shared_ptr<grenade::vx::logical_network::Network>")) Network
{
	std::map<
	    PopulationDescriptor,
	    std::variant<Population, ExternalPopulation, BackgroundSpikeSourcePopulation>> const
	    populations;
	std::map<ProjectionDescriptor, Projection> const projections;
	std::optional<MADCRecording> const madc_recording;
	std::optional<CADCRecording> const cadc_recording;
	std::map<PlasticityRuleDescriptor, PlasticityRule> const plasticity_rules;

	bool operator==(Network const& other) const SYMBOL_VISIBLE;
	bool operator!=(Network const& other) const SYMBOL_VISIBLE;

	GENPYBIND(stringstream)
	friend std::ostream& operator<<(std::ostream& os, Network const& network) SYMBOL_VISIBLE;
};

} // namespace logical_network

} // namespace grenade::vx
