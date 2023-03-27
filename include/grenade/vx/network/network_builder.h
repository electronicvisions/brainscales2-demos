#pragma once
#include "grenade/vx/genpybind.h"
#include "grenade/vx/network/madc_recording.h"
#include "grenade/vx/network/network.h"
#include "grenade/vx/network/plasticity_rule.h"
#include "grenade/vx/network/population.h"
#include "grenade/vx/network/projection.h"
#include "hate/visibility.h"
#include <chrono>
#include <map>
#include <memory>
#include <variant>

namespace log4cxx {
class Logger;
typedef std::shared_ptr<Logger> LoggerPtr;
} // namespace log4cxx

namespace grenade::vx GENPYBIND_TAG_GRENADE_VX {

namespace network {

class GENPYBIND(visible) NetworkBuilder
{
public:
	/**
	 * Add on-chip population.
	 * The population is expected to feature unique and unused neuron locations.
	 * @param population Population to add
	 */
	PopulationDescriptor add(Population const& population) SYMBOL_VISIBLE;

	/**
	 * Add off-chip population.
	 * @param population Population to add
	 */
	PopulationDescriptor add(ExternalPopulation const& population) SYMBOL_VISIBLE;

	/**
	 * Add on-chip background spike source population.
	 * @param population Population to add
	 */
	PopulationDescriptor add(BackgroundSpikeSourcePopulation const& population) SYMBOL_VISIBLE;

	/**
	 * Add projection between already added populations.
	 * The projection is expected to be free of single connections present in already added
	 * projections. A single connection is considered equal, if it connects the same pre- and
	 * post-synaptic neurons and features the same receptor type.
	 * @param projection Projection to add
	 */
	ProjectionDescriptor add(Projection const& projection) SYMBOL_VISIBLE;

	/**
	 * Add MADC recording of a single neuron.
	 * Only one MADC recording per network is allowed.
	 * If another recording is present at the recorded neuron, their source specification is
	 * required to match.
	 * @param madc_recording MADC recording to add
	 */
	void add(MADCRecording const& madc_recording) SYMBOL_VISIBLE;

	/**
	 * Add CADC recording of a collection of neurons.
	 * Only one CADC recording per network is allowed.
	 * If another recording is present at a recorded neuron, their source specification is required
	 * to match.
	 * @param cadc_recording CADC recording to add
	 */
	void add(CADCRecording const& cadc_recording) SYMBOL_VISIBLE;

	/*
	 * Add plasticity rule on already added projections.
	 * The projections are expected to be dense and in order.
	 * @param plasticity_rule PlasticityRule to add
	 */
	PlasticityRuleDescriptor add(PlasticityRule const& plasticity_rule) SYMBOL_VISIBLE;

	NetworkBuilder() SYMBOL_VISIBLE;

	std::shared_ptr<Network> done() SYMBOL_VISIBLE;

private:
	std::map<
	    PopulationDescriptor,
	    std::variant<Population, ExternalPopulation, BackgroundSpikeSourcePopulation>>
	    m_populations{};
	std::map<ProjectionDescriptor, Projection> m_projections{};
	std::optional<MADCRecording> m_madc_recording{std::nullopt};
	std::optional<CADCRecording> m_cadc_recording{std::nullopt};
	std::map<PlasticityRuleDescriptor, PlasticityRule> m_plasticity_rules{};
	std::chrono::microseconds m_duration;
	log4cxx::LoggerPtr m_logger;
};

} // namespace network

} // namespace grenade::vx
