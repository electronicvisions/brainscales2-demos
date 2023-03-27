#pragma once
#include "grenade/vx/network/plasticity_rule.h"
#include "hate/visibility.h"
#include <optional>
#include <set>
#include <vector>

#if defined(__GENPYBIND__) || defined(__GENPYBIND_GENERATED__)
#include <pybind11/stl.h>
#endif


namespace grenade::vx {
struct IODataMap;
} // namespace grenade::vx

namespace grenade::vx GENPYBIND_TAG_GRENADE_VX {
namespace network {

struct PlasticityRuleDescriptor;
struct NetworkGraph;

struct GENPYBIND(visible) OnlyRecordingPlasticityRuleGenerator
{
	/**
	 * Observables, which can be recorded.
	 */
	enum class Observable
	{
		weights,
		correlation_causal,
		correlation_acausal
	};

	OnlyRecordingPlasticityRuleGenerator(std::set<Observable> const& observables) SYMBOL_VISIBLE;

	/**
	 * Generate plasticity rule which only executes given recording.
	 * Timing and projection information is left default/empty.
	 */
	PlasticityRule generate() const SYMBOL_VISIBLE;

private:
	std::set<Observable> m_observables;
};

} // namespace grenade::vx
} // namespace network
