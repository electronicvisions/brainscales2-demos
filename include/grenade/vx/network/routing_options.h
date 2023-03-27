#pragma once
#include "grenade/vx/genpybind.h"
#include "grenade/vx/network/synapse_driver_on_dls_manager.h"
#include "lola/vx/v3/synapse.h"
#include <chrono>
#include <iosfwd>
#include <optional>

#if defined(__GENPYBIND__) || defined(__GENPYBIND_GENERATED__)
#include <pybind11/chrono.h>
#endif

namespace grenade::vx GENPYBIND_TAG_GRENADE_VX {

namespace network {

/**
 * Options to be passed to routing algorithm.
 */
struct GENPYBIND(visible) RoutingOptions
{
	RoutingOptions() SYMBOL_VISIBLE;

	/**
	 * Policy to be used in synapse driver allocation.
	 */
	typedef SynapseDriverOnDLSManager::AllocationPolicy AllocationPolicy;
	SynapseDriverOnDLSManager::AllocationPolicy synapse_driver_allocation_policy;
	typedef SynapseDriverOnDLSManager::AllocationPolicyGreedy AllocationPolicyGreedy
	    GENPYBIND(opaque);
	typedef SynapseDriverOnDLSManager::AllocationPolicyBacktracking AllocationPolicyBacktracking
	    GENPYBIND(opaque);

	/**
	 * Optional timeout to be used in synapse driver allocation algorithm for iteration over label
	 * combinations.
	 */
	std::optional<std::chrono::milliseconds> synapse_driver_allocation_timeout;

	GENPYBIND(stringstream)
	friend std::ostream& operator<<(std::ostream& os, RoutingOptions const& options) SYMBOL_VISIBLE;
};

GENPYBIND_MANUAL({
	auto cls = pybind11::class_<::grenade::vx::network::RoutingOptions::AllocationPolicy>(
	    parent, "_RoutingOptions_AllocationPolicy");
	cls.def(
	       pybind11::init<
	           ::grenade::vx::network::SynapseDriverOnDLSManager::AllocationPolicyGreedy>(),
	       pybind11::arg("value") =
	           ::grenade::vx::network::SynapseDriverOnDLSManager::AllocationPolicyGreedy())
	    .def(
	        pybind11::init<
	            ::grenade::vx::network::SynapseDriverOnDLSManager::AllocationPolicyBacktracking>(),
	        pybind11::arg("value"));
	parent.attr("RoutingOptions").attr("AllocationPolicy") =
	    parent.attr("_RoutingOptions_AllocationPolicy");
})

} // namespace network

} // namespace grenade::vx
