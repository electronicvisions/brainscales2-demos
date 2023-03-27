#pragma once
#include "grenade/vx/network/synapse_driver_on_padi_bus_manager.h"
#include "halco/common/geometry.h"
#include "halco/hicann-dls/vx/v3/padi.h"
#include "halco/hicann-dls/vx/v3/synapse_driver.h"
#include "haldls/vx/v3/synapse_driver.h"
#include "hate/visibility.h"
#include <chrono>
#include <memory>
#include <optional>
#include <set>
#include <vector>

namespace log4cxx {
class Logger;
typedef std::shared_ptr<Logger> LoggerPtr;
} // namespace log4cxx

namespace grenade::vx::network {

/**
 * Allocation manager for synapse drivers on all PADI-busses of one chip.
 * Given requested allocations and a policy to use, places these requests onto the available synapse
 * drivers.
 */
struct SynapseDriverOnDLSManager
{
	/**
	 * Label to identify events at synapse driver(s).
	 */
	typedef SynapseDriverOnPADIBusManager::Label Label;
	/**
	 * Synapse driver location.
	 */
	typedef SynapseDriverOnPADIBusManager::SynapseDriver SynapseDriver;
	/**
	 * Allocation policy to use per PADI-bus.
	 */
	typedef SynapseDriverOnPADIBusManager::AllocationPolicy AllocationPolicy;
	typedef SynapseDriverOnPADIBusManager::AllocationPolicyGreedy AllocationPolicyGreedy;
	typedef SynapseDriverOnPADIBusManager::AllocationPolicyBacktracking
	    AllocationPolicyBacktracking;

	/**
	 * Properties of a potential allocation of synapse drivers.
	 */
	struct AllocationRequest
	{
		typedef SynapseDriverOnPADIBusManager::AllocationRequest::Shape Shape;
		/**
		 * Shape(s) of allocation.
		 */
		std::map<halco::hicann_dls::vx::v3::PADIBusOnDLS, std::vector<Shape>> shapes;
		/**
		 * Potential labels of events which shall be forwarded exclusively to allocated synapse
		 * drivers.
		 */
		std::vector<Label> labels;
		/**
		 * Descriptor to be used to identify a depdendency between the label space of multiple
		 * allocation requests. Their labels are constrained such that their index in
		 * AllocationRequest::labels has to be equal.
		 */
		struct GENPYBIND(inline_base("*")) DependentLabelGroup
		    : public halco::common::detail::BaseType<DependentLabelGroup, size_t>
		{
			constexpr explicit DependentLabelGroup(value_type const value = 0) : base_t(value) {}
		};
		std::optional<DependentLabelGroup> dependent_label_group;

		/**
		 * Get whether request is valid by checking for non-empty labels and shapes.
		 */
		bool valid() const SYMBOL_VISIBLE;

		bool operator==(AllocationRequest const& other) const SYMBOL_VISIBLE;
		bool operator!=(AllocationRequest const& other) const SYMBOL_VISIBLE;

		friend std::ostream& operator<<(std::ostream& os, AllocationRequest const& config)
		    SYMBOL_VISIBLE;
	};

	/**
	 * Allocation of synapse drivers.
	 */
	struct Allocation
	{
		/**
		 * Collection of synapse drivers with their mask.
		 */
		std::map<halco::hicann_dls::vx::v3::PADIBusOnDLS, SynapseDriverOnPADIBusManager::Allocation>
		    synapse_drivers;
		/**
		 * Labels of events.
		 */
		Label label;

		friend std::ostream& operator<<(std::ostream& os, Allocation const& config) SYMBOL_VISIBLE;
	};

	/**
	 * Construct manager with unavailable synapse driver locations, which will be excluded for
	 * placing allocation(s).
	 * @param unavailable_synapse_drivers Synapse drivers excluded from allocation(s).
	 */
	SynapseDriverOnDLSManager(
	    std::set<halco::hicann_dls::vx::v3::SynapseDriverOnDLS> const& unavailable_synapse_drivers =
	        {}) SYMBOL_VISIBLE;

	/**
	 * Find placement of requested allocations.
	 * @param requested_allocations Collection of requested allocations
	 * @param allocation_policy Policy to use for partitioning of available synapse drivers
	 * @param timeout Optional timeout for brute-forcing label combinations per dependent
	 * PADI-busses
	 * @return Actual allocations on success, std::nullopt on failure of placement
	 */
	std::optional<std::vector<Allocation>> solve(
	    std::vector<AllocationRequest> const& requested_allocations,
	    AllocationPolicy const& allocation_policy = AllocationPolicyBacktracking(),
	    std::optional<std::chrono::milliseconds> const& timeout = std::nullopt) SYMBOL_VISIBLE;

private:
	halco::common::
	    typed_array<SynapseDriverOnPADIBusManager, halco::hicann_dls::vx::v3::PADIBusOnDLS>
	        m_synapse_driver_on_padi_bus_manager;
	log4cxx::LoggerPtr m_logger;
};

} // namespace grenade::vx::networke
