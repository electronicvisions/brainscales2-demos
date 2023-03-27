#pragma once
#include "halco/hicann-dls/vx/v3/synapse_driver.h"
#include "haldls/vx/v3/padi.h"
#include "haldls/vx/v3/synapse_driver.h"
#include "hate/visibility.h"
#include <chrono>
#include <iosfwd>
#include <map>
#include <optional>
#include <set>
#include <variant>
#include <vector>

namespace grenade::vx::network {

/**
 * Allocation manager for synapse drivers on a single PADI-bus.
 * Given requested allocations and a policy to use, places these requests onto the available
 * synapse drivers.
 */
struct SynapseDriverOnPADIBusManager
{
	/**
	 * Label to identify events at synapse driver(s).
	 */
	typedef haldls::vx::v3::PADIEvent::RowSelectAddress Label;
	/**
	 * Mask to filter events at synapse driver(s) depending on their label.
	 */
	typedef haldls::vx::v3::SynapseDriverConfig::RowAddressCompareMask Mask;
	/**
	 * Synapse driver location.
	 */
	typedef halco::hicann_dls::vx::v3::SynapseDriverOnPADIBus SynapseDriver;

	/**
	 * Properties of a requested allocation of synapse drivers.
	 * A request has a single label for incoming events and possibly multiple requested synapse
	 * driver collections with specified shape.
	 */
	struct AllocationRequest
	{
		/**
		 * Single synapse driver collection.
		 */
		struct Shape
		{
			/**
			 * Number of synapse drivers.
			 */
			size_t size;
			/**
			 * Whether the synapse drivers shall be contiguous, i.e. without holes.
			 */
			bool contiguous;

			bool operator==(Shape const& other) const SYMBOL_VISIBLE;
			bool operator!=(Shape const& other) const SYMBOL_VISIBLE;

			friend std::ostream& operator<<(std::ostream& os, Shape const& value) SYMBOL_VISIBLE;
		};
		/**
		 * Collection of requested synapse drivers.
		 */
		std::vector<Shape> shapes;
		/**
		 * Label of events which shall be forwarded exclusively to allocated synapse drivers.
		 */
		Label label;

		/**
		 * Get accumulated size of all shapes of the request.
		 */
		size_t size() const SYMBOL_VISIBLE;

		/**
		 * Get whether the request is sensitive to ordering of shapes on the available synapse
		 * drivers. This is the case exactly if any shape shall be contiguous and the allocation
		 * does not contain only equal shapes.
		 */
		bool is_sensitive_for_shape_allocation_order() const SYMBOL_VISIBLE;

		bool operator==(AllocationRequest const& other) const SYMBOL_VISIBLE;
		bool operator!=(AllocationRequest const& other) const SYMBOL_VISIBLE;

		friend std::ostream& operator<<(std::ostream& os, AllocationRequest const& value)
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
		std::vector<std::vector<std::pair<SynapseDriver, Mask>>> synapse_drivers;

		friend std::ostream& operator<<(std::ostream& os, Allocation const& allocation)
		    SYMBOL_VISIBLE;
	};

	/**
	 * Construct manager with unavailable synapse driver locations, which will be excluded for
	 * placing allocation(s).
	 * @param unavailable_synapse_drivers Synapse drivers excluded from allocation(s).
	 */
	SynapseDriverOnPADIBusManager(std::set<SynapseDriver> const& unavailable_synapse_drivers = {})
	    SYMBOL_VISIBLE;

	struct AllocationPolicyGreedy
	{
		AllocationPolicyGreedy(bool enable_exclusive_first = true) SYMBOL_VISIBLE;
		bool enable_exclusive_first;
	};

	struct AllocationPolicyBacktracking
	{
		AllocationPolicyBacktracking(
		    std::optional<std::chrono::milliseconds> const& max_duration = std::nullopt)
		    SYMBOL_VISIBLE;
		std::optional<std::chrono::milliseconds> max_duration;
	};

	typedef std::variant<AllocationPolicyGreedy, AllocationPolicyBacktracking> AllocationPolicy;

	/**
	 * Find placement of requested allocations.
	 * @param requested_allocations Collection of requested allocations
	 * @param allocation_policy Policy to use for partitioning of available synapse drivers
	 * @return Actual allocations on success, std::nullopt on failure of placement
	 */
	std::optional<std::vector<Allocation>> solve(
	    std::vector<AllocationRequest> const& requested_allocations,
	    AllocationPolicy const& allocation_policy = AllocationPolicyBacktracking()) SYMBOL_VISIBLE;

private:
	std::set<SynapseDriver> m_unavailable_synapse_drivers;
};

std::ostream& operator<<(
    std::ostream& os, SynapseDriverOnPADIBusManager::AllocationPolicy const& value) SYMBOL_VISIBLE;

} // namespace grenade::vx::network
