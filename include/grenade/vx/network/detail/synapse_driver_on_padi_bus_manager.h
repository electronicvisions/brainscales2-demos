#pragma once
#include "grenade/vx/network/synapse_driver_on_padi_bus_manager.h"
#include "hate/visibility.h"
#include <map>
#include <set>
#include <vector>

namespace grenade::vx::network::detail {

/**
 * Implementation details of the allocation manager for a single PADI-bus.
 */
struct SynapseDriverOnPADIBusManager
{
	typedef grenade::vx::network::SynapseDriverOnPADIBusManager::Label Label;
	typedef grenade::vx::network::SynapseDriverOnPADIBusManager::Mask Mask;
	typedef grenade::vx::network::SynapseDriverOnPADIBusManager::SynapseDriver SynapseDriver;
	typedef grenade::vx::network::SynapseDriverOnPADIBusManager::AllocationRequest
	    AllocationRequest;
	typedef grenade::vx::network::SynapseDriverOnPADIBusManager::Allocation Allocation;
	typedef grenade::vx::network::SynapseDriverOnPADIBusManager::AllocationPolicy AllocationPolicy;

	/**
	 * Check whether given label is forwarded to synapses at specified synapse driver with mask.
	 * @param label Event label
	 * @param mask Mask at synapse driver
	 * @param synapse_driver Synapse driver location
	 * @return Boolean value
	 */
	static bool forwards(Label const& label, Mask const& mask, SynapseDriver const& synapse_driver)
	    SYMBOL_VISIBLE;

	/**
	 * Check whether the requested allocations feature unique labels.
	 * @param requested_allocations Requested allocations to check
	 * @return Boolean value
	 */
	static bool has_unique_labels(std::vector<AllocationRequest> const& requested_allocations)
	    SYMBOL_VISIBLE;

	/**
	 * Check whether the accumulated size of all requested allocations is less than or equal to the
	 * number of available synapse drivers.
	 * @param unavailable_synapse_drivers Unavailable synapse drivers
	 * @param requested_allocations Requested allocations to check
	 * @return Boolean value
	 */
	static bool allocations_fit_available_size(
	    std::set<SynapseDriver> const& unavailable_synapse_drivers,
	    std::vector<AllocationRequest> const& requested_allocations) SYMBOL_VISIBLE;

	/**
	 * Mask values which isolate a given requested allocation.
	 * Isolation means the label of no other requested allocation has the same value after masking.
	 * @tparam Key Index in the requested allocations
	 * @tparam Value Collection of masks
	 */
	typedef std::map<size_t, std::vector<Mask>> IsolatingMasks;

	/**
	 * Generate isolating mask values.
	 * @param requested_allocations Requested allocations to generate isolating masks for
	 * @return Generated isolating masks
	 */
	static IsolatingMasks generate_isolating_masks(
	    std::vector<AllocationRequest> const& requested_allocations) SYMBOL_VISIBLE;

	/**
	 * Get whether all requested allocations can in principle be isolated by at least one mask
	 * value.
	 * @param isolating_masks Isolating masks to given requested allocations
	 * @param requested_allocations Requested allocations
	 * @return Boolean value
	 */
	static bool allocations_can_be_isolated(
	    IsolatingMasks const& isolating_masks,
	    std::vector<AllocationRequest> const& requested_allocations) SYMBOL_VISIBLE;

	/**
	 * Synapse drivers which allow isolation of requested allocation from all others given the mask
	 * value. Isolation means no other requested allocation's events are forwarded to the synapses
	 * of that driver.
	 * @tparam Key Synapse driver location
	 * @tparam Value Pair of requested allocation index and mask value which isolated the request
	 */
	typedef std::map<SynapseDriver, std::vector<std::pair<size_t, Mask>>> IsolatedSynapseDrivers;

	/**
	 * Generate isolated synapse driver locations.
	 * @param unavailable_synapse_drivers Synapse drivers to exclude from selection
	 * @param isolating_masks Isolating masks to given requested allocations
	 * @param requested_allocations Requested allocations
	 * @return Generated isolated synapse drivers
	 */
	static IsolatedSynapseDrivers generate_isolated_synapse_drivers(
	    std::set<SynapseDriver> const& unavailable_synapse_drivers,
	    IsolatingMasks const& isolating_masks,
	    std::vector<AllocationRequest> const& requested_allocations) SYMBOL_VISIBLE;

	/**
	 * Get whether all requested allocations can in principle be placed individually when
	 * disregarding the placement of all other requested allocations.
	 * @param isolated_synapse_drivers Isolated synapse drivers
	 * @param requested_allocations Requested allocations
	 * @return Boolean value
	 */
	static bool allocations_can_be_placed_individually(
	    IsolatedSynapseDrivers const& isolated_synapse_drivers,
	    std::vector<AllocationRequest> const& requested_allocations) SYMBOL_VISIBLE;

	/**
	 * Get whether set of synapse drivers is contiguous.
	 * An empty collection is defined to be contiguous.
	 * @param synapse_drivers Synapse driver set to test
	 * @return Boolean value
	 */
	static bool is_contiguous(std::set<SynapseDriver> const& synapse_drivers) SYMBOL_VISIBLE;

	/**
	 * Check whether allocations adhere to the requested allocations' requirements.
	 * @param allocations Allocations to given requested allocations
	 * @param requested_allocations Requested allocations
	 * @return Boolean value
	 */
	static bool valid(
	    std::vector<Allocation> const& allocations,
	    std::vector<AllocationRequest> const& requested_allocations) SYMBOL_VISIBLE;

	/**
	 * Place requested allocations in a greedy manner.
	 * First the first requested allocation is placed, then the second and so on.
	 * Additionally to iterating the synapse drivers linearly, synapse driver locations which only
	 * allow isolation of a single request can be used first.
	 * @param requested_allocations Requested allocations
	 * @param isolated_synapse_drivers Isolated synapse drivers
	 * @param exclusive_first First use exclusive synapse drivers
	 * @return Allocations
	 */
	static std::vector<Allocation> allocate_greedy(
	    std::vector<AllocationRequest> const& requested_allocations,
	    IsolatedSynapseDrivers const& isolated_synapse_drivers,
	    bool exclusive_first) SYMBOL_VISIBLE;

	/**
	 * Place requested allocations using backtracking.
	 * If a solution is possible, it is found (possibly at the cost of larger time consumption).
	 * If any requested allocation is sensitive to shape allocation order, the algorithm does not
	 * explore different order permutations.
	 * This means that for a requested allocation {Shape{2, not contiguous}, Shape{3, contiguous}}
	 * and a possible allocation {SynapseDriver(0), SynapseDriver(1), SynapseDriver(2),
	 * SynapseDriver(10), SynapseDriver(17)}, the algorithm does not flip the order of the requested
	 * shapes to allocate Shape{2, not contiguous} to {SynapseDriver(10), SynapseDriver(17)}, but
	 * tries {SynapseDriver(0), SynapseDriver(1)} and then fails to place the second contiguous
	 * shape.
	 * However, the occurrence of requests which are sensitive to shape allocation order is assumed
	 * to be rare, since contiguous shapes are only of use in the context of PPU algorithms and then
	 * additionally, the shapes in a request need to be inhomogeneous, which is only the case if
	 * different synapse driver configuration groups are requested. In these complex corner cases,
	 * if additionally the algorithm fails to find any solution, the user can permute the shapes
	 * himself outside of this manager.
	 * @param requested_allocations Requested allocations
	 * @param isolated_synapse_drivers Isolated synapse drivers
	 * @param max_duration Optional specification of maximal wall-clock time to perform backtracking
	 * @return Allocations
	 */
	static std::vector<Allocation> allocate_backtracking(
	    std::vector<AllocationRequest> const& requested_allocations,
	    IsolatedSynapseDrivers const& isolated_synapse_drivers,
	    std::optional<std::chrono::milliseconds> const& max_duration = std::nullopt) SYMBOL_VISIBLE;
};

} // namespace grenade::vx::network::detail
