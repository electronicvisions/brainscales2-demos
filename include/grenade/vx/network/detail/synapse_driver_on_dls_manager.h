#pragma once
#include "grenade/vx/network/synapse_driver_on_dls_manager.h"
#include "grenade/vx/network/synapse_driver_on_padi_bus_manager.h"
#include "halco/common/geometry.h"
#include "halco/hicann-dls/vx/v3/padi.h"
#include "halco/hicann-dls/vx/v3/synapse_driver.h"
#include "haldls/vx/v3/synapse_driver.h"
#include "hate/visibility.h"
#include <set>
#include <vector>

namespace grenade::vx::network::detail {

/**
 * Implementation details of the allocation manager for all PADI-busses.
 */
struct SynapseDriverOnDLSManager
{
	typedef grenade::vx::network::SynapseDriverOnDLSManager::Label Label;
	typedef grenade::vx::network::SynapseDriverOnDLSManager::SynapseDriver SynapseDriver;
	typedef grenade::vx::network::SynapseDriverOnDLSManager::AllocationRequest AllocationRequest;
	typedef grenade::vx::network::SynapseDriverOnDLSManager::Allocation Allocation;
	typedef grenade::vx::network::SynapseDriverOnDLSManager::AllocationPolicy AllocationPolicy;

	/**
	 * Get PADI-bus collections for which overarching allocation requests exist.
	 * For example the allocation requests A{PADIBusOnDLS(1), PADIBusOnDLS(3)} and
	 * B{PADIBusOnDLS(3), PADIBusOnDLS(4)} lead to the PADI-bus collection {PADIBusOnDLS(1),
	 * PADIBusOnDLS(3), PADIBusOnDLS(4)}.
	 * @param requested_allocations Requested allocations from which to extract interdependent
	 * collections
	 */
	static std::set<std::set<halco::hicann_dls::vx::v3::PADIBusOnDLS>>
	get_interdependent_padi_busses(std::vector<AllocationRequest> const& requested_allocations)
	    SYMBOL_VISIBLE;

	/**
	 * Allocation requests ordered by PADI-bus.
	 * @tparam Key PADI-bus location
	 * @tparam Value Vector of single allocation requests and indices in the complete list of
	 * requested allocations
	 */
	typedef std::map<
	    halco::hicann_dls::vx::v3::PADIBusOnDLS,
	    std::pair<
	        std::vector<SynapseDriverOnPADIBusManager::AllocationRequest>,
	        std::vector<size_t>>>
	    AllocationRequestPerPADIBus;

	/**
	 * Get allocation requests per PADI-bus.
	 * The label value is not filled and will be set according to label indices for each allocation
	 * trial.
	 * @param requested_allocations Requested allocations
	 * @return Requested allocations ordered by PADI-busses
	 */
	static AllocationRequestPerPADIBus get_requested_allocations_per_padi_bus(
	    std::vector<AllocationRequest> const& requested_allocations) SYMBOL_VISIBLE;

	/**
	 * Get allocation requests on the specified PADI-busses, which are not part of a dependent label
	 * group.
	 * @param requested_allocations Requested allocations
	 * @param padi_busses PADI-bus collection to filter requests for
	 * @return List of indices in requested_allocations
	 */
	static std::vector<size_t> get_independent_allocation_requests(
	    std::vector<AllocationRequest> const& requested_allocations,
	    std::set<halco::hicann_dls::vx::v3::PADIBusOnDLS> const& padi_busses) SYMBOL_VISIBLE;

	/**
	 * Get unique dependent label groups on specified PADI-busses.
	 * @param requested_allocations Requested allocations
	 * @param padi_busses PADI-bus collections to filter request for
	 */
	static std::vector<AllocationRequest::DependentLabelGroup> get_unique_dependent_label_groups(
	    std::vector<AllocationRequest> const& requested_allocations,
	    std::set<halco::hicann_dls::vx::v3::PADIBusOnDLS> const& padi_busses) SYMBOL_VISIBLE;

	/**
	 * Get label space.
	 * The independent allocation requests are placed first in the label space, the dependent label
	 * groups are placed afterwards.
	 * @param independent_allocation_requests Independent allocation request indices
	 * @param unique_dependent_label_groups Unique dependent label groups
	 * @param requested_allocations Requested allocations
	 * @return Label space
	 */
	static std::vector<int64_t> get_label_space(
	    std::vector<size_t> const& independent_allocation_requests,
	    std::vector<AllocationRequest::DependentLabelGroup> const& unique_dependent_label_groups,
	    std::vector<AllocationRequest> const& requested_allocations) SYMBOL_VISIBLE;

	/**
	 * Get index in label space for specified requested allocation.
	 * @param independent_allocation_requests Independent allocation request indices
	 * @param unique_dependent_label_groups Unique dependent label groups
	 * @param requested_allocations Requested allocations
	 * @param index Index into requested allocations
	 * @return Index into label space
	 */
	static size_t get_label_space_index(
	    std::vector<size_t> const& independent_allocation_requests,
	    std::vector<AllocationRequest::DependentLabelGroup> const& unique_dependent_label_groups,
	    std::vector<AllocationRequest> const& requested_allocations,
	    size_t index) SYMBOL_VISIBLE;

	/**
	 * Update label values of PADI-bus-local requested allocations according to indices in label
	 * space. Updating instead of generation is chosen because this operation is performed many
	 * times.
	 * @param requested_allocation PADI-bus-local requested allocations
	 * @param requested_allocations All requested allocations
	 * @param label_indices Indices in label space for lookup of label values
	 * @param independent_allocation_requests Independent allocation requests required for
	 * translation of a value in label_indices to requested allocations
	 * @param unique_dependent_label_groups Unique dependent label groups required for translation
	 * of a value in label_indices to requested allocations
	 */
	static void update_labels(
	    AllocationRequestPerPADIBus::mapped_type& requested_allocation,
	    std::vector<AllocationRequest> const& requested_allocations,
	    std::vector<int64_t> const& label_indices,
	    std::vector<size_t> const& independent_allocation_requests,
	    std::vector<AllocationRequest::DependentLabelGroup> const& unique_dependent_label_groups)
	    SYMBOL_VISIBLE;

	/**
	 * Update allocation solution for a single PADI-bus.
	 * Updating instead of generation is chosen because this operation is performed many times.
	 * @param allocation All allocation solutions
	 * @param local_allocation PADI-bus-local allocation solution
	 * @param requested_allocation PADI-bus-local requested allocations
	 * @param padi_bus PADI-bus for which solution is given
	 * @param requested_allocations All requested allocations
	 * @param label_indices Indices in label space for lookup of label values
	 * @param independent_allocation_requests Independent allocation requests required for
	 * translation of a value in label_indices to requested allocations
	 * @param unique_dependent_label_groups Unique dependent label groups required for translation
	 * of a value in label_indices to requested allocations
	 */
	static void update_solution(
	    std::vector<Allocation>& allocation,
	    std::vector<SynapseDriverOnPADIBusManager::Allocation> const& local_allocations,
	    AllocationRequestPerPADIBus::mapped_type const& requested_allocation,
	    halco::hicann_dls::vx::v3::PADIBusOnDLS const& padi_bus,
	    std::vector<AllocationRequest> const& requested_allocations,
	    std::vector<int64_t> const& label_indices,
	    std::vector<size_t> const& independent_allocation_requests,
	    std::vector<AllocationRequest::DependentLabelGroup> const& unique_dependent_label_groups)
	    SYMBOL_VISIBLE;

	/**
	 * Check whether the solution is valid with regard to the request.
	 * @param allocations All allocation solutions
	 * @param requested_allocations All requested allocations
	 * @return Boolean value
	 */
	static bool valid_solution(
	    std::vector<Allocation> const& allocations,
	    std::vector<AllocationRequest> const& requested_allocations) SYMBOL_VISIBLE;
};

} // namespace grenade::vx::network::detail
