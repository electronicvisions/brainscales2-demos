#pragma once
#include "grenade/vx/event.h"
#include "grenade/vx/execution_instance.h"
#include "grenade/vx/execution_time_info.h"
#include "grenade/vx/genpybind.h"
#include "grenade/vx/graph_representation.h"
#include "grenade/vx/port.h"
#include "grenade/vx/types.h"
#include "haldls/vx/v3/timer.h"
#include "hate/visibility.h"
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <variant>

namespace grenade::vx GENPYBIND_TAG_GRENADE_VX {

/**
 * Data map used for external data exchange in graph execution.
 * For each type of data a separate member allows access.
 */
struct GENPYBIND(visible) IODataMap
{
	typedef std::variant<
	    std::vector<TimedDataSequence<std::vector<UInt32>>>,
	    std::vector<TimedDataSequence<std::vector<UInt5>>>,
	    std::vector<TimedDataSequence<std::vector<Int8>>>,
	    std::vector<TimedSpikeSequence>,
	    std::vector<TimedSpikeFromChipSequence>,
	    std::vector<TimedMADCSampleFromChipSequence>>
	    Entry;

	/**
	 * Get whether the data held in the entry match the port shape and type information.
	 * @param entry Entry to check
	 * @param port Port to check
	 * @return  Boolean value
	 */
	static bool is_match(Entry const& entry, Port const& port) SYMBOL_VISIBLE;

	/**
	 * Data is connected to specified vertex descriptors.
	 * Batch-support is enabled by storing batch-size many data elements aside each-other.
	 */
	std::map<detail::vertex_descriptor, Entry> data;

	/**
	 * Runtime time-interval data.
	 * The runtime start time coincides with the spike events' and MADC recording start time.
	 * Event data is only recorded during the runtime.
	 * If the runtime data is empty it is ignored.
	 */
	std::unordered_map<coordinate::ExecutionInstance, std::vector<haldls::vx::v3::Timer::Value>>
	    runtime;

	/**
	 * Optional time information of performed execution to be filled by executor.
	 */
	std::optional<ExecutionTimeInfo> execution_time_info;

	IODataMap() SYMBOL_VISIBLE;

	IODataMap(IODataMap const&) = delete;

	IODataMap(IODataMap&& other) SYMBOL_VISIBLE;

	IODataMap& operator=(IODataMap&& other) SYMBOL_VISIBLE GENPYBIND(hidden);

	/**
	 * Merge other map content into this one's.
	 * @param other Other map to merge into this instance
	 */
	void merge(IODataMap&& other) SYMBOL_VISIBLE GENPYBIND(hidden);

	/**
	 * Merge other map content into this one's.
	 * @param other Other map to merge into this instance
	 */
	void merge(IODataMap& other) SYMBOL_VISIBLE;

	/**
	 * Clear content of map.
	 */
	void clear() SYMBOL_VISIBLE;

	/**
	 * Get whether the map does not contain any elements.
	 * @return Boolean value
	 */
	bool empty() const SYMBOL_VISIBLE;

	/**
	 * Get number of elements in each batch of data.
	 * @return Number of elements in batch
	 */
	size_t batch_size() const SYMBOL_VISIBLE;

	/**
	 * Check that all map entries feature the same batch_size value.
	 * @return Boolean value
	 */
	bool valid() const SYMBOL_VISIBLE;

private:
	/**
	 * Mutex guarding mutable operation merge() and clear().
	 * Mutable access to map content shall be guarded with this mutex.
	 */
	std::unique_ptr<std::mutex> mutex;
};


/**
 * Data map of constant references to data used for external data input in graph execution.
 * For each type of data a separate member allows access.
 */
struct ConstantReferenceIODataMap
{
	typedef IODataMap::Entry Entry;

	/**
	 * Data is connected to specified vertex descriptors.
	 * Batch-support is enabled by storing batch-size many data elements aside each-other.
	 */
	std::map<detail::vertex_descriptor, Entry const&> data;

	/** Runtime data. */
	std::unordered_map<coordinate::ExecutionInstance, std::vector<haldls::vx::v3::Timer::Value>>
	    runtime;

	ConstantReferenceIODataMap() SYMBOL_VISIBLE;

	/**
	 * Clear content of map.
	 */
	void clear() SYMBOL_VISIBLE;
};

} // namespace grenade::vx
