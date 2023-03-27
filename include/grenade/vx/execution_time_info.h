#pragma once
#include "grenade/vx/execution_instance.h"
#include "grenade/vx/genpybind.h"
#include "halco/hicann-dls/vx/v3/chip.h"
#include "hate/visibility.h"
#include <chrono>
#include <iosfwd>
#include <map>
#include <unordered_map>

#if defined(__GENPYBIND__) || defined(__GENPYBIND_GENERATED__)
#include <pybind11/chrono.h>
#endif

namespace grenade::vx GENPYBIND_TAG_GRENADE_VX {

struct GENPYBIND(visible) ExecutionTimeInfo
{
	/**
	 * Time spent in execution on hardware.
	 * This is the accumulated time each connection spends in execution state of executing playback
	 * program instruction streams, that is from encoding and sending instructions to receiving all
	 * responses and decoding them up to the halt response.
	 */
	std::map<halco::hicann_dls::vx::v3::DLSGlobal, std::chrono::nanoseconds>
	    execution_duration_per_hardware;

	/**
	 * Time spent in realtime section on hardware.
	 * This is the accumulated time for each execution instance of the interval [0, runtime) for
	 * each batch entry. It is equivalent to the accumulated duration of the intervals during which
	 * event recording is enabled for each batch entry.
	 */
	std::unordered_map<coordinate::ExecutionInstance, std::chrono::nanoseconds>
	    realtime_duration_per_execution_instance;

	/**
	 * Total duration of execution.
	 * This includes graph traversal, compilation of playback programs and post-processing of
	 * result. data.
	 */
	std::chrono::nanoseconds execution_duration;

	/**
	 * Merge other execution time info.
	 * This merges all map-like structures and overwrites the others.
	 * @param other Other execution time info to merge
	 */
	void merge(ExecutionTimeInfo& other) SYMBOL_VISIBLE;

	/**
	 * Merge other execution time info.
	 * This merges all map-like structures and overwrites the others.
	 * @param other Other execution time info to merge
	 */
	void merge(ExecutionTimeInfo&& other) SYMBOL_VISIBLE GENPYBIND(hidden);

	GENPYBIND(stringstream)
	friend std::ostream& operator<<(std::ostream& os, ExecutionTimeInfo const& data) SYMBOL_VISIBLE;
};

} // namespace grenade::vx
