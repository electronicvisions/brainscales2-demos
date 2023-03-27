#pragma once
#include <map>
#include <unordered_map>

#include "grenade/vx/backend/connection.h"
#include "grenade/vx/connection_state_storage.h"
#include "grenade/vx/execution_instance.h"
#include "grenade/vx/execution_instance_playback_hooks.h"
#include "halco/hicann-dls/vx/v3/chip.h"
#include "hate/visibility.h"
#include "lola/vx/v3/chip.h"

#if defined(__GENPYBIND__) || defined(__GENPYBIND_GENERATED__)
#include "pyhxcomm/common/managed_connection.h"
#endif

namespace grenade::vx {

class Graph;
struct IODataMap;
class IODataList;
struct ExecutionInstancePlaybackHooks;
class JITGraphExecutor;

/**
 * Just-in-time graph executor.
 * No partial playback stream is pre-generated and nothing is reused for another run with e.g.
 * different input. This especially allows implementation without promise support, since before
 * processing each new chip instance, all necessary results are already available by value.
 */
class JITGraphExecutor
{
public:
	typedef std::tuple<bool> init_parameters_type;

	static constexpr char name[] = "JITGraphExecutor";

	typedef std::unordered_map<coordinate::ExecutionInstance, lola::vx::v3::Chip> ChipConfigs;

	typedef std::unordered_map<coordinate::ExecutionInstance, ExecutionInstancePlaybackHooks>
	    PlaybackHooks;

	/**
	 * Construct executor with active connections from environment.
	 * @param enable_differential_config Whether to enable differential configuration writes instead
	 * of full ones
	 */
	JITGraphExecutor(bool enable_differential_config = true) SYMBOL_VISIBLE;

	/**
	 * Construct executor with given active connections.
	 * @param connections Connections to acquire and provide
	 * @param enable_differential_config Whether to enable differential configuration writes instead
	 * of full ones
	 */
	JITGraphExecutor(
	    std::map<halco::hicann_dls::vx::v3::DLSGlobal, backend::Connection>&& connections,
	    bool enable_differential_config = true) SYMBOL_VISIBLE;

	/**
	 * Get identifiers of connections contained in executor.
	 * @return Set of connection identifiers
	 */
	std::set<halco::hicann_dls::vx::v3::DLSGlobal> contained_connections() const SYMBOL_VISIBLE;

	/**
	 * Release contained connections.
	 * @return Connections to the associated hardware
	 */
	std::map<halco::hicann_dls::vx::v3::DLSGlobal, backend::Connection>&& release_connections()
	    SYMBOL_VISIBLE;

	std::map<halco::hicann_dls::vx::v3::DLSGlobal, hxcomm::ConnectionTimeInfo> get_time_info() const
	    SYMBOL_VISIBLE;

	std::map<halco::hicann_dls::vx::v3::DLSGlobal, std::string> get_unique_identifier(
	    std::optional<std::string> const& hwdb_path) const SYMBOL_VISIBLE;

	std::map<halco::hicann_dls::vx::v3::DLSGlobal, std::string> get_bitfile_info() const
	    SYMBOL_VISIBLE;

	std::map<halco::hicann_dls::vx::v3::DLSGlobal, std::string> get_remote_repo_state() const
	    SYMBOL_VISIBLE;

private:
	std::map<halco::hicann_dls::vx::v3::DLSGlobal, backend::Connection> m_connections;
	std::map<halco::hicann_dls::vx::v3::DLSGlobal, ConnectionStateStorage>
	    m_connection_state_storages;
	bool m_enable_differential_config;

	/**
	 * Check whether the given graph can be executed.
	 * @param graph Graph instance
	 */
	bool is_executable_on(Graph const& graph);

	/**
	 * Check that graph can be executed.
	 * This function combines `is_executable_on` and
	 * `has_dangling_inputs`.
	 * @param graph Graph to check
	 */
	void check(Graph const& graph);

	friend IODataMap run(
	    JITGraphExecutor& executor,
	    Graph const& graph,
	    IODataMap const& input,
	    ChipConfigs const& initial_config,
	    PlaybackHooks& playback_hooks);
};


/**
 * Run the specified graph with specified inputs on the supplied executor.
 * @param executor Executor to use
 * @param graph Graph to execute
 * @param input List of input values to use
 * @param initial_config Map of initial configuration
 * @param playback_hooks Map of playback sequence collections to be inserted at specified
 * execution instances
 */
IODataMap run(
    JITGraphExecutor& executor,
    Graph const& graph,
    IODataMap const& input,
    JITGraphExecutor::ChipConfigs const& initial_config,
    JITGraphExecutor::PlaybackHooks& playback_hooks) SYMBOL_VISIBLE;

/**
 * Run the specified graph with specified inputs on the supplied executor.
 * @param executor Executor to use
 * @param graph Graph to execute
 * @param input List of input values to use
 * @param initial_config Map of initial configuration
 */
IODataMap run(
    JITGraphExecutor& executor,
    Graph const& graph,
    IODataMap const& input,
    JITGraphExecutor::ChipConfigs const& initial_config) SYMBOL_VISIBLE;

/**
 * Run the specified graph with specified inputs on the supplied executor.
 * @param executor Executor to use
 * @param graph Graph to execute
 * @param input List of input values to use
 * @param initial_config Map of initial configuration
 * @param playback_hooks List of playback sequence collections to be inserted at specified
 * execution instances
 * @param only_unconnected_output Whether to return only values to output vertices without out
 * edges
 */
IODataList run(
    JITGraphExecutor& executor,
    Graph const& graph,
    IODataList const& input,
    JITGraphExecutor::ChipConfigs const& initial_config,
    JITGraphExecutor::PlaybackHooks& playback_hooks,
    bool only_unconnected_output = true) SYMBOL_VISIBLE;

/**
 * Run the specified graph with specified inputs on the supplied executor.
 * @param executor Executor to use
 * @param graph Graph to execute
 * @param input List of input values to use
 * @param initial_config Map of initial configuration
 * @param only_unconnected_output Whether to return only values to output vertices without out
 * edges
 */
IODataList run(
    JITGraphExecutor& executor,
    Graph const& graph,
    IODataList const& input,
    JITGraphExecutor::ChipConfigs const& initial_config,
    bool only_unconnected_output = true) SYMBOL_VISIBLE;

} // namespace grenade::vx

GENPYBIND_MANUAL({
	pyhxcomm::ManagedPyBind11Helper<grenade::vx::JITGraphExecutor> helper(
	    parent, BOOST_HANA_STRING("JITGraphExecutor"));
})
