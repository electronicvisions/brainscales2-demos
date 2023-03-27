#pragma once
#include "hate/visibility.h"
#include "hxcomm/common/connection_time_info.h"
#include "hxcomm/vx/connection_variant.h"
#include "stadls/vx/run_time_info.h"
#include "stadls/vx/v3/init_generator.h"
#include "stadls/vx/v3/playback_program.h"
#include "stadls/vx/v3/reinit_stack_entry.h"
#include <optional>
#include <string>
#include <variant>

#if defined(__GENPYBIND__) || defined(__GENPYBIND_GENERATED__)
#include "pyhxcomm/common/managed_connection.h"
#endif

namespace grenade::vx::backend {

struct Connection;
stadls::vx::RunTimeInfo run(Connection&, stadls::vx::v3::PlaybackProgram&);
stadls::vx::RunTimeInfo run(Connection&, stadls::vx::v3::PlaybackProgram&&);


/**
 * Connection to hardware/simulation ensuring proper initialization.
 * In addition to the initialized connection a corresponding reinit stack is being held, which
 * requires an equal lifetime.
 */
struct Connection
{
	/** Name of connection used in Python wrapping. */
	static constexpr char name[] = "Connection";

	/** Accepted initialization generators. */
	typedef std::variant<stadls::vx::v3::ExperimentInit, stadls::vx::v3::DigitalInit> Init;

	/**
	 * Construct connection from environment and initialize with default constructed ExperimentInit.
	 */
	Connection() SYMBOL_VISIBLE;

	/**
	 * Construct connection from hxcomm connection and initialize with default constructed
	 * ExperimentInit. Ownership of the hxcomm connection is transferred to this object.
	 * @param connection Connection from hxcomm to use
	 */
	Connection(hxcomm::vx::ConnectionVariant&& connection) SYMBOL_VISIBLE;

	/**
	 * Construct connection from hxcomm connection and initialization generator.
	 * Ownership of the hxcomm connection is transferred to this object.
	 * @param connection Connection from hxcomm to use
	 * @param init Initialization to use
	 */
	Connection(hxcomm::vx::ConnectionVariant&& connection, Init const& init) SYMBOL_VISIBLE;

	/**
	 * Get time information of execution(s).
	 * @return Time information
	 */
	hxcomm::ConnectionTimeInfo get_time_info() const SYMBOL_VISIBLE;

	/**
	 * Get unique identifier from hwdb.
	 * @param hwdb_path Optional path to hwdb
	 * @return Unique identifier
	 */
	std::string get_unique_identifier(std::optional<std::string> const& hwdb_path) const
	    SYMBOL_VISIBLE;

	/**
	 * Get bitfile information.
	 * @return Bitfile info
	 */
	std::string get_bitfile_info() const SYMBOL_VISIBLE;

	/**
	 * Get server-side remote repository state information.
	 * Only non-empty for hxcomm connection being QuiggeldyConnection.
	 * @return Repository state
	 */
	std::string get_remote_repo_state() const SYMBOL_VISIBLE;

	/**
	 * Release ownership of hxcomm connection.
	 * @return Previously owned hxcomm connection
	 */
	hxcomm::vx::ConnectionVariant&& release() SYMBOL_VISIBLE;

	/**
	 * Create entry on reinit stack.
	 * @return Created reinit stack entry
	 */
	stadls::vx::v3::ReinitStackEntry create_reinit_stack_entry() SYMBOL_VISIBLE;

	/**
	 * Get whether owned hxcomm connection is QuiggeldyConnection.
	 * @return Boolean value
	 */
	bool is_quiggeldy() const SYMBOL_VISIBLE;

private:
	hxcomm::vx::ConnectionVariant m_connection;

	/**
	 * Expected highspeed link notification count in run().
	 * During initialization the notification count is expected to match the number of enabled
	 * links, afterwards, no (zero) notifications are expected.
	 */
	size_t m_expected_link_notification_count;

	/** Reinit stack entry of initialization for reapplication. */
	stadls::vx::v3::ReinitStackEntry m_init;

	friend stadls::vx::RunTimeInfo run(Connection&, stadls::vx::v3::PlaybackProgram&);
	friend stadls::vx::RunTimeInfo run(Connection&, stadls::vx::v3::PlaybackProgram&&);
};

} // namespace grenade::vx::backend

/**
 * Wrap connection to Python as context manager named `Connection`.
 */
GENPYBIND_MANUAL({
	pyhxcomm::ManagedPyBind11Helper<grenade::vx::backend::Connection> helper(
	    parent, BOOST_HANA_STRING("Connection"));
})
