#pragma once
#include "grenade/vx/connection_type.h"
#include "grenade/vx/port.h"
#include "halco/common/typed_array.h"
#include "halco/hicann-dls/vx/v3/synapse.h"
#include "halco/hicann-dls/vx/v3/synapse_driver.h"
#include "haldls/vx/v3/synapse_driver.h"
#include "hate/visibility.h"
#include <iosfwd>

namespace cereal {
struct access;
} // namespace cereal

namespace grenade::vx {

struct PortRestriction;

namespace vertex {

struct PADIBus;

/**
 * Synapse driver.
 */
struct SynapseDriver
{
	constexpr static bool can_connect_different_execution_instances = false;

	typedef halco::hicann_dls::vx::v3::SynapseDriverOnDLS Coordinate;

	struct Config
	{
		typedef haldls::vx::v3::SynapseDriverConfig::RowAddressCompareMask RowAddressCompareMask;
		RowAddressCompareMask row_address_compare_mask{};

		typedef halco::common::typed_array<
		    haldls::vx::v3::SynapseDriverConfig::RowMode,
		    halco::hicann_dls::vx::v3::SynapseRowOnSynapseDriver>
		    RowModes;
		RowModes row_modes{};

		bool enable_address_out{false};

		bool operator==(Config const& other) const SYMBOL_VISIBLE;
		bool operator!=(Config const& other) const SYMBOL_VISIBLE;

	private:
		friend struct cereal::access;
		template <typename Archive>
		void serialize(Archive& ar, std::uint32_t);
	};

	SynapseDriver() = default;

	/**
	 * Construct synapse driver at specified location with specified configuration.
	 * @param coordinate Location
	 * @param config Configuration
	 */
	SynapseDriver(Coordinate const& coordinate, Config const& config) SYMBOL_VISIBLE;

	constexpr static bool variadic_input = false;
	constexpr std::array<Port, 1> inputs() const
	{
		return {Port(1, ConnectionType::SynapseDriverInputLabel)};
	}

	constexpr Port output() const
	{
		return Port(1, ConnectionType::SynapseInputLabel);
	}

	bool supports_input_from(
	    PADIBus const& input,
	    std::optional<PortRestriction> const& restriction) const SYMBOL_VISIBLE;

	Coordinate get_coordinate() const SYMBOL_VISIBLE;
	Config get_config() const SYMBOL_VISIBLE;

	friend std::ostream& operator<<(std::ostream& os, SynapseDriver const& config) SYMBOL_VISIBLE;

	bool operator==(SynapseDriver const& other) const SYMBOL_VISIBLE;
	bool operator!=(SynapseDriver const& other) const SYMBOL_VISIBLE;

private:
	Coordinate m_coordinate{};
	Config m_config{};

	friend struct cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // vertex

} // grenade::vx
