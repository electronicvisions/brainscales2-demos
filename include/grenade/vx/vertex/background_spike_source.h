#pragma once
#include "grenade/vx/connection_type.h"
#include "grenade/vx/port.h"
#include "halco/hicann-dls/vx/v3/background.h"
#include "haldls/vx/background.h"
#include "hate/visibility.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>

namespace cereal {
struct access;
} // namespace cereal

namespace grenade::vx {

struct PortRestriction;

namespace vertex {

struct ExternalInput;

/**
 * Background event source to the Crossbar.
 */
struct BackgroundSpikeSource
{
	constexpr static bool can_connect_different_execution_instances = false;

	typedef haldls::vx::BackgroundSpikeSource Config;
	typedef halco::hicann_dls::vx::BackgroundSpikeSourceOnDLS Coordinate;

	BackgroundSpikeSource() = default;

	/**
	 * Construct BackgroundSpikeSource.
	 * @param config Config to use
	 * @param coordinate Coordinate to use
	 */
	BackgroundSpikeSource(Config const& config, Coordinate const& coordinate) SYMBOL_VISIBLE;

	constexpr static bool variadic_input = false;
	constexpr std::array<Port, 0> inputs() const
	{
		return {};
	}

	constexpr Port output() const
	{
		return Port(1, ConnectionType::CrossbarInputLabel);
	}

	friend std::ostream& operator<<(std::ostream& os, BackgroundSpikeSource const& config)
	    SYMBOL_VISIBLE;

	bool operator==(BackgroundSpikeSource const& other) const SYMBOL_VISIBLE;
	bool operator!=(BackgroundSpikeSource const& other) const SYMBOL_VISIBLE;

	Config const& get_config() const SYMBOL_VISIBLE;
	Coordinate const& get_coordinate() const SYMBOL_VISIBLE;

private:
	friend struct cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);

	Config m_config;
	Coordinate m_coordinate;
};

} // vertex

} // grenade::vx
