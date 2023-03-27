#pragma once
#include "grenade/vx/connection_type.h"
#include "grenade/vx/port.h"
#include "hate/visibility.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <optional>

namespace cereal {
struct access;
} // namespace cereal

namespace grenade::vx {

struct PortRestriction;

namespace vertex {

struct CrossbarNode;

/**
 * Output from the Crossbar to the FPGA.
 * Since the data from the individual channels are merged, they are also presented merged here.
 */
struct CrossbarL2Output
{
	constexpr static bool can_connect_different_execution_instances = false;

	/**
	 * Construct CrossbarL2Output.
	 */
	CrossbarL2Output() {}

	constexpr static bool variadic_input = true;
	constexpr std::array<Port, 1> inputs() const
	{
		return {Port(1, ConnectionType::CrossbarOutputLabel)};
	}

	constexpr Port output() const
	{
		return Port(1, ConnectionType::TimedSpikeFromChipSequence);
	}

	friend std::ostream& operator<<(std::ostream& os, CrossbarL2Output const& config)
	    SYMBOL_VISIBLE;

	bool supports_input_from(
	    CrossbarNode const& input,
	    std::optional<PortRestriction> const& restriction) const SYMBOL_VISIBLE;

	bool operator==(CrossbarL2Output const& other) const SYMBOL_VISIBLE;
	bool operator!=(CrossbarL2Output const& other) const SYMBOL_VISIBLE;

private:
	friend struct cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // vertex

} // grenade::vx
