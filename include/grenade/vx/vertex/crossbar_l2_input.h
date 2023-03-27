#pragma once
#include "grenade/vx/connection_type.h"
#include "grenade/vx/port.h"
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
 * Input from the FPGA to the Crossbar.
 * Since the data to the individual channels is split afterwards, they are presented merged here.
 */
struct CrossbarL2Input
{
	constexpr static bool can_connect_different_execution_instances = false;

	/**
	 * Construct CrossbarL2Input.
	 */
	CrossbarL2Input() = default;

	constexpr static bool variadic_input = false;
	constexpr std::array<Port, 1> inputs() const
	{
		return {Port(1, ConnectionType::TimedSpikeSequence)};
	}

	constexpr Port output() const
	{
		return Port(1, ConnectionType::CrossbarInputLabel);
	}

	friend std::ostream& operator<<(std::ostream& os, CrossbarL2Input const& config) SYMBOL_VISIBLE;

	bool operator==(CrossbarL2Input const& other) const SYMBOL_VISIBLE;
	bool operator!=(CrossbarL2Input const& other) const SYMBOL_VISIBLE;

private:
	friend struct cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // vertex

} // grenade::vx
