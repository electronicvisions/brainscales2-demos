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
} // namespace

namespace grenade::vx::vertex {

/**
 * ArgMax of multiple inputs.
 */
struct ArgMax
{
	constexpr static bool can_connect_different_execution_instances = false;

	ArgMax() = default;

	/**
	 * Construct operation with specified size.
	 * @param size Number of data values
	 * @param type Type of data to compute argmax over
	 */
	explicit ArgMax(size_t size, ConnectionType type) SYMBOL_VISIBLE;

	constexpr static bool variadic_input = false;
	std::array<Port, 1> inputs() const SYMBOL_VISIBLE;

	Port output() const SYMBOL_VISIBLE;

	friend std::ostream& operator<<(std::ostream& os, ArgMax const& config) SYMBOL_VISIBLE;

	bool operator==(ArgMax const& other) const SYMBOL_VISIBLE;
	bool operator!=(ArgMax const& other) const SYMBOL_VISIBLE;

private:
	size_t m_size{};
	ConnectionType m_type{};

	friend struct cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // grenade::vx::vertex
