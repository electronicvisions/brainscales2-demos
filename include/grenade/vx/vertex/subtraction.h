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

namespace grenade::vx::vertex {

/**
 * Subtraction of multiple inputs of Int8 data type.
 */
struct Subtraction
{
	constexpr static bool can_connect_different_execution_instances = false;

	Subtraction() = default;

	/**
	 * Construct subtraction with specified size.
	 * @param size Number of data values per input
	 */
	explicit Subtraction(size_t size) SYMBOL_VISIBLE;

	constexpr static bool variadic_input = true;
	std::array<Port, 1> inputs() const SYMBOL_VISIBLE;

	Port output() const SYMBOL_VISIBLE;

	friend std::ostream& operator<<(std::ostream& os, Subtraction const& config) SYMBOL_VISIBLE;

	bool operator==(Subtraction const& other) const SYMBOL_VISIBLE;
	bool operator!=(Subtraction const& other) const SYMBOL_VISIBLE;

private:
	size_t m_size{};

	friend struct cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // grenade::vx::vertex
