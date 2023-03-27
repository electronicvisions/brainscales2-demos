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
 * ConvertingReLU of multiple inputs from Int8 to UInt5.
 */
struct ConvertingReLU
{
	constexpr static bool can_connect_different_execution_instances = false;

	ConvertingReLU() = default;

	/**
	 * Construct operation with specified size and configuration.
	 * @param size Number of data values per input
	 * @param shift Number of bits to shift after relu before clamping
	 */
	explicit ConvertingReLU(size_t size, uint32_t shift) SYMBOL_VISIBLE;

	uint32_t get_shift() const SYMBOL_VISIBLE;

	constexpr static bool variadic_input = false;
	std::array<Port, 1> inputs() const SYMBOL_VISIBLE;

	Port output() const SYMBOL_VISIBLE;

	friend std::ostream& operator<<(std::ostream& os, ConvertingReLU const& config) SYMBOL_VISIBLE;

	bool operator==(ConvertingReLU const& other) const SYMBOL_VISIBLE;
	bool operator!=(ConvertingReLU const& other) const SYMBOL_VISIBLE;

private:
	size_t m_size{};
	uint32_t m_shift{};


	friend struct cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // grenade::vx::vertex
