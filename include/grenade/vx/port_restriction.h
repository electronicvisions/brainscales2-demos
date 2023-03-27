#pragma once
#include "hate/visibility.h"
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <optional>

namespace cereal {
struct access;
} // namespace cereal

namespace grenade::vx {

struct Port;

/**
 * Interval restriction of a port.
 * This allows connecting a subset of a vertex's output to another vertex's input.
 */
struct PortRestriction
{
	PortRestriction() = default;
	PortRestriction(size_t min, size_t max) SYMBOL_VISIBLE;

	bool operator==(PortRestriction const& other) const SYMBOL_VISIBLE;
	bool operator!=(PortRestriction const& other) const SYMBOL_VISIBLE;

	friend std::ostream& operator<<(std::ostream& os, PortRestriction const& data) SYMBOL_VISIBLE;

	/**
	 * Get whether this port restriction is a valid restriction of the given port.
	 * This is the case exactly if the range [min, max] is within the port's range [0, size).
	 * @param port Port to check
	 * @return Boolean validity value
	 */
	bool is_restriction_of(Port const& port) const SYMBOL_VISIBLE;

	/**
	 * Get size of port restriction range [min, max].
	 * @return Size value
	 */
	size_t size() const SYMBOL_VISIBLE;

	/**
	 * Get minimal inclusive index.
	 * @return Minimum of range
	 */
	size_t min() const SYMBOL_VISIBLE;

	/**
	 * Get maximal inclusive index.
	 * @return Maximum of range
	 */
	size_t max() const SYMBOL_VISIBLE;

private:
	size_t m_min{0};
	size_t m_max{0};

	friend struct cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // namespace grenade::vx
