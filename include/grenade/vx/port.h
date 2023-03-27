#pragma once
#include <cstddef>
#include <iosfwd>

#include "grenade/vx/connection_type.h"
#include "hate/visibility.h"

namespace grenade::vx {

/**
 * Description of a single data port of a vertex.
 * A port is described by the type of data to be transfered and a size of parallel transfered
 * elements.
 */
struct Port
{
	/** Number of entries. */
	size_t size;

	/** Connection type. */
	ConnectionType type;

	constexpr explicit Port(size_t const size, ConnectionType const type) : size(size), type(type)
	{}

	constexpr bool operator==(Port const& other) const
	{
		return (size == other.size) && (type == other.type);
	}

	constexpr bool operator!=(Port const& other) const
	{
		return !(*this == other);
	}

	friend std::ostream& operator<<(std::ostream& os, Port const& port) SYMBOL_VISIBLE;
};

} // grenade::vx
