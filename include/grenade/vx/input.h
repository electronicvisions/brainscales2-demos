#pragma once
#include "grenade/vx/graph.h"
#include "grenade/vx/port_restriction.h"
#include "hate/visibility.h"
#include <iosfwd>
#include <optional>

namespace grenade::vx {

/**
 * Input to a vertex in the data-flow graph comprised of a descriptor and an optional port
 * restriction.
 */
struct Input
{
	typedef Graph::vertex_descriptor descriptor_type;

	descriptor_type descriptor;
	std::optional<PortRestriction> port_restriction;

	Input(descriptor_type const& descriptor) :
	    descriptor(descriptor), port_restriction(std::nullopt)
	{}

	Input(descriptor_type const& descriptor, PortRestriction const& port_restriction) :
	    descriptor(descriptor), port_restriction(port_restriction)
	{}

	bool operator==(Input const& other) const SYMBOL_VISIBLE;
	bool operator!=(Input const& other) const SYMBOL_VISIBLE;

	friend std::ostream& operator<<(std::ostream& os, Input const& data) SYMBOL_VISIBLE;
};

} // namespace grenade::vx
