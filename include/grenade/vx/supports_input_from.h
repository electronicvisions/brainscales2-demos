#pragma once
#include "hate/type_traits.h"
#include <optional>

namespace grenade::vx {

struct PortRestriction;

namespace detail {

template <typename Vertex, typename InputVertex>
using has_supports_input_from = decltype(std::declval<Vertex>().supports_input_from(
    std::declval<InputVertex>(), std::declval<std::optional<PortRestriction>>()));

} // namespace detail

/**
 * Get whether given vertex supports input from given input vertex with optional port restriction.
 * Default assumption is that the connection is supported. Restrictions can be defined by a member
 * function of signature:
 *     supports_input_from(OtherVertex const&, std::optional<PortRestriction> const&).
 * @tparam Vertex Type of vertex
 * @tparam InputVertex Type of input vertex
 * @param vertex Vertex to check
 * @param input_vertex Input vertex to check
 * @param port_restriction Optional port restriction to apply to check
 */
template <typename Vertex, typename InputVertex>
bool supports_input_from(
    Vertex const& vertex,
    InputVertex const& input,
    std::optional<PortRestriction> const& port_restriction)
{
	if constexpr (hate::is_detected_v<detail::has_supports_input_from, Vertex, InputVertex>) {
		return vertex.supports_input_from(input, port_restriction);
	}
	return true;
}

} // namespace grenade::vx
