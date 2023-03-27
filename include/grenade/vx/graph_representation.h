#pragma once
#include <boost/graph/adjacency_list.hpp>

namespace grenade::vx::detail {

/** Bidirectional graph. */
typedef boost::adjacency_list<
    boost::vecS,
    boost::vecS,
    boost::bidirectionalS,
    boost::no_property,
    boost::no_property>
    graph_type;

typedef graph_type::vertex_descriptor vertex_descriptor;
typedef graph_type::edge_descriptor edge_descriptor;

} // namespace grenade::vx::detail

namespace std {

template <>
struct hash<grenade::vx::detail::graph_type::edge_descriptor>
{
	size_t operator()(grenade::vx::detail::graph_type::edge_descriptor const& e) const
	{
		return boost::hash<grenade::vx::detail::graph_type::edge_descriptor>{}(e);
	}
};

} // namespace std
