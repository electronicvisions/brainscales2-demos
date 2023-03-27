#include "grenade/vx/execution_instance.h"
#include "grenade/vx/input.h"
#include "hate/timer.h"

#include <memory>

namespace grenade::vx {

template <typename VertexT>
Graph::vertex_descriptor Graph::add(
    VertexT&& vertex,
    coordinate::ExecutionInstance const execution_instance,
    std::vector<Input> const inputs)
{
	hate::Timer const timer;
	// check validity of inputs with regard to vertex to be added
	if constexpr (std::is_same_v<std::decay_t<VertexT>, vertex_descriptor>) {
		check_inputs(get_vertex_property(vertex), execution_instance, inputs);
	} else {
		check_inputs(vertex, execution_instance, inputs);
	}

	// add vertex to graph
	assert(m_graph);
	auto const descriptor = boost::add_vertex(*m_graph);
	assert(descriptor == m_vertex_property_map.size());

	if constexpr (std::is_same_v<std::decay_t<VertexT>, vertex_descriptor>) {
		m_vertex_property_map.emplace_back(m_vertex_property_map.at(vertex));
	} else {
		m_vertex_property_map.emplace_back(std::make_shared<Vertex>(std::forward<VertexT>(vertex)));
	}

	// add edges
	add_edges(descriptor, execution_instance, inputs);

	// log successfull add operation of vertex
	add_log(descriptor, execution_instance, timer);

	return descriptor;
}

template <typename VertexT>
void Graph::update(vertex_descriptor const vertex_reference, VertexT&& vertex)
{
	Vertex vertex_variant(std::forward<VertexT>(vertex));
	update(vertex_reference, std::move(vertex_variant));
}

template <typename VertexT>
void Graph::update_and_relocate(
    vertex_descriptor const vertex_reference, VertexT&& vertex, std::vector<Input> inputs)
{
	Vertex vertex_variant(std::forward<VertexT>(vertex));
	update_and_relocate(vertex_reference, std::move(vertex_variant), inputs);
}

} // namespace grenade::vx
