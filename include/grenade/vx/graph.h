#pragma once
#include "grenade/vx/execution_instance.h"
#include "grenade/vx/genpybind.h"
#include "grenade/vx/graph_representation.h"
#include "grenade/vx/port_restriction.h"
#include "grenade/vx/vertex.h"
#include "hate/visibility.h"
#include <cstddef>
#include <iosfwd>
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>
#include <boost/bimap.hpp>
#include <boost/bimap/multiset_of.hpp>
#include <boost/bimap/set_of.hpp>
#include <boost/bimap/unordered_set_of.hpp>

namespace cereal {
struct access;
} // namespace cereal

namespace hate {
class Timer;
} // namespace hate

namespace log4cxx {
class Logger;
typedef std::shared_ptr<log4cxx::Logger> LoggerPtr;
} // namespace log4cxx

namespace grenade::vx GENPYBIND_TAG_GRENADE_VX {

struct Input;

/**
 * Placed computation graph.
 *
 * A vertex represents a unit which processes data.
 * An edge represents the data flow.
 *
 * Vertices are physically and temporally placed on a specific ExecutionInstance.
 * Data flow between different execution instances is restricted to off-chip data.
 * Access to execution instance subgraphs and mapping between execution instance dependency graph
 * and the complete data-flow graph is provided.
 *
 * The dependence graph of execution instances has to be acyclic in order to be executable, while
 * a subgraph tied to a execution instance can partly be cyclic, e.g. via recurrent routing of
 * on-chip events. This is enforced by checks on addition of vertices/edges.
 */
class GENPYBIND(visible) Graph
{
public:
	typedef detail::graph_type graph_type;
	typedef detail::vertex_descriptor vertex_descriptor;
	typedef detail::edge_descriptor edge_descriptor;

	/** Edge properties indexed by edge descriptor. */
	typedef std::unordered_map<edge_descriptor, std::optional<PortRestriction>>
	    edge_property_map_type;

	/** Map between execution instance graph vertex descriptor and execution instance. */
	typedef boost::
	    bimap<vertex_descriptor, boost::bimaps::unordered_set_of<coordinate::ExecutionInstance>>
	        execution_instance_map_type;

	/**
	 * Map between descriptors of vertices and descriptor of execution instance graph vertex.
	 * An execution instance graph vertex might relate to multiple vertices.
	 */
	typedef boost::bimap<
	    boost::bimaps::set_of<vertex_descriptor>,
	    boost::bimaps::multiset_of<vertex_descriptor>>
	    vertex_descriptor_map_type;

	/** Vertex properties indexed by vertex descriptor. */
	typedef std::vector<std::shared_ptr<Vertex>> vertex_property_map_type;

	/**
	 * Construct graph.
	 * @param enable_acyclicity_check Enable check for acyclicity in execution instance graph on
	 * every add call where a connection between previously unconnected execution instances is made.
	 */
	Graph(bool enable_acyclicity_check = true) GENPYBIND(hidden) SYMBOL_VISIBLE;

	Graph(Graph const&) GENPYBIND(hidden) SYMBOL_VISIBLE;
	Graph(Graph&&) GENPYBIND(hidden) SYMBOL_VISIBLE;
	Graph& operator=(Graph const&) GENPYBIND(hidden) SYMBOL_VISIBLE;
	Graph& operator=(Graph&&) GENPYBIND(hidden) SYMBOL_VISIBLE;

	/**
	 * Add vertex on specified execution instance with specified inputs.
	 * No checks are performed against whether the section of specified chip instance is already
	 * used.
	 * TODO: We might want to have a more fancy vertex descriptor return type
	 * Perform checks for:
	 *  - connection between vertex types is allowed
	 *  - vertex inputs match provided input descriptors output
	 *  - connection does not lead to acyclicity
	 *  - connection goes forward in time
	 * @param vertex Vertex configuration
	 * @param execution_instance Execution instance to place on
	 * @param inputs Positional list input vertex descriptors (with optional port restriction)
	 * @return Vertex descriptor of added vertex
	 */
	template <typename VertexT>
	vertex_descriptor add(
	    VertexT&& vertex,
	    coordinate::ExecutionInstance execution_instance,
	    std::vector<Input> inputs) GENPYBIND(hidden);

	/**
	 * Add vertex by reference on specified execution instance with specified inputs.
	 * This is to be used to update input relations.
	 * TODO: We might want to have a more fancy vertex descriptor return type
	 * Perform checks for:
	 *  - connection between vertex types is allowed
	 *  - vertex inputs match provided input descriptors output
	 *  - connection does not lead to acyclicity
	 *  - connection goes forward in time
	 * @param vertex_reference Vertex reference configuration
	 * @param execution_instance Execution instance to place on
	 * @param inputs Positional list input vertex descriptors (with optional port restriction)
	 * @return Vertex descriptor of added vertex
	 */
	vertex_descriptor add(
	    vertex_descriptor vertex_reference,
	    coordinate::ExecutionInstance execution_instance,
	    std::vector<Input> inputs) GENPYBIND(hidden) SYMBOL_VISIBLE;

	/**
	 * Update value of vertex.
	 * Ensures, that no value is updated, which would make graph illegal, i.e. the new vertex
	 * property has to be compatible with its inputs and all outgoing vertices have to be compatible
	 * with the new property.
	 * @param vertex_reference Reference to vertex to update value for
	 * @param vertex Value to set
	 */
	template <typename VertexT>
	void update(vertex_descriptor vertex_reference, VertexT&& vertex) GENPYBIND(hidden);

	/**
	 * Update value of vertex and relocate vertex with new inputs.
	 * Ensures, that no value is updated, which would make graph illegal, i.e. the new vertex
	 * property has to be compatible with its new inputs and all outgoing vertices have to be
	 * compatible with the new property.
	 * @param vertex_reference Vertex to relocate
	 * @param inputs New inputs to use
	 */
	template <typename VertexT>
	void update_and_relocate(
	    vertex_descriptor vertex_reference, VertexT&& vertex, std::vector<Input> inputs)
	    GENPYBIND(hidden);

	/**
	 * Get constant reference to underlying graph.
	 * @return Constant reference to underlying graph
	 */
	graph_type const& get_graph() const GENPYBIND(hidden) SYMBOL_VISIBLE;

	/**
	 * Get constant reference to underlying graph of execution instances.
	 * @return Constant reference to underlying graph of execution instances
	 */
	graph_type const& get_execution_instance_graph() const GENPYBIND(hidden) SYMBOL_VISIBLE;

	/**
	 * Get constant reference to a vertex property.
	 * @param descriptor Vertex descriptor to get property for
	 * @return Constant reference to a vertex property
	 */
	Vertex const& get_vertex_property(vertex_descriptor descriptor) const
	    GENPYBIND(hidden) SYMBOL_VISIBLE;

	/**
	 * Get constant reference to edge property map.
	 * @return Constant reference to edge property map
	 */
	edge_property_map_type const& get_edge_property_map() const GENPYBIND(hidden) SYMBOL_VISIBLE;

	/**
	 * Get constant reference to vertex property map.
	 * @return Constant reference to vertex property map
	 */
	execution_instance_map_type const& get_execution_instance_map() const
	    GENPYBIND(hidden) SYMBOL_VISIBLE;

	/**
	 * Get constant reference to vertex descriptor map.
	 * @return Constant reference to vertex descriptor map
	 */
	vertex_descriptor_map_type const& get_vertex_descriptor_map() const
	    GENPYBIND(hidden) SYMBOL_VISIBLE;

	/**
	 * Get whether the underlying execution instance graph is acyclic.
	 * This is a necessary requirement for executability.
	 * @return Boolean value
	 */
	bool is_acyclic_execution_instance_graph() const GENPYBIND(hidden);

	/**
	 * Ostream operator of graph.
	 * Prints graph topology in graphviz' dot format.
	 * Vertices are in the format `Type(descriptor)`, which allows identification of referenced
	 * vertices.
	 */
	GENPYBIND(stringstream)
	friend std::ostream& operator<<(std::ostream& os, Graph const& graph) SYMBOL_VISIBLE;

	bool operator==(Graph const& other) const GENPYBIND(hidden) SYMBOL_VISIBLE;
	bool operator!=(Graph const& other) const GENPYBIND(hidden) SYMBOL_VISIBLE;

private:
	bool m_enable_acyclicity_check;
	std::unique_ptr<graph_type> m_graph;
	std::unique_ptr<graph_type> m_execution_instance_graph;
	edge_property_map_type m_edge_property_map;
	vertex_property_map_type m_vertex_property_map;
	vertex_descriptor_map_type m_vertex_descriptor_map;
	execution_instance_map_type m_execution_instance_map;
	log4cxx::LoggerPtr m_logger;

	void add_edges(
	    vertex_descriptor descriptor,
	    coordinate::ExecutionInstance const& execution_instance,
	    std::vector<Input> const& inputs) SYMBOL_VISIBLE;
	void add_log(
	    vertex_descriptor descriptor,
	    coordinate::ExecutionInstance const& execution_instance,
	    hate::Timer const& timer) SYMBOL_VISIBLE;

	template <bool VariadicInput>
	static void check_inputs_size(size_t vertex_inputs_size, size_t inputs_size);

	template <typename VertexPort, typename InputVertexPort>
	static void check_input_port(
	    VertexPort const& vertex_port,
	    InputVertexPort const& input_vertex_port,
	    std::optional<PortRestriction> const& input_vertex_port_restriction);

	template <typename Vertex, typename InputVertex>
	static void check_supports_input_from(
	    Vertex const& vertex,
	    InputVertex const& input_vertex,
	    std::optional<PortRestriction> const& input_port_restriction);

	template <typename Vertex, typename InputVertex>
	void check_execution_instances(
	    Vertex const& vertex,
	    InputVertex const& input_vertex,
	    coordinate::ExecutionInstance const& vertex_execution_instance,
	    coordinate::ExecutionInstance const& input_vertex_execution_instance);

	void check_inputs(
	    Vertex const& vertex,
	    coordinate::ExecutionInstance const& execution_instance,
	    std::vector<Input> const& inputs) SYMBOL_VISIBLE;

	void update(vertex_descriptor vertex_reference, Vertex&& vertex) SYMBOL_VISIBLE;
	void update_and_relocate(
	    vertex_descriptor vertex_reference,
	    Vertex&& vertex,
	    std::vector<Input> inputs) SYMBOL_VISIBLE;

	friend struct cereal::access;
	template <typename Archive>
	void load(Archive& ar, std::uint32_t);
	template <typename Archive>
	void save(Archive& ar, std::uint32_t) const;
};

} // namespace grenade::vx

#include "grenade/vx/graph.tcc"
