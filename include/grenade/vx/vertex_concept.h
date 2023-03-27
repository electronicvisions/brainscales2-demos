#pragma once
#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace grenade::vx {

struct Port;

namespace detail {

template <typename Inputs>
struct IsInputsReturn : std::false_type
{};

template <size_t N>
struct IsInputsReturn<std::array<Port, N>> : std::true_type
{};

template <>
struct IsInputsReturn<std::vector<Port>> : std::true_type
{};

} // namespace detail

/**
 * A vertex is an entity which has a defined number of input ports and one output port with defined
 * type and size.
 */
template <typename Vertex>
struct VertexConcept
{
	/* Each vertex has a single output port. */
	template <typename V, typename = void>
	struct has_output : std::false_type
	{};
	template <typename V>
	struct has_output<V, std::void_t<decltype(&V::output)>>
	{
		constexpr static bool value = std::is_same_v<decltype(&V::output), Port (V::*)() const>;
	};
	template <typename V>
	constexpr static bool has_output_v = has_output<V>::value;
	static_assert(has_output_v<Vertex>, "Vertex missing output method.");

	/*
	 * The inputs are to be iterable (`std::array` or `std::vector`) of type `Port`.
	 */
	template <typename V, typename = void>
	struct has_inputs : std::false_type
	{};
	template <typename V>
	struct has_inputs<V, std::void_t<decltype(&V::inputs)>>
	{
		constexpr static bool value =
		    detail::IsInputsReturn<decltype(std::declval<V const>().inputs())>::value;
	};
	template <typename V>
	constexpr static bool has_inputs_v = has_inputs<V>::value;
	static_assert(has_inputs_v<Vertex>, "Vertex missing inputs method.");

	/*
	 * The last element of the input ports can be variadic meaning it can be extended to an
	 * arbitrary number (including zero) of distinct ports. For this to be the case `variadic_input`
	 * has to be set to true.
	 */
	template <typename V, typename = void>
	struct has_variadic_input : std::false_type
	{};
	template <typename V>
	struct has_variadic_input<V, std::void_t<decltype(&V::variadic_input)>>
	{
		constexpr static bool value = std::is_same_v<decltype(V::variadic_input), bool const>;
	};
	template <typename V>
	constexpr static bool has_variadic_input_v = has_variadic_input<V>::value;
	static_assert(has_variadic_input_v<Vertex>, "Vertex missing variadic_input specifier.");

	/*
	 * Vertices which are allowed connections between different execution instances are to set
	 * `can_connect_different_execution_instances` to true.
	 */
	template <typename V, typename = void>
	struct has_can_connect_different_execution_instances : std::false_type
	{};
	template <typename V>
	struct has_can_connect_different_execution_instances<
	    V,
	    std::void_t<decltype(&V::can_connect_different_execution_instances)>>
	{
		constexpr static bool value =
		    std::is_same_v<decltype(V::can_connect_different_execution_instances), bool const>;
	};
	template <typename V>
	constexpr static bool has_can_connect_different_execution_instances_v =
	    has_can_connect_different_execution_instances<V>::value;
	static_assert(
	    has_can_connect_different_execution_instances_v<Vertex>,
	    "Vertex missing can_connect_different_execution_instances specifier.");

	/**
	 * In addition connections between vertices can be restricted via:
	 *     supports_input_from(InputVertexT const& input, std::optional<PortRestriction> const&)
	 * This defaults to true if it is not specialized.
	 */
};

namespace detail {

template <typename VertexVariant>
struct CheckVertexConcept;

template <typename... Vertex>
struct CheckVertexConcept<std::variant<Vertex...>> : VertexConcept<Vertex>...
{};

} // namespace detail

} // namespace grenade::vx
