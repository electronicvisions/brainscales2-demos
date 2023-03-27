#pragma once
#include "grenade/vx/connection_type.h"
#include "grenade/vx/io_data_map.h"
#include "grenade/vx/port.h"
#include "hate/visibility.h"
#include <memory>
#include <ostream>
#include <stddef.h>
#include <variant>
#include <vector>

namespace cereal {
struct access;
} // namespace cereal

namespace grenade::vx::vertex {

/**
 * Formatted data input from memory.
 */
struct Transformation
{
	constexpr static bool can_connect_different_execution_instances = true;

	/**
	 * Function base for transforming a single input value to a single output value.
	 */
	struct SYMBOL_VISIBLE Function
	{
		typedef IODataMap::Entry Value;

		virtual ~Function() = 0;

		/**
		 * Provided input ports provided.
		 * @return Port
		 */
		virtual std::vector<Port> inputs() const = 0;

		/**
		 * Single output port provided.
		 * @return Port
		 */
		virtual Port output() const = 0;

		/**
		 * Apply function on input value.
		 * @param value Input value
		 * @return Transformed output value
		 */
		virtual Value apply(std::vector<Value> const& value) const = 0;

		virtual bool equal(Function const& other) const = 0;
	};

	Transformation() = default;

	/**
	 * Construct Transformation with specified function.
	 * @param function Function to apply on transformation
	 */
	explicit Transformation(std::unique_ptr<Function> function) SYMBOL_VISIBLE;

	constexpr static bool variadic_input = false;
	std::vector<Port> inputs() const SYMBOL_VISIBLE;
	Port output() const SYMBOL_VISIBLE;

	/**
	 * Apply transformation.
	 * @param value Input value
	 * @return Output value
	 */
	Function::Value apply(std::vector<Function::Value> const& value) const SYMBOL_VISIBLE;

	friend std::ostream& operator<<(std::ostream& os, Transformation const& config) SYMBOL_VISIBLE;

	bool operator==(Transformation const& other) const SYMBOL_VISIBLE;
	bool operator!=(Transformation const& other) const SYMBOL_VISIBLE;

private:
	std::unique_ptr<Function> m_function{};

	friend struct cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // grenade::vx::vertex
