#pragma once
#include "grenade/vx/connection_type.h"
#include "grenade/vx/vertex/transformation.h"
#include <vector>

namespace cereal {
struct access;
} // namespace cereal

namespace grenade::vx::transformation {

struct SYMBOL_VISIBLE Concatenation : public vertex::Transformation::Function
{
	~Concatenation();

	Concatenation() = default;

	/**
	 * Construct concatenation transformation with data type and sizes to concatenate.
	 * @param type Data type
	 * @param sizes Sizes
	 */
	Concatenation(ConnectionType type, std::vector<size_t> const& sizes);

	std::vector<Port> inputs() const;
	Port output() const;

	bool equal(vertex::Transformation::Function const& other) const;

	Value apply(std::vector<Value> const& value) const;

private:
	ConnectionType m_type{};
	std::vector<size_t> m_sizes{};

	friend struct cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // namespace grenade::vx::transformation
