#pragma once
#include "grenade/vx/connection_type.h"
#include "grenade/vx/port.h"
#include "halco/common/geometry.h"
#include "hate/visibility.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <vector>

namespace cereal {
struct access;
} // namespace cereal

namespace grenade::vx::vertex {

/**
 * External input data source.
 */
struct ExternalInput
{
	constexpr static bool can_connect_different_execution_instances = false;

	ExternalInput() = default;

	/**
	 * Construct external input data source with specified coordinate list.
	 * @param output_type Number of external input entries
	 * @param size Number of external input entries
	 */
	explicit ExternalInput(ConnectionType output_type, size_t size) SYMBOL_VISIBLE;

	constexpr static bool variadic_input = false;
	constexpr std::array<Port, 0> inputs() const
	{
		return {};
	}

	Port output() const SYMBOL_VISIBLE;

	friend std::ostream& operator<<(std::ostream& os, ExternalInput const& config) SYMBOL_VISIBLE;

	bool operator==(ExternalInput const& other) const SYMBOL_VISIBLE;
	bool operator!=(ExternalInput const& other) const SYMBOL_VISIBLE;

private:
	size_t m_size{};
	ConnectionType m_output_type{};

	friend struct cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // namespace grenade::vx::vertex
