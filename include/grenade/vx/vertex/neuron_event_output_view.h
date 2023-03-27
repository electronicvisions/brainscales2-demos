#pragma once
#include "grenade/vx/connection_type.h"
#include "grenade/vx/port.h"
#include "halco/hicann-dls/vx/v3/neuron.h"
#include "hate/visibility.h"
#include <array>
#include <iosfwd>
#include <optional>

namespace cereal {
struct access;
} // namespace cereal

namespace grenade::vx {

struct PortRestriction;

namespace vertex {

struct NeuronView;

/**
 * A view of neuron event outputs into the routing crossbar.
 */
struct NeuronEventOutputView
{
	constexpr static bool can_connect_different_execution_instances = false;

	typedef std::vector<halco::hicann_dls::vx::v3::NeuronColumnOnDLS> Columns;
	typedef halco::hicann_dls::vx::v3::NeuronRowOnDLS Row;
	typedef std::map<Row, std::vector<Columns>> Neurons;

	NeuronEventOutputView() = default;

	/**
	 * Construct NeuronEventOutputView with specified neurons.
	 * @param neurons Incoming neurons
	 */
	NeuronEventOutputView(Neurons const& neurons) SYMBOL_VISIBLE;

	Neurons const& get_neurons() const SYMBOL_VISIBLE;

	constexpr static bool variadic_input = false;
	/**
	 * Input ports are organized in the same order as the specified neuron views.
	 */
	std::vector<Port> inputs() const SYMBOL_VISIBLE;

	/**
	 * Output ports are sorted neuron event output channels.
	 */
	Port output() const SYMBOL_VISIBLE;

	friend std::ostream& operator<<(std::ostream& os, NeuronEventOutputView const& config)
	    SYMBOL_VISIBLE;

	bool supports_input_from(
	    NeuronView const& input,
	    std::optional<PortRestriction> const& restriction) const SYMBOL_VISIBLE;

	bool operator==(NeuronEventOutputView const& other) const SYMBOL_VISIBLE;
	bool operator!=(NeuronEventOutputView const& other) const SYMBOL_VISIBLE;

private:
	Neurons m_neurons{};

	friend struct cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // vertex

} // grenade::vx
