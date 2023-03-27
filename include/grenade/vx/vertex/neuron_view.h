#pragma once
#include "grenade/vx/connection_type.h"
#include "grenade/vx/port.h"
#include "grenade/vx/ppu/neuron_view_handle.h"
#include "halco/hicann-dls/vx/v3/neuron.h"
#include "hate/visibility.h"
#include "lola/vx/v3/neuron.h"
#include <array>
#include <cstddef>
#include <iosfwd>
#include <optional>
#include <vector>

namespace cereal {
struct access;
} // namespace cereal

namespace grenade::vx {

struct PortRestriction;

namespace vertex {

struct SynapseArrayView;

/**
 * A view of neuron circuits.
 */
struct NeuronView
{
	constexpr static bool can_connect_different_execution_instances = false;

	typedef std::vector<halco::hicann_dls::vx::v3::NeuronColumnOnDLS> Columns;
	typedef std::vector<bool> EnableResets;
	typedef halco::hicann_dls::vx::v3::NeuronRowOnDLS Row;

	struct Config
	{
		typedef lola::vx::v3::AtomicNeuron::EventRouting::Address Label;
		std::optional<Label> label;
		bool enable_reset;

		bool operator==(Config const& other) const SYMBOL_VISIBLE;
		bool operator!=(Config const& other) const SYMBOL_VISIBLE;

	private:
		friend struct cereal::access;
		template <typename Archive>
		void serialize(Archive& ar, std::uint32_t version);
	};
	typedef std::vector<Config> Configs;

	NeuronView() = default;

	/**
	 * Construct NeuronView with specified neurons.
	 * @param columns Neuron columns
	 * @param enable_resets Enable values for initial reset of the neurons
	 * @param row Neuron row
	 */
	template <typename ColumnsT, typename ConfigsT, typename RowT>
	explicit NeuronView(ColumnsT&& columns, ConfigsT&& enable_resets, RowT&& row);

	Columns const& get_columns() const SYMBOL_VISIBLE;
	Configs const& get_configs() const SYMBOL_VISIBLE;
	Row const& get_row() const SYMBOL_VISIBLE;

	/**
	 * Convert to neuron view handle for PPU programs.
	 */
	ppu::NeuronViewHandle toNeuronViewHandle() const SYMBOL_VISIBLE;

	constexpr static bool variadic_input = true;
	std::array<Port, 1> inputs() const SYMBOL_VISIBLE;

	Port output() const SYMBOL_VISIBLE;

	friend std::ostream& operator<<(std::ostream& os, NeuronView const& config) SYMBOL_VISIBLE;

	bool supports_input_from(
	    SynapseArrayView const& input,
	    std::optional<PortRestriction> const& restriction) const SYMBOL_VISIBLE;

	bool operator==(NeuronView const& other) const SYMBOL_VISIBLE;
	bool operator!=(NeuronView const& other) const SYMBOL_VISIBLE;

private:
	void check(Columns const& columns, Configs const& configs) SYMBOL_VISIBLE;

	Columns m_columns{};
	Configs m_configs{};
	Row m_row{};

	friend struct cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // vertex

} // grenade::vx

#include "grenade/vx/vertex/neuron_view.tcc"
