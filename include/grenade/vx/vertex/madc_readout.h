#pragma once
#include "grenade/vx/connection_type.h"
#include "grenade/vx/port.h"
#include "halco/hicann-dls/vx/v3/neuron.h"
#include "hate/visibility.h"
#include "lola/vx/v3/neuron.h"
#include <array>
#include <cstddef>
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
 * Readout of neuron voltages via the MADC.
 */
struct MADCReadoutView
{
	constexpr static bool can_connect_different_execution_instances = false;

	typedef halco::hicann_dls::vx::v3::AtomicNeuronOnDLS Coord;
	typedef lola::vx::v3::AtomicNeuron::Readout::Source Config;

	MADCReadoutView() = default;

	/**
	 * Construct MADCReadoutView.
	 * @param coord Neuron to read out
	 * @param config Source to read out at neuron
	 */
	explicit MADCReadoutView(Coord const& coord, Config const& config) SYMBOL_VISIBLE;

	Coord const& get_coord() const SYMBOL_VISIBLE;
	Config const& get_config() const SYMBOL_VISIBLE;

	constexpr static bool variadic_input = false;
	std::array<Port, 1> inputs() const SYMBOL_VISIBLE;

	Port output() const SYMBOL_VISIBLE;

	friend std::ostream& operator<<(std::ostream& os, MADCReadoutView const& config) SYMBOL_VISIBLE;

	bool supports_input_from(
	    NeuronView const& input,
	    std::optional<PortRestriction> const& restriction) const SYMBOL_VISIBLE;

	bool operator==(MADCReadoutView const& other) const SYMBOL_VISIBLE;
	bool operator!=(MADCReadoutView const& other) const SYMBOL_VISIBLE;

private:
	Coord m_coord{};
	Config m_config{};

	friend struct cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // vertex

} // grenade::vx
