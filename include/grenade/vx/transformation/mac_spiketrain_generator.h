#pragma once
#include "grenade/vx/port.h"
#include "grenade/vx/vertex/transformation.h"
#include "halco/common/typed_array.h"
#include "halco/hicann-dls/vx/v3/chip.h"
#include "halco/hicann-dls/vx/v3/event.h"
#include "halco/hicann-dls/vx/v3/synapse_driver.h"
#include "haldls/vx/v3/event.h"
#include "haldls/vx/v3/timer.h"
#include <vector>
#include <gtest/gtest_prod.h>

namespace cereal {
struct access;
} // namespace cereal

class MACSpikeTrainGenerator_get_spike_label_Test;

namespace grenade::vx::transformation {

struct MACSpikeTrainGenerator : public vertex::Transformation::Function
{
	~MACSpikeTrainGenerator() SYMBOL_VISIBLE;

	MACSpikeTrainGenerator() = default;

	/**
	 * Construct spiketrain generator transformation.
	 * @param hemisphere_sizes Hemisphere sizes for which to generate spikeTrain for.
	 *        This setting corresponds to the number of inputs expected from the transformation,
	 *        where for each hemisphere size > 0 an input is expected.
	 * @param num_sends Number of times a input activation is sent to the specific row
	 * @param wait_between_events Wait time between input events in FPGA cycles
	 */
	MACSpikeTrainGenerator(
	    halco::common::typed_array<size_t, halco::hicann_dls::vx::v3::HemisphereOnDLS> const&
	        hemisphere_sizes,
	    size_t num_sends,
	    haldls::vx::v3::Timer::Value wait_between_events) SYMBOL_VISIBLE;

	std::vector<Port> inputs() const SYMBOL_VISIBLE;
	Port output() const SYMBOL_VISIBLE;

	bool equal(vertex::Transformation::Function const& other) const SYMBOL_VISIBLE;

	Value apply(std::vector<Value> const& value) const SYMBOL_VISIBLE;

private:
	FRIEND_TEST(::MACSpikeTrainGenerator, get_spike_label);

	/**
	 * Get spike label value from location and activation value.
	 * @param row Synapse driver to send to
	 * @param value Activation value to send
	 * @return SpikeLabel value if activation value is larger than zero
	 */
	static std::optional<halco::hicann_dls::vx::v3::SpikeLabel> get_spike_label(
	    halco::hicann_dls::vx::v3::SynapseDriverOnDLS const& driver,
	    UInt5 const value) SYMBOL_VISIBLE;

	halco::common::typed_array<size_t, halco::hicann_dls::vx::v3::HemisphereOnDLS>
	    m_hemisphere_sizes;
	size_t m_num_sends{};
	haldls::vx::Timer::Value m_wait_between_events{};

	friend struct cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // namespace grenade::vx::transformation
