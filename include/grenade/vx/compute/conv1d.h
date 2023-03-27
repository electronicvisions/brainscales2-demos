#pragma once
#include <vector>
#include <gtest/gtest_prod.h>

#include "grenade/vx/compute/mac.h"
#include "grenade/vx/types.h"
#include "haldls/vx/v3/timer.h"

namespace cereal {
struct access;
} // namespace cereal

namespace lola::vx::v3 {
class Chip;
} // namespace lola::vx::v3

namespace grenade::vx {

class JITGraphExecutor;

namespace compute {

/**
 * Compute a multiply-accumulate operation with signed weights.
 * Neurons and synapse rows are filled monotonously.
 * If more synapses are needed than fit on a single chip sequential unrolling is used.
 */
class Conv1d
{
public:
	typedef MAC::Weight Weight;

	/** Weights of shape (out_channels, in_channels, size) */
	typedef std::vector<std::vector<std::vector<Weight>>> Weights;
	/** Activations with batch as outer dimension and (in_channels, values) in row-major as inner
	 * dimension. */
	typedef std::vector<std::vector<UInt5>> Activations;

	Conv1d() = default;

	/**
	 * Create single Conv1d compute graph wrapper.
	 * @param weights Weight matrix.
	 * @param input_size Size of one input trace
	 * @param stride Stride of convolution
	 * @param num_sends Number of times a input activation is sent to the specific row
	 * @param wait_between_events Wait time between input events in FPGA cycles
	 * @param enable_loopback Enable loopback of events with statistic analysis
	 */
	template <typename WeightsT>
	Conv1d(
	    WeightsT&& weights,
	    size_t input_size,
	    size_t stride,
	    size_t num_sends = 1,
	    haldls::vx::v3::Timer::Value wait_between_events = haldls::vx::v3::Timer::Value(25),
	    bool enable_loopback = false);

	/**
	 * Run given set of activations given the weights from construction.
	 * @param inputs Input activations to use
	 * @param config Static chip configuration to be used
	 * @param executor Executor backend to use
	 * @return Resulting accumulated membrane potentials
	 */
	std::vector<std::vector<Int8>> run(
	    Activations const& inputs,
	    lola::vx::v3::Chip const& config,
	    JITGraphExecutor& executor) const SYMBOL_VISIBLE;

	size_t input_size() const SYMBOL_VISIBLE;
	size_t output_size() const SYMBOL_VISIBLE;

private:
	void build_mac(Weights&& weights) SYMBOL_VISIBLE;

	bool m_enable_loopback{false};
	size_t m_input_size{};
	size_t m_kernel_size{};
	size_t m_in_channels{};
	size_t m_out_channels{};
	size_t m_stride{};
	Graph m_graph{};

	MAC m_mac{};

	size_t m_num_sends{};
	haldls::vx::v3::Timer::Value m_wait_between_events{};

	friend struct cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // namespace compute

} // namespace grenade::vx

#include "grenade/vx/compute/conv1d.tcc"
