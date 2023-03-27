#pragma once
#include "grenade/cerealization.h"
#include "grenade/vx/compute/addition.h"
#include "grenade/vx/compute/argmax.h"
#include "grenade/vx/compute/conv1d.h"
#include "grenade/vx/compute/converting_relu.h"
#include "grenade/vx/compute/mac.h"
#include "grenade/vx/compute/relu.h"
#include "grenade/vx/io_data_list.h"
#include "hate/visibility.h"
#include <list>
#include <variant>

namespace cereal {
struct access;
} // namespace cereal

namespace lola::vx::v3 {
class Chip;
} // namespace lola::vx::v3

namespace grenade::vx {

class JITGraphExecutor;

namespace compute {

struct Sequence
{
	typedef std::variant<Addition, ArgMax, Conv1d, MAC, ReLU, ConvertingReLU> Entry;
	/**
	 * Vectorized number data for input and output without timing information.
	 * Outer dimension: batch-entries
	 * Inner dimension: values per batch entry
	 */
	typedef std::variant<
	    std::vector<std::vector<UInt5>>,
	    std::vector<std::vector<Int8>>,
	    std::vector<std::vector<UInt32>>>
	    IOData;

	std::list<Entry> data;

	Sequence() = default;

	IOData run(IOData const& input, lola::vx::v3::Chip const& config, JITGraphExecutor& executor)
	    SYMBOL_VISIBLE;

private:
	friend struct cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // namespace compute

} // namespace grenade::vx

EXTERN_INSTANTIATE_CEREAL_SERIALIZE(grenade::vx::compute::Sequence)
