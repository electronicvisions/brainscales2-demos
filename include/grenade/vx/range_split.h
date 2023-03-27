#pragma once
#include <cstddef>
#include <vector>
#include "hate/visibility.h"

namespace grenade::vx {

/**
 * Split one-dimensional range into ranges of maximal given size.
 */
class RangeSplit
{
public:
	/** Sub-range consisting of local size and offset. */
	struct SubRange
	{
		size_t size;
		size_t offset;
	};

	/**
	 * Construct range split with maximal size of ranges to split into.
	 * @param split_size Maximal size of ranges to split into
	 */
	explicit RangeSplit(size_t split_size) SYMBOL_VISIBLE;

	/**
	 * Split given range [0, size) into ranges.
	 * @param size Size of range to split
	 * @return Sub-ranges adhering to maximal size
	 */
	std::vector<SubRange> operator()(size_t size) const SYMBOL_VISIBLE;

private:
	size_t m_split_size;
};

} // namespace grenade::vx
