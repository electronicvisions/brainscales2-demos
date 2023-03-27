#pragma once
#include "hate/bitset.h"
#ifdef __ppu__
#include "libnux/vx/dls.h"
#endif


namespace grenade::vx::ppu {

struct NeuronViewHandle
{
	/**
	 * Columns in neuron view.
	 * TODO: replace numbers by halco constants
	 */
	hate::bitset<256, uint32_t> columns;

#ifdef __ppu__
	/**
	 * Hemisphere location information.
	 */
	libnux::vx::PPUOnDLS hemisphere;
#endif
};

} // namespace grenade::vx::ppu
