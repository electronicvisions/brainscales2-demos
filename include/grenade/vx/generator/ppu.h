#pragma once
#include "grenade/vx/ppu/detail/status.h"
#include "halco/hicann-dls/vx/v3/ppu.h"
#include "hate/nil.h"
#include "hate/visibility.h"
#include "stadls/vx/v3/playback_generator.h"

namespace grenade::vx::generator {

/**
 * Generator for a playback program snippet from inserting a PPU command and blocking until
 * completion.
 */
struct BlockingPPUCommand
{
	typedef hate::Nil Result;
	typedef stadls::vx::v3::PlaybackProgramBuilder Builder;

	/**
	 * Construct blocking PPU command.
	 * @param coord PPU memory location at chich to place the command and to poll
	 * @param status Command to place
	 */
	BlockingPPUCommand(
	    halco::hicann_dls::vx::v3::PPUMemoryWordOnPPU const& coord,
	    grenade::vx::ppu::detail::Status const status) :
	    m_coord(coord), m_status(status)
	{}

protected:
	stadls::vx::v3::PlaybackGeneratorReturn<Result> generate() const SYMBOL_VISIBLE;

	friend auto stadls::vx::generate<BlockingPPUCommand>(BlockingPPUCommand const&);

private:
	halco::hicann_dls::vx::v3::PPUMemoryWordOnPPU m_coord;
	grenade::vx::ppu::detail::Status m_status;
};

} // namespace grenade::vx::generator
