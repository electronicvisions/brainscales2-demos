#pragma once
#include "grenade/vx/genpybind.h"
#include "stadls/vx/v3/playback_program_builder.h"

namespace grenade::vx GENPYBIND_TAG_GRENADE_VX {

/**
 * Playback program hooks for an execution instance.
 */
struct GENPYBIND(visible) ExecutionInstancePlaybackHooks
{
	ExecutionInstancePlaybackHooks() = default;
	ExecutionInstancePlaybackHooks(
	    stadls::vx::v3::PlaybackProgramBuilder& pre_static_config,
	    stadls::vx::v3::PlaybackProgramBuilder& pre_realtime,
	    stadls::vx::v3::PlaybackProgramBuilder& inside_realtime_begin,
	    stadls::vx::v3::PlaybackProgramBuilder& inside_realtime_end,
	    stadls::vx::v3::PlaybackProgramBuilder& post_realtime) SYMBOL_VISIBLE;
	ExecutionInstancePlaybackHooks(ExecutionInstancePlaybackHooks const&) = delete;
	ExecutionInstancePlaybackHooks(ExecutionInstancePlaybackHooks&&) = default;
	ExecutionInstancePlaybackHooks& operator=(ExecutionInstancePlaybackHooks const&) = delete;
	ExecutionInstancePlaybackHooks& operator=(ExecutionInstancePlaybackHooks&&) = default;

	stadls::vx::v3::PlaybackProgramBuilder GENPYBIND(hidden) pre_static_config;
	stadls::vx::v3::PlaybackProgramBuilder GENPYBIND(hidden) pre_realtime;
	stadls::vx::v3::PlaybackProgramBuilder GENPYBIND(hidden) inside_realtime_begin;
	stadls::vx::v3::PlaybackProgramBuilder GENPYBIND(hidden) inside_realtime_end;
	stadls::vx::v3::PlaybackProgramBuilder GENPYBIND(hidden) post_realtime;
};

} // namespace grenade::vx
