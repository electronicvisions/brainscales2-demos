#pragma once
#include <cstdint>

namespace grenade::vx::ppu::detail {

enum class Status : uint32_t
{
	initial,
	idle,
	reset_neurons,
	baseline_read,
	read,
	periodic_read,
	inside_periodic_read,
	stop_periodic_read,
	stop,
	scheduler
};

} // namespace grenade::vx::ppu::detail
