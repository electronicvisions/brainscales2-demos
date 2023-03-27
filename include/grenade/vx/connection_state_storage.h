#pragma once
#include "fisch/vx/word_access/type/omnibus.h"
#include "lola/vx/v3/chip.h"
#include "stadls/vx/v3/reinit_stack_entry.h"
#include <vector>

namespace grenade::vx {

struct ConnectionStateStorage
{
	bool enable_differential_config;
	lola::vx::v3::Chip current_config;
	std::vector<fisch::vx::word_access_type::Omnibus> current_config_words;
	stadls::vx::v3::ReinitStackEntry reinit_base;
	stadls::vx::v3::ReinitStackEntry reinit_differential;
	stadls::vx::v3::ReinitStackEntry reinit_schedule_out_replacement;
	stadls::vx::v3::ReinitStackEntry reinit_capmem_settling_wait;
	stadls::vx::v3::ReinitStackEntry reinit_trigger;
};

} // namespace grenade::vx
