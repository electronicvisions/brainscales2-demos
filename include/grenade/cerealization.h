#pragma once
#include "haldls/cerealization.tcc"

#define EXPLICIT_INSTANTIATE_CEREAL_LOAD_SAVE(CLASS_NAME)                                          \
	template SYMBOL_VISIBLE void CLASS_NAME ::load(                                                \
	    cereal::JSONInputArchive&, std::uint32_t const);                                           \
	template SYMBOL_VISIBLE void CLASS_NAME ::load(                                                \
	    cereal::PortableBinaryInputArchive&, std::uint32_t const);                                 \
	template SYMBOL_VISIBLE void CLASS_NAME ::save(                                                \
	    cereal::JSONOutputArchive&, std::uint32_t const) const;                                    \
	template SYMBOL_VISIBLE void CLASS_NAME ::save(                                                \
	    cereal::PortableBinaryOutputArchive&, std::uint32_t const) const;
