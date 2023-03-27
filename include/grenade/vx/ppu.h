#pragma once
#include "halco/common/typed_array.h"
#include "haldls/vx/v3/neuron.h"
#include "haldls/vx/v3/ppu.h"
#include "hate/visibility.h"
#include "lola/vx/v3/ppu.h"
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

namespace grenade::vx {

constexpr static size_t num_cadc_samples_in_extmem = 100;
constexpr static size_t ppu_vector_alignment = 128;

/**
 * Convert column byte values to PPUMemoryBlock.
 */
haldls::vx::v3::PPUMemoryBlock to_vector_unit_row(
    halco::common::typed_array<int8_t, halco::hicann_dls::vx::v3::NeuronColumnOnDLS> const& values)
    SYMBOL_VISIBLE;

/**
 * Convert PPUMemoryBlock to column byte values.
 */
halco::common::typed_array<int8_t, halco::hicann_dls::vx::v3::NeuronColumnOnDLS>
from_vector_unit_row(haldls::vx::v3::PPUMemoryBlock const& values) SYMBOL_VISIBLE;

/**
 * Create (and automatically delete) temporary directory.
 */
struct TemporaryDirectory
{
	TemporaryDirectory(std::string directory_template) SYMBOL_VISIBLE;
	~TemporaryDirectory() SYMBOL_VISIBLE;

	std::filesystem::path get_path() const SYMBOL_VISIBLE;

private:
	std::filesystem::path m_path;
};

/**
 * Get full path to linker file.
 * @param name Name to search for
 */
std::string get_linker_file(std::string const& name) SYMBOL_VISIBLE;

/**
 * Get include paths.
 */
std::string get_include_paths() SYMBOL_VISIBLE;

/**
 * Get library paths.
 */
std::string get_library_paths() SYMBOL_VISIBLE;

/**
 * Get full path to libnux runtime.
 * @param name Name to search for
 */
std::string get_libnux_runtime(std::string const& name) SYMBOL_VISIBLE;

/**
 * Get full path to PPU program source.
 */
std::string get_program_base_source() SYMBOL_VISIBLE;

/**
 * Compiler for PPU programs.
 */
struct Compiler
{
	static constexpr auto name = "powerpc-ppu-g++";

	Compiler() SYMBOL_VISIBLE;

	std::vector<std::string> options_before_source = {
	    "-std=gnu++17",
	    "-fdiagnostics-color=always",
	    "-O2",
	    "-g",
	    "-fno-omit-frame-pointer",
	    "-fno-strict-aliasing",
	    "-Wall",
	    "-Wextra",
	    "-pedantic",
	    "-ffreestanding",
	    "-mcpu=nux",
	    "-fno-exceptions",
	    "-fno-rtti",
	    "-fno-non-call-exceptions",
	    "-fno-common",
	    "-ffunction-sections",
	    "-fdata-sections",
	    "-fno-threadsafe-statics",
	    "-fstack-usage",
	    "-mcpu=s2pp_hx",
	    get_include_paths(),
	    "-DSYSTEM_HICANN_DLS_MINI",
	    "-DLIBNUX_TIME_RESOLUTION_SHIFT=0",
	    "-fuse-ld=bfd",
	    "-Wl,--gc-sections",
	    "-nostdlib",
	    "-T" + get_linker_file("elf32nux.x"),
	    "-Wl,--defsym=mailbox_size=4096",
	};

	std::vector<std::string> options_after_source = {
	    "-Bstatic",
	    get_library_paths(),
	    "-lgcc",
	    "-lnux_vx_v3",
	    "-lhalco_common_ppu_vx",
	    "-lhalco_hicann_dls_ppu_vx",
	    "-lhalco_hicann_dls_ppu_vx_v3",
	    "-lfisch_ppu_vx",
	    "-lhaldls_ppu_vx_v3",
	    "-lnux_vx_v3",
	    "-Wl,--whole-archive",
	    get_libnux_runtime("nux_runtime_vx_v3.o"),
	    "-Wl,--no-whole-archive",
	    "-lc",
	    "-Bdynamic",
	};

	/**
	 * Compile sources into target program.
	 */
	std::pair<lola::vx::v3::PPUElfFile::symbols_type, lola::vx::v3::PPUElfFile::Memory> compile(
	    std::vector<std::string> sources) SYMBOL_VISIBLE;
};


/**
 * Compiler with global cache of compiled programs.
 */
struct CachingCompiler : public Compiler
{
	/**
	 * Cache for compiled PPU programs.
	 * The program information is indexed by hashed compilation options and the source code supplied
	 * on top of the base program. It assumes, that the base source and included headers are not
	 * modified concurrently.
	 */
	struct ProgramCache
	{
		/**
		 * Program information comprised of the symbols and memory image.
		 */
		typedef std::pair<lola::vx::v3::PPUElfFile::symbols_type, lola::vx::v3::PPUElfFile::Memory>
		    Program;

		/**
		 * Sources used for compilation of program serving as hash source into the cache.
		 */
		struct Source
		{
			/**
			 * Compiler options before the source location specification.
			 */
			std::vector<std::string> options_before_source;
			/**
			 * Compiler options after the source location specification.
			 */
			std::vector<std::string> options_after_source;
			/**
			 * Source code additional to the constant base program sources.
			 */
			std::vector<std::string> source_codes;

			std::string sha1() const SYMBOL_VISIBLE;
		};

		std::map<std::string, Program> data;
		std::mutex data_mutex;
	};

	/**
	 * Compile sources into target program or return from cache if already compiled.
	 */
	std::pair<lola::vx::v3::PPUElfFile::symbols_type, lola::vx::v3::PPUElfFile::Memory> compile(
	    std::vector<std::string> sources) SYMBOL_VISIBLE;

private:
	static ProgramCache& get_program_cache();
};

} // namespace grenade::vx
