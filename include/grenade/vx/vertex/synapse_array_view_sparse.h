#pragma once
#include "grenade/vx/connection_type.h"
#include "grenade/vx/port.h"
#include "halco/hicann-dls/vx/v3/synapse.h"
#include "halco/hicann-dls/vx/v3/synram.h"
#include "hate/visibility.h"
#include "lola/vx/v3/synapse.h"
#include <array>
#include <cstddef>
#include <iosfwd>
#include <vector>
#include <boost/range/iterator_range.hpp>

namespace cereal {
struct access;
} // namespace cereal

namespace grenade::vx::ppu {
struct SynapseArrayViewHandle;
} // namespace grenade::vx::ppu

namespace grenade::vx::vertex {

/**
 * A sparse view of synapses connected to a set of synapse drivers.
 */
struct SynapseArrayViewSparse
{
	constexpr static bool can_connect_different_execution_instances = false;

	typedef std::vector<halco::hicann_dls::vx::v3::SynapseRowOnSynram> Rows;
	typedef halco::hicann_dls::vx::v3::SynramOnDLS Synram;
	typedef std::vector<halco::hicann_dls::vx::v3::SynapseOnSynapseRow> Columns;

	struct Synapse
	{
		lola::vx::v3::SynapseMatrix::Label label;
		lola::vx::v3::SynapseMatrix::Weight weight;
		size_t index_row;
		size_t index_column;

		bool operator==(Synapse const& other) const SYMBOL_VISIBLE;
		bool operator!=(Synapse const& other) const SYMBOL_VISIBLE;

	private:
		friend struct cereal::access;
		template <typename Archive>
		void serialize(Archive& ar, std::uint32_t);
	};

	typedef std::vector<Synapse> Synapses;

	SynapseArrayViewSparse() = default;

	/**
	 * Construct synapse array view.
	 * @param synram Synram location of synapses
	 * @param rows Coordinates of rows
	 * @param columns Coordinates of columns
	 * @param synapses Synapse values
	 */
	template <typename SynramT, typename RowsT, typename ColumnsT, typename SynapsesT>
	explicit SynapseArrayViewSparse(
	    SynramT&& synram, RowsT&& rows, ColumnsT&& columns, SynapsesT&& synapses);

	/**
	 * Accessor to synapse row coordinates via a range.
	 * @return Range of synapse row coordinates
	 */
	boost::iterator_range<Rows::const_iterator> get_rows() const SYMBOL_VISIBLE;

	/**
	 * Accessor to synapse column coordinates via a range.
	 * @return Range of synapse column coordinates
	 */
	boost::iterator_range<Columns::const_iterator> get_columns() const SYMBOL_VISIBLE;

	/**
	 * Accessor to synapse configuration via a range.
	 * @return Range of synapse configuration
	 */
	boost::iterator_range<Synapses::const_iterator> get_synapses() const SYMBOL_VISIBLE;

	Synram const& get_synram() const SYMBOL_VISIBLE;

	/**
	 * Convert to synapse array view handle for PPU programs.
	 */
	ppu::SynapseArrayViewHandle toSynapseArrayViewHandle() const SYMBOL_VISIBLE;

	constexpr static bool variadic_input = false;
	std::vector<Port> inputs() const SYMBOL_VISIBLE;

	Port output() const SYMBOL_VISIBLE;

	friend std::ostream& operator<<(std::ostream& os, SynapseArrayViewSparse const& config)
	    SYMBOL_VISIBLE;

	bool operator==(SynapseArrayViewSparse const& other) const SYMBOL_VISIBLE;
	bool operator!=(SynapseArrayViewSparse const& other) const SYMBOL_VISIBLE;

private:
	Synram m_synram{};
	Rows m_rows{};
	Columns m_columns{};
	Synapses m_synapses{};

	void check(Rows const& rows, Columns const& columns, Synapses const& synapses) SYMBOL_VISIBLE;

	friend struct cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // grenade::vx::vertex

#include "grenade/vx/vertex/synapse_array_view_sparse.tcc"
