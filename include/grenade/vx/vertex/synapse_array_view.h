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

namespace grenade::vx::vertex {

/**
 * A rectangular view of synapses connected to a set of synapse drivers.
 */
struct SynapseArrayView
{
	constexpr static bool can_connect_different_execution_instances = false;

	typedef std::vector<halco::hicann_dls::vx::v3::SynapseRowOnSynram> Rows;
	typedef halco::hicann_dls::vx::v3::SynramOnDLS Synram;
	typedef std::vector<halco::hicann_dls::vx::v3::SynapseOnSynapseRow> Columns;
	typedef std::vector<std::vector<lola::vx::v3::SynapseMatrix::Label>> Labels;
	typedef std::vector<std::vector<lola::vx::v3::SynapseMatrix::Weight>> Weights;

	SynapseArrayView() = default;

	/**
	 * Construct synapse array view.
	 * @param synram Synram location of synapses
	 * @param rows Coordinates of rows
	 * @param columns Coordinates of columns
	 * @param weights Weight values
	 * @param labels Label values
	 */
	template <
	    typename SynramT,
	    typename RowsT,
	    typename ColumnsT,
	    typename WeightsT,
	    typename LabelsT>
	explicit SynapseArrayView(
	    SynramT&& synram, RowsT&& rows, ColumnsT&& columns, WeightsT&& weights, LabelsT&& labels);

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
	 * Accessor to weight configuration via a range.
	 * @return Range of weight configuration
	 */
	boost::iterator_range<Weights::const_iterator> get_weights() const SYMBOL_VISIBLE;

	/**
	 * Accessor to label configuration via a range.
	 * @return Range of label configuration
	 */
	boost::iterator_range<Labels::const_iterator> get_labels() const SYMBOL_VISIBLE;

	Synram const& get_synram() const SYMBOL_VISIBLE;

	constexpr static bool variadic_input = false;
	std::vector<Port> inputs() const SYMBOL_VISIBLE;

	Port output() const SYMBOL_VISIBLE;

	friend std::ostream& operator<<(std::ostream& os, SynapseArrayView const& config)
	    SYMBOL_VISIBLE;

	bool operator==(SynapseArrayView const& other) const SYMBOL_VISIBLE;
	bool operator!=(SynapseArrayView const& other) const SYMBOL_VISIBLE;

private:
	Synram m_synram{};
	Rows m_rows{};
	Columns m_columns{};
	Weights m_weights{};
	Labels m_labels{};

	void check(
	    Rows const& rows, Columns const& columns, Weights const& weights, Labels const& labels)
	    SYMBOL_VISIBLE;

	friend struct cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // grenade::vx::vertex

#include "grenade/vx/vertex/synapse_array_view.tcc"
