namespace grenade::vx::vertex {

template <typename SynramT, typename RowsT, typename ColumnsT, typename SynapsesT>
SynapseArrayViewSparse::SynapseArrayViewSparse(
    SynramT&& synram, RowsT&& rows, ColumnsT&& columns, SynapsesT&& synapses) :
    m_synram(std::forward<SynramT>(synram)), m_rows(), m_columns(), m_synapses()
{
	check(rows, columns, synapses);
	m_rows = std::forward<RowsT>(rows);
	m_columns = std::forward<ColumnsT>(columns);
	m_synapses = std::forward<SynapsesT>(synapses);
}

} // namespace grenade::vx::vertex
