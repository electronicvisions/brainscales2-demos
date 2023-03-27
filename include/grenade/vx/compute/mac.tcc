namespace grenade::vx::compute {

template <typename WeightsT>
MAC::MAC(
    WeightsT&& weights,
    size_t num_sends,
    haldls::vx::Timer::Value wait_between_events,
    bool enable_loopback) :
    m_enable_loopback(enable_loopback),
    m_graph(false),
    m_input_vertex(),
    m_output_vertex(),
    m_weights(std::forward<Weights>(weights)),
    m_num_sends(num_sends),
    m_wait_between_events(wait_between_events)
{
	build_graph();
}

} // namespace grenade::vx::compute
