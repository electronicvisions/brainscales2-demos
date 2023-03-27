namespace grenade::vx::compute {

template <typename WeightsT>
Conv1d::Conv1d(
    WeightsT&& weights,
    size_t input_size,
    size_t stride,
    size_t num_sends,
    haldls::vx::Timer::Value wait_between_events,
    bool enable_loopback) :
    m_enable_loopback(enable_loopback),
    m_input_size(input_size),
    m_stride(stride),
    m_num_sends(num_sends),
    m_wait_between_events(wait_between_events)
{
	build_mac(std::forward<Weights>(weights));
}

} // namespace grenade::vx::compute
