#pragma once
#include "grenade/vx/genpybind.h"
#include "grenade/vx/io_data_map.h"
#include "grenade/vx/network/network.h"
#include "grenade/vx/network/network_graph.h"
#include "halco/hicann-dls/vx/v3/neuron.h"
#include "haldls/vx/v3/systime.h"

#if defined(__GENPYBIND__) || defined(__GENPYBIND_GENERATED__)
#include "hate/timer.h"
#include <log4cxx/logger.h>
#endif

namespace grenade::vx GENPYBIND_TAG_GRENADE_VX {

namespace network {

/**
 * Extract spikes corresponding to neurons in the network.
 * Spikes which don't correspond to a neuron in the network are ignored.
 * @param data Data containing spike labels
 * @param network_graph Network graph to use for matching of spike labels to neurons
 * @return Time-series neuron spike data per batch entry
 */
std::vector<
    std::map<halco::hicann_dls::vx::v3::AtomicNeuronOnDLS, std::vector<haldls::vx::v3::ChipTime>>>
extract_neuron_spikes(IODataMap const& data, NetworkGraph const& network_graph) SYMBOL_VISIBLE;


/**
 * Extract MADC samples to be recorded for a network.
 * @param data Data containing MADC samples
 * @param network_graph Network graph to use for vertex descriptor lookup of the MADC samples
 * @return Time-series MADC sample data per batch entry
 */
std::vector<
    std::vector<std::pair<haldls::vx::v3::ChipTime, haldls::vx::v3::MADCSampleFromChip::Value>>>
extract_madc_samples(IODataMap const& data, NetworkGraph const& network_graph) SYMBOL_VISIBLE;


/**
 * Extract CADC samples to be recorded for a network.
 * @param data Data containing CADC samples
 * @param network_graph Network graph to use for vertex descriptor lookup of the CADC samples
 * @return Time-series CADC sample data per batch entry. Samples are sorted by their ChipTime per
 * batch-entry and contain their corresponding AtomicNeuronOnDLS location alongside the ADC value.
 */
std::vector<std::vector<
    std::tuple<haldls::vx::v3::ChipTime, halco::hicann_dls::vx::v3::AtomicNeuronOnDLS, Int8>>>
extract_cadc_samples(IODataMap const& data, NetworkGraph const& network_graph) SYMBOL_VISIBLE;


/**
 * Extract to be recorded observable data of a plasticity rule.
 * @param data Data containing observables
 * @param network_graph Network graph to use for vertex descriptor lookup of the observables
 * @param descriptor Descriptor to plasticity rule to extract observable data for
 * @return Observable data per batch entry
 */
PlasticityRule::RecordingData GENPYBIND(visible) extract_plasticity_rule_recording_data(
    IODataMap const& data,
    NetworkGraph const& network_graph,
    PlasticityRuleDescriptor descriptor) SYMBOL_VISIBLE;


/**
 * PyNN's format expects times in floating-point ms, neurons as integer representation of the
 * AtomicNeuron enum value and MADC sample values as integer values.
 * Currently, only batch-size one is supported, i.e. one time sequence.
 */
GENPYBIND_MANUAL({
	auto const convert_ms = [](auto const t) {
		return static_cast<float>(t) / 1000. /
		       static_cast<float>(grenade::vx::TimedSpike::Time::fpga_clock_cycles_per_us);
	};
	auto const extract_neuron_spikes =
	    [convert_ms](
	        grenade::vx::IODataMap const& data,
	        grenade::vx::network::NetworkGraph const& network_graph) {
		    hate::Timer timer;
		    auto logger = log4cxx::Logger::getLogger("pygrenade.network.extract_neuron_spikes");
		    auto const spikes = grenade::vx::network::extract_neuron_spikes(data, network_graph);
		    std::vector<std::map<int, pybind11::array_t<double>>> ret(spikes.size());
		    for (size_t b = 0; b < spikes.size(); ++b) {
			    for (auto const& [neuron, times] : spikes.at(b)) {
				    pybind11::array_t<double> pytimes(static_cast<pybind11::ssize_t>(times.size()));
				    for (size_t i = 0; i < times.size(); ++i) {
					    pytimes.mutable_at(i) = convert_ms(times.at(i));
				    }
				    ret.at(b)[neuron.toEnum().value()] = pytimes;
			    }
		    }
		    LOG4CXX_TRACE(logger, "Execution duration: " << timer.print() << ".");
		    return ret;
	    };
	auto const extract_madc_samples = [convert_ms](
	                                      grenade::vx::IODataMap const& data,
	                                      grenade::vx::network::NetworkGraph const& network_graph) {
		hate::Timer timer;
		auto logger = log4cxx::Logger::getLogger("pygrenade.network.extract_madc_samples");
		auto const samples = grenade::vx::network::extract_madc_samples(data, network_graph);
		std::vector<std::pair<pybind11::array_t<float>, pybind11::array_t<int>>> ret(
		    samples.size());
		for (size_t b = 0; b < samples.size(); ++b) {
			auto const madc_samples = samples.at(b);
			pybind11::array_t<float> times(static_cast<pybind11::ssize_t>(madc_samples.size()));
			pybind11::array_t<int> values(static_cast<pybind11::ssize_t>(madc_samples.size()));
			for (size_t i = 0; i < madc_samples.size(); ++i) {
				auto const& sample = madc_samples.at(i);
				times.mutable_at(i) = convert_ms(sample.first);
				values.mutable_at(i) = sample.second.toEnum().value();
			}
			ret.at(b) = std::make_pair(times, values);
		}
		LOG4CXX_TRACE(logger, "Execution duration: " << timer.print() << ".");
		return ret;
	};
	auto const extract_cadc_samples = [convert_ms](
	                                      grenade::vx::IODataMap const& data,
	                                      grenade::vx::network::NetworkGraph const& network_graph) {
		hate::Timer timer;
		auto logger = log4cxx::Logger::getLogger("pygrenade.network.extract_cadc_samples");
		auto const samples = grenade::vx::network::extract_cadc_samples(data, network_graph);
		std::vector<
		    std::tuple<pybind11::array_t<float>, pybind11::array_t<int>, pybind11::array_t<int>>>
		    ret(samples.size());
		for (size_t b = 0; b < samples.size(); ++b) {
			auto const cadc_samples = samples.at(b);
			if (cadc_samples.empty()) {
				ret.at(b) = std::make_tuple(
				    pybind11::array_t<float>(0), pybind11::array_t<int>(0),
				    pybind11::array_t<int>(0));
				continue;
			}
			pybind11::array_t<float> times(static_cast<pybind11::ssize_t>(cadc_samples.size()));
			pybind11::array_t<int> neurons(static_cast<pybind11::ssize_t>(cadc_samples.size()));
			pybind11::array_t<int> values(static_cast<pybind11::ssize_t>(cadc_samples.size()));
			for (size_t i = 0; i < cadc_samples.size(); ++i) {
				auto const& sample = cadc_samples.at(i);
				times.mutable_at(i) = convert_ms(std::get<0>(sample));
				neurons.mutable_at(i) = std::get<1>(sample).toEnum().value();
				values.mutable_at(i) = std::get<2>(sample).value();
			}
			ret.at(b) = std::make_tuple(times, neurons, values);
		}
		LOG4CXX_TRACE(logger, "Execution duration: " << timer.print() << ".");
		return ret;
	};
	parent.def(
	    "extract_neuron_spikes", extract_neuron_spikes, pybind11::arg("data"),
	    pybind11::arg("network_graph"));
	parent.def(
	    "extract_madc_samples", extract_madc_samples, pybind11::arg("data"),
	    pybind11::arg("network_graph"));
	parent.def(
	    "extract_cadc_samples", extract_cadc_samples, pybind11::arg("data"),
	    pybind11::arg("network_graph"));
})

} // namespace network

} // namespace grenade::vx
