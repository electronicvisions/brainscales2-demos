#pragma once
#include "grenade/vx/event.h"
#include "grenade/vx/genpybind.h"
#include "grenade/vx/io_data_map.h"
#include "grenade/vx/logical_network/network_graph.h"
#include "grenade/vx/network/generate_input.h"
#include "grenade/vx/network/network_graph.h"

namespace grenade::vx GENPYBIND_TAG_GRENADE_VX {

namespace logical_network GENPYBIND_MODULE {

class GENPYBIND(visible) InputGenerator
{
public:
	InputGenerator(
	    NetworkGraph const& network_graph,
	    network::NetworkGraph const& hardware_network_graph,
	    size_t batch_size = 1) SYMBOL_VISIBLE;

	void add(std::vector<TimedSpike::Time> const& times, PopulationDescriptor population)
	    SYMBOL_VISIBLE;

	void add(
	    std::vector<std::vector<TimedSpike::Time>> const& times,
	    PopulationDescriptor population) SYMBOL_VISIBLE;

	void add(
	    std::vector<std::vector<std::vector<TimedSpike::Time>>> const& times,
	    PopulationDescriptor population) SYMBOL_VISIBLE;

	GENPYBIND_MANUAL({
		auto const convert_ms = [](float const t) {
			return grenade::vx::TimedSpike::Time(
			    std::llround(t * 1000. * grenade::vx::TimedSpike::Time::fpga_clock_cycles_per_us));
		};
		parent.def(
		    "add",
		    [convert_ms](
		        GENPYBIND_PARENT_TYPE& self,
		        std::vector<std::vector<std::vector<float>>> const& times,
		        grenade::vx::logical_network::PopulationDescriptor const population) {
			    std::vector<std::vector<std::vector<grenade::vx::TimedSpike::Time>>> gtimes;
			    gtimes.resize(times.size());
			    for (size_t b = 0; auto& gt_b : gtimes) {
				    gt_b.resize(times.at(b).size());
				    for (size_t i = 0; auto& gt : gt_b) {
					    gt.reserve(times.at(b).at(i).size());
					    i++;
				    }
				    for (size_t i = 0; auto& t : times.at(b)) {
					    std::transform(
					        t.begin(), t.end(), std::back_inserter(gtimes.at(b).at(i)), convert_ms);
					    i++;
				    }
				    b++;
			    }
			    self.add(gtimes, population);
		    },
		    parent->py::arg("times"), parent->py::arg("population"));
		parent.def(
		    "add",
		    [convert_ms](
		        GENPYBIND_PARENT_TYPE& self, std::vector<float> const& times,
		        grenade::vx::logical_network::PopulationDescriptor const population) {
			    std::vector<grenade::vx::TimedSpike::Time> gtimes;
			    gtimes.reserve(times.size());
			    std::transform(times.begin(), times.end(), std::back_inserter(gtimes), convert_ms);
			    self.add(gtimes, population);
		    },
		    parent->py::arg("times"), parent->py::arg("population"));
		parent.def(
		    "add",
		    [convert_ms](
		        GENPYBIND_PARENT_TYPE& self, std::vector<std::vector<float>> const& times,
		        grenade::vx::logical_network::PopulationDescriptor const population) {
			    std::vector<std::vector<grenade::vx::TimedSpike::Time>> gtimes;
			    gtimes.resize(times.size());
			    for (size_t i = 0; auto& gt : gtimes) {
				    gt.reserve(times.at(i).size());
				    i++;
			    }
			    for (size_t i = 0; auto& t : times) {
				    std::transform(
				        t.begin(), t.end(), std::back_inserter(gtimes.at(i)), convert_ms);
				    i++;
			    }
			    self.add(gtimes, population);
		    },
		    parent->py::arg("times"), parent->py::arg("population"));
	})

	IODataMap done() SYMBOL_VISIBLE;

private:
	network::InputGenerator m_input_generator;
	NetworkGraph const& m_network_graph;
};

} // namespace logical_network

} // namespace grenade::vx
