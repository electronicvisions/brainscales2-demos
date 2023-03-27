#include "grenade/vx/network/detail/source_on_padi_bus_manager.h"

#include "grenade/vx/network/projection.h"
#include "halco/common/iter_all.h"
#include "halco/hicann-dls/vx/v3/neuron.h"
#include "halco/hicann-dls/vx/v3/synapse.h"
#include "hate/math.h"
#include "lola/vx/v3/synapse.h"
#include <numeric>

namespace grenade::vx::network::detail {

using namespace halco::common;
using namespace halco::hicann_dls::vx::v3;

template <typename S>
typed_array<size_t, PADIBusBlockOnDLS> SourceOnPADIBusManager::get_num_synapse_drivers(
    std::vector<S> const& sources, std::vector<size_t> const& filter)
{
	std::map<Projection::ReceptorType, halco::common::typed_array<size_t, AtomicNeuronOnDLS>>
	    in_degree;
	in_degree[Projection::ReceptorType::excitatory].fill(0);
	in_degree[Projection::ReceptorType::inhibitory].fill(0);

	for (auto const& f : filter) {
		for (auto const& [receptor_type, out_degree] : sources.at(f).out_degree) {
			for (auto const an : iter_all<AtomicNeuronOnDLS>()) {
				in_degree[receptor_type][an] += out_degree[an];
			}
		}
	}

	std::map<Projection::ReceptorType, typed_array<size_t, PADIBusBlockOnDLS>> num_synapse_rows;
	num_synapse_rows[Projection::ReceptorType::excitatory].fill(0);
	num_synapse_rows[Projection::ReceptorType::inhibitory].fill(0);
	for (auto const& [receptor_type, in] : in_degree) {
		for (auto const an : iter_all<AtomicNeuronOnDLS>()) {
			auto& local_num_synapse_rows =
			    num_synapse_rows[receptor_type][an.toNeuronRowOnDLS().toPADIBusBlockOnDLS()];
			local_num_synapse_rows = std::max(local_num_synapse_rows, in[an]);
		}
	}
	typed_array<size_t, PADIBusBlockOnDLS> num_synapse_drivers;
	for (auto const padi_bus : iter_all<PADIBusBlockOnDLS>()) {
		num_synapse_drivers[padi_bus] = hate::math::round_up_integer_division(
		    std::accumulate(
		        num_synapse_rows.begin(), num_synapse_rows.end(), 0,
		        [padi_bus](auto const& a, auto const& b) { return a + b.second[padi_bus]; }),
		    SynapseRowOnSynapseDriver::size);
	}
	return num_synapse_drivers;
}

} // namespace grenade::vx::network::detail
