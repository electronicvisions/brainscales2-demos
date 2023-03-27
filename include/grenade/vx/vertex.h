#pragma once
#include "grenade/vx/vertex/addition.h"
#include "grenade/vx/vertex/argmax.h"
#include "grenade/vx/vertex/background_spike_source.h"
#include "grenade/vx/vertex/cadc_membrane_readout_view.h"
#include "grenade/vx/vertex/converting_relu.h"
#include "grenade/vx/vertex/crossbar_l2_input.h"
#include "grenade/vx/vertex/crossbar_l2_output.h"
#include "grenade/vx/vertex/crossbar_node.h"
#include "grenade/vx/vertex/data_input.h"
#include "grenade/vx/vertex/data_output.h"
#include "grenade/vx/vertex/external_input.h"
#include "grenade/vx/vertex/madc_readout.h"
#include "grenade/vx/vertex/neuron_event_output_view.h"
#include "grenade/vx/vertex/neuron_view.h"
#include "grenade/vx/vertex/padi_bus.h"
#include "grenade/vx/vertex/plasticity_rule.h"
#include "grenade/vx/vertex/relu.h"
#include "grenade/vx/vertex/subtraction.h"
#include "grenade/vx/vertex/synapse_array_view.h"
#include "grenade/vx/vertex/synapse_array_view_sparse.h"
#include "grenade/vx/vertex/synapse_driver.h"
#include "grenade/vx/vertex/transformation.h"
#include "grenade/vx/vertex_concept.h"
#include <variant>

namespace grenade::vx {

/** Vertex configuration as variant over possible types. */
typedef std::variant<
    vertex::Subtraction,
    vertex::PlasticityRule,
    vertex::BackgroundSpikeSource,
    vertex::ArgMax,
    vertex::CrossbarL2Input,
    vertex::CrossbarL2Output,
    vertex::CrossbarNode,
    vertex::PADIBus,
    vertex::SynapseDriver,
    vertex::SynapseArrayViewSparse,
    vertex::SynapseArrayView,
    vertex::ConvertingReLU,
    vertex::ReLU,
    vertex::Addition,
    vertex::ExternalInput,
    vertex::DataInput,
    vertex::DataOutput,
    vertex::Transformation,
    vertex::NeuronView,
    vertex::NeuronEventOutputView,
    vertex::MADCReadoutView,
    vertex::CADCMembraneReadoutView>
    Vertex;

static_assert(
    sizeof(detail::CheckVertexConcept<Vertex>), "Vertices don't adhere to VertexConcept.");

} // namespace grenade::vx
