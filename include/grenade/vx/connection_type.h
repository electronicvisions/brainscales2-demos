#pragma once
#include "hate/visibility.h"
#include <array>
#include <iosfwd>

namespace grenade::vx {

/**
 * Type of data transfered by a connection.
 * Connections between vertices are only allowed if the input vertex output connection type is equal
 * to the output vertex input connection type.
 */
enum class ConnectionType
{
	SynapseInputLabel,                   // PADI payload, 5b in ML, 6b in SNN
	UInt32,                              // ArgMax output index
	UInt5,                               // HAGEN input activation
	Int8,                                // CADC readout, PPU operation
	SynapticInput,                       // Accumulated (analog) synaptic input for a neuron
	MembraneVoltage,                     // Neuron membrane voltage for input of CADC readout
	TimedSpikeSequence,                  // Spike sequence to chip
	TimedSpikeFromChipSequence,          // Spike sequence from chip
	TimedMADCSampleFromChipSequence,     // MADC sample sequence from chip
	DataTimedSpikeSequence,              // Spike sequence to chip data
	DataTimedSpikeFromChipSequence,      // Spike sequence from chip data
	DataTimedMADCSampleFromChipSequence, // MADC sample sequence from chip data
	DataInt8,                            // PPU computation or CADC readout value
	DataUInt5,                           // HAGEN input activation data
	DataUInt32,                          // Index data output
	CrossbarInputLabel,                  // 14Bit label into crossbar
	CrossbarOutputLabel,                 // 14Bit label out of crossbar
	SynapseDriverInputLabel
};

std::ostream& operator<<(std::ostream& os, ConnectionType const& type) SYMBOL_VISIBLE;

/**
 * Only memory operations are allowed to connect between different execution instances.
 */
constexpr auto can_connect_different_execution_instances =
    std::array{ConnectionType::DataUInt5,
               ConnectionType::DataInt8,
               ConnectionType::DataTimedSpikeSequence,
               ConnectionType::DataTimedSpikeFromChipSequence,
               ConnectionType::DataUInt32,
               ConnectionType::DataTimedMADCSampleFromChipSequence};

} // namespace grenade::vx
