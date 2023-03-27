# Routing abstraction for logical networks

The `logical_network` namespace contains facilities to describe a `Network` of placed `Population`s of logical neurons, unplaced and unrouted `Projection`s of connections corresponding to possibly multiple hardware synapse circuits in-between compartments of neurons in these populations, `MADCRecording` on a single atomic neuron, `CADCRecording` on a collection of atomic neurons as well as `PlasticityRule`s on projections.
In addition, `ExternalPopulation`s allow feeding-in external spike trains and `BackgroundSpikeSourcePopulation`s describe the on-chip background generators.

The only difference to the `network` namespace therefore is the description of logical neurons and projections in-between their neurons' compartments with possibly multiple hardware synapses per connection.

Given such a `Network`, an automated routing algorithm called via `NetworkGraph build_network_graph(Network)` solves the correspondence between logical neurons and atomic neurons as well as abstract synapses and `network::Network` hardware synapses.
It therrefore selects which atomic neurons of the compartments to connect and the distribution of hardware synapses.

The `NetworkGraph` then contains the original `Network` as well as the routed `network::Network`.
This can then be used further as described in the `network` namespace.

Spike-trains to the `ExternalPopulation`s in the network are supplied to an `InputGenerator` builder-pattern, which transforms the supplied spike-trains to the raw `IODataMap` format required by `run()`.
