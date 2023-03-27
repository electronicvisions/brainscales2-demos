# Routing abstraction for atomic networks

The `network` namespace contains facilities to describe a `Network` of placed `Population`s of atomic neurons, unplaced and unrouted `Projection`s of single hardware synapses in-between these populations, `MADCRecording` on a single neuron, `CADCRecording` on a collection of neurons as well as `PlasticityRule`s on projections.
In addition, `ExternalPopulation`s allow feeding-in external spike trains and `BackgroundSpikeSourcePopulation`s describe the on-chip background generators.

Given such a `Network`, an automated routing algorithm called via `RoutingResult build_routing(Network)` solves the correspondence between abstract synapses and hardware entities as well as the configuration of the routing entities in the event paths.

A `NetworkGraph` can then be constructed in a canonical way containing the network and the hardware `Graph` representation of it with incorporation of the routing result via `NetworkGraph build_network_graph(Network, RoutingResult)`.
When the network changed but no new routing invocation is necessary, which can be checked via `requires_routing(Network old, Network new)`, the network graph can be updated via `update_network_graph(NetworkGraph& old, Network new)`.

The network graph can be executed via `IODataMap run(JITGraphExecutor&, Chip, NetworkGraph, IODataMap)`, where an executor is required, a chip configuration describes all non-routing-specific configuration and the supplied `IODataMap` contains the spike-trains of the external populations.

The resulting `IODataMap` contains all events recorded during the execution and other recorded data in raw format.
Extraction of the data in a network-compatible format is performed via `extract_neuron_spikes` and `extract_{m,c}adc_samples`.
Spike-trains to the `ExternalPopulation`s in the network are supplied to an `InputGenerator` builder-pattern, which transforms the supplied spike-trains to the raw `IODataMap` format required by `run()`.
