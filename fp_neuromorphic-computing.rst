Neuromorphic computing
======================

The energy-efficiency of the computation taking place in the nervous system has long inspired researchers and engineers in building similarly efficient machines simply by adopting architectural as well as dynamical properties of the biological counterpart.
There exists a whole spectrum of neuromorphic systems with some relying on fully digital logic -- then mostly focusing on asynchronous and parallel communication -- and others utilizing analog circuits to further efficiency gains.
These different systems all focus on different aspects and target a variety of applications.
The following sections introduce some of the computing paradigms and platforms as well as a range of use cases ranging from neuroscience research to machine intelligence.

Analog computation
------------------

Neuronal computation is fundamentally based on the conservation of charge:
Neurons accumulate and integrate stimuli over space (i.e., their dendrites) and time to perform, e.g., spatio-temporal summation and coincidence detection.
This is a stark contrast to today's processors, which operate on mostly binary digital representations.
Here, any state change or computation involves charging or discharging of multiple digital states and logic circuits are typically composed of a plethora of transistors.
This observation led `Carver Mead (1990) <https://ieeexplore.ieee.org/document/58356>`_ to envision *neuromorphic electronic systems* based on simple analog circuits performing neuron-like analog computation.

Computation through structure
-----------------------------

The specific computation implemented by a neural circuit is inherently defined by the individual cell properties and -- more centrally - the network structure and synaptic strengths.
This effectively results in a *collocation of processing and memory resources* and, again, represents a contrast to current processors.
The latter almost all implement the von Neumann architecture, where program instructions as well as the data to be processed are stored in memory separate from the processing unit, inherently resulting in a frequent expensive transfer of data through the infamous *von Neumann bottleneck*.

The collocation of memory and computing resources observed in neural tissue sparked interest in *in-memory computation*, where simple computational units perform calculations on locally stored operands/parameters as well as a stream of input data.
In-memory computation typically emerges in the context of machine learning accelerators where vector-matrix multiplications dominate the overall workload and at the same time can be straight-forwardly rendered to an array of synapse-like multiply-accumulate units.
Importantly, in-memory computation differs from traditional sequential processing in that it feeds off its high intrinsic parallelism.

Event-based communication
-------------------------

The analog, time-continuous computation in a neuron is contrasted by a rather binary, event-based communication:
Upon threshold crossing, neurons generate a spike-like action potential, which propagates along the axon and is finally relayed to other neurons.
These spikes mark singular points in time and the neuron remains silent otherwise, not unlike an intrinsic zero suppression.

Spikes can encode information in their sole presence, their absolute or relative timing, or their rates averaged over time or multiple neurons.
Especially the limit of only few spikes can realize an energy-efficient communication scheme.

With its event-based communication the nervous system, furthermore, realizes an asynchronous operation.
Avoiding the need for global synchronization can optimize the energy footprint and speed of processing elements but introduces challenges with respect to reliability.
Here, the nervous system appears to have found a solution resilient to temporal jitter or even spike loss and can thus inspire further research.

.. TODO: mention plasticity and learning
