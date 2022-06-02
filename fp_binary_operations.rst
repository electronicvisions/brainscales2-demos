Binary operations using spiking neurons
=======================================

We want to investigate how neurons can be used to solve binary tasks.
The two states we consider are spiking and not spiking. In the following
you are asked to implement some basic logic operations.

The OR-operation
----------------

To realize the OR-operation we consider a network of three neurons.
There are two input neurons connected to one output neuron. The output
neuron is supposed to elicit a spike, if neuron 1 or neuron 2 fires.
This logic is also shown in a truth table.

.. raw:: html
    <table><tr>
    <td style="padding:0 100px 0 100px;"> <img src="_static/common/network2in.png" width="300"/> </td>

    <td style="padding:0 100px 0 100px;"> <table>
      <tr>
        <th>neuron 1</th>
        <th>neuron 2</th>
        <th>output neuron</th>
      </tr>
      <tr>
        <td style="text-align: center">-</td>
        <td style="text-align: center">-</td>
        <td style="text-align: center">-</td>
      </tr>
      <tr>
        <td style="text-align: center">x</td>
        <td style="text-align: center">-</td>
        <td style="text-align: center">x</td>
      </tr>
      <tr>
        <td style="text-align: center">-</td>
        <td style="text-align: center">x</td>
        <td style="text-align: center">x</td>
      </tr>
      <tr>
        <td style="text-align: center">x</td>
        <td style="text-align: center">x</td>
        <td style="text-align: center">x</td>
      </tr>
    </table> </td>

    </tr><table>

In order to implement this network, we need two excitatory synapses.
Their weight must be chosen such that a stimulation coming from neuron 1
or 2 is strong enough to elicit a spike in the output neuron.

.. code:: ipython3

    import pynn_brainscales.brainscales2 as pynn
    import matplotlib.pyplot as plt
    %matplotlib inline

    # set the environment
    from _static.common.helpers import setup_hardware_client, get_nightly_calibration

    setup_hardware_client()

    # load automatic calibration and set up pynn
    calib = get_nightly_calibration()
    pynn.setup(initial_config=calib)

    # create input neurons
    # these forward spikes at the times specified in `spike_time`

    ########## change here ##########
    spike_time1 = [0.2]
    spike_time2 = []
    #################################

    neuron1 = pynn.Population(1, pynn.cells.SpikeSourceArray(spike_times=spike_time1))
    neuron2 = pynn.Population(1, pynn.cells.SpikeSourceArray(spike_times=spike_time2))

    # create output neuron
    output_neuron = pynn.Population(1, pynn.cells.HXNeuron())

    # monitor its activity
    output_neuron.record(["spikes", "v"])

    # define synapses and set their weights (range: 0 - 63)

    ########## change here ##########
    synapse_weight1 = 63
    synapse_weight2 = 32
    #################################

    synapse1 = pynn.synapses.StaticSynapse(weight=synapse_weight1)
    synapse2 = pynn.synapses.StaticSynapse(weight=synapse_weight2)

    # create neuron connections
    pynn.Projection(neuron1, output_neuron, pynn.AllToAllConnector(),
                    synapse_type=synapse1, receptor_type="excitatory")
    pynn.Projection(neuron2, output_neuron, pynn.AllToAllConnector(),
                    synapse_type=synapse2, receptor_type="excitatory")

    # run the network for a specified time,
    # the duration is set in milliseconds
    duration = 0.5
    pynn.run(duration)

    # examine the spikes of the output neuron
    spiketrain = output_neuron.get_data("spikes").segments[0].spiketrains[0]
    print(f"The output neuron fired {len(spiketrain)} times.")
    print(f"The spiketimes were: {spiketrain}")

    # the membrane potential of the output neuron can be visualized, too
    mem_v = output_neuron.get_data("v").segments[0].irregularlysampledsignals[0]

    plt.figure()
    plt.plot(mem_v.times, mem_v)
    plt.xlabel("time [ms]")
    plt.ylabel("membrane potential [LSB]")
    plt.show()

    pynn.end()

Check if the network works correctly. To do so, try different
combinations of stimuli of the input neurons and different synaptic
weights. Maybe, even with maximum weight the input is not strong enough
to forward a spike. What ways can you think of to overcome this problem?
Check your hypotheses!

The AND-operation
-----------------

Now it is your turn to implement a network. The next operation we
consider is the AND-operation. Similar to the OR-operation, we again
have two input neurons and one output neuron. This time, however, the
latter should only fire exactly if neuron 1 and neuron 2 fire.

.. raw:: html
    <table><tr>
    <td style="padding:0 100px 0 100px;"> <img src="_static/common/network2in.png" width="300"/> </td>

    <td style="padding:0 100px 0 100px;"> <table>
      <tr>
        <th>neuron 1</th>
        <th>neuron 2</th>
        <th>output neuron</th>
      </tr>
      <tr>
        <td style="text-align: center">-</td>
        <td style="text-align: center">-</td>
        <td style="text-align: center">-</td>
      </tr>
      <tr>
        <td style="text-align: center">x</td>
        <td style="text-align: center">-</td>
        <td style="text-align: center">-</td>
      </tr>
      <tr>
        <td style="text-align: center">-</td>
        <td style="text-align: center">x</td>
        <td style="text-align: center">-</td>
      </tr>
      <tr>
        <td style="text-align: center">x</td>
        <td style="text-align: center">x</td>
        <td style="text-align: center">x</td>
      </tr>
    </table> </td>

    </tr><table>

Think about how the network could look like and program it
using the previous code.

.. code:: ipython3

    import pynn_brainscales.brainscales2 as pynn
    import matplotlib.pyplot as plt
    %matplotlib inline

    # set the environment
    from _static.common.helpers import setup_hardware_client, get_nightly_calibration

    setup_hardware_client()

    # load automatic calibration and set up pynn
    calib = get_nightly_calibration()
    pynn.setup(initial_config=calib)

    # your code


    pynn.end()

The NOT-operation
-----------------

Furthermore, let’s consider the NOT-operation. Here, we only have one
input neuron whose signal is to be negated by the output neuron.

.. raw:: html
    <table><tr>
    <td style="padding:0 100px 0 100px;"> <img src="_static/common/network1in.png" width="300"/> </td>

    <td style="padding:0 100px 0 100px;"> <table>
      <tr>
        <th>input neuron</th>
        <th>output neuron</th>
      </tr>
      <tr>
        <td style="text-align: center">-</td>
        <td style="text-align: center">x</td>
      </tr>
      <tr>
        <td style="text-align: center">x</td>
        <td style="text-align: center">-</td>
      </tr>
    </table> </td>

    </tr><table>

Think about how a network can look like that fulfills this task.

Hint: You will need a help-neuron.

.. code:: ipython3

    import pynn_brainscales.brainscales2 as pynn
    import matplotlib.pyplot as plt
    %matplotlib inline

    # set the environment
    from _static.common.helpers import setup_hardware_client, get_nightly_calibration

    setup_hardware_client()

    # load automatic calibration and set up pynn
    calib = get_nightly_calibration()
    pynn.setup(initial_config=calib)

    # your code


    pynn.end()

The XOR-operation
-----------------

Lastly, let’s consider the XOR-operation (e**X**\ clusive **OR**). Here,
the output neuron should only fire if exactly one of the input neurons
fires, but not if both of them fire.

.. raw:: html
    <table><tr>
    <td style="padding:0 100px 0 100px;"> <img src="_static/common/network2in.png" width="300"/> </td>

    <td style="padding:0 100px 0 100px;"> <table>
      <tr>
        <th>neuron 1</th>
        <th>neuron 2</th>
        <th>output neuron</th>
      </tr>
      <tr>
        <td style="text-align: center">-</td>
        <td style="text-align: center">-</td>
        <td style="text-align: center">-</td>
      </tr>
      <tr>
        <td style="text-align: center">x</td>
        <td style="text-align: center">-</td>
        <td style="text-align: center">x</td>
      </tr>
      <tr>
        <td style="text-align: center">-</td>
        <td style="text-align: center">x</td>
        <td style="text-align: center">x</td>
      </tr>
      <tr>
        <td style="text-align: center">x</td>
        <td style="text-align: center">x</td>
        <td style="text-align: center">-</td>
      </tr>
    </table> </td>

    </tr><table>

Start by sketching a network that in your opinion should fulfill this
task. Then, implement it and prove its functioning.

.. code:: ipython3

    import pynn_brainscales.brainscales2 as pynn
    import matplotlib.pyplot as plt
    %matplotlib inline

    # set the environment
    from _static.common.helpers import setup_hardware_client, get_nightly_calibration

    setup_hardware_client()

    # load automatic calibration and set up pynn
    calib = get_nightly_calibration()
    pynn.setup(initial_config=calib)

    # your code


    pynn.end()

These operations are quite simple, but very powerful. Connected in
series, it is possible to build any Boolean expression however
complicated it might be. This is the basis for what modern processors
do. Concluding, we see that neurons are Turing-complete, i.e. in
principle they can do anything a computer does, too.
