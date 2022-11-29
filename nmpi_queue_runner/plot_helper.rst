.. code:: ipython3

    import matplotlib.pyplot as plt
    import numpy as np
    from neo.io import PickleIO

    def plot_membrane_dynamics(path : str):
        """
        Load the neuron data from the given path and plot the membrane potential
        and spiketrain of the first neuron.
        :param path: Path to the result file
        """
        # Experimental results are given in the 'neo' data format, the following
        # lines extract membrane traces as well as spikes and construct a simple
        # figure.
        block = PickleIO(path).read_block()
        for segment in block.segments:
            if len(segment.irregularlysampledsignals) != 1:
                raise ValueError("Plotting is supported for populations of size 1.")
            mem_v = segment.irregularlysampledsignals[0]
            try:
                for spiketime in segment.spiketrains[0]:
                    plt.axvline(spiketime, color="black")
            except IndexError:
                print("No spikes found to plot.")
            plt.plot(mem_v.times, mem_v, alpha=0.5)
        plt.xlabel("Wall clock time [ms]")
        plt.ylabel("ADC readout [a.u.]")
        plt.ylim(0, 1023)  # ADC precision: 10bit -> value range: 0-1023
        plt.show()

