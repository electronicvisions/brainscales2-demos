Spiking Characteristics of Leaky Integrate-and-Fire Models
==========================================================

In this exercise, we will investigate the steady-state firing dynamics of LIF neurons under constant current injection (see `Chapter 1, Section 3 <https://neuronaldynamics.epfl.ch/online/Ch1.S3.html>`_ for more information about LIF neurons).
The resulting input-output response is graphically represented using the frequency-current (f-I) curve.
The f-I curve serves multiple purposes in both, experimental and computational neuroscience.
It provides insights into how neurons encode input current to output spikes.

The shape of the f-I curve also reflects the neuron's type and its intrinsic properties.
Specifically, type I neurons have continuous f-I curves whereas type II neurons have discontinuous f-I curves.
This indicates that type I neurons can fire at low frequencies, while type II neurons cannot.
For more information on neuron types, see `Chapter 4, Section 4 <https://neuronaldynamics.epfl.ch/online/Ch4.S4.html>`_.
You can also refer to `Chapter 2, Section 2 <https://neuronaldynamics.epfl.ch/online/Ch2.S2.html>`_ for f-I curves of the biophysical model of Hodgkin and Huxley.

To realize this goal, we set up an experiment consisting of one neuron and inject a constant current onto its membrane.
We will then investigate the effect of different neuron parameters on the shape of the f-I curve.
Finally, we will determine the neuron type from the f-I curve and discuss models that yield different neuron types.


Experiment Setup
----------------

To prepare for the execution of experiments on BrainScaleS-2, we establish the connection to the system and import the required packages.

.. include:: common_quiggeldy_setup.rst

.. code:: ipython3

    %matplotlib inline
    import numpy as np
    import matplotlib.pyplot as plt
    import quantities as pq

    import pynn_brainscales.brainscales2 as pynn


Recording the f-I Curve
-----------------------

We deal here with a single neuron and inject a constant current onto its membrane.
Neuron parameters in this exercise are in hardware units.
To configure a constant current source, we use the following parameters in the neuron population definition:
``constant_current_enable`` (True) and ``constant_current_i_offset`` (range=0-1022).
After running the experiment, we read-out the spikes and compute the firing rate.
Finally, we visualize our results by plotting the firing rate versus the magnitude of injected current.


Exercises
^^^^^^^^^

- **Task 1:**
  For ease of reproducibility, write a function ``fI_experiment`` that takes as arguments the neuron refractory period, leak conductance, array of injected currents, and runtime duration of each experiment.
  The rest of the neuron parameters can be hard-coded in the function.
  Your function should sweep over the current values and run an experiment for each value.
  In each experiment, record the spikes and compute the firing rate.
  You can reuse code from :ref:`lif_parameters`.
  Your function should return the spike rate for plotting against the current range.

- **Task 2:**
  Generate an f-I curve for a runtime duration of ``1 ms``, current range over (0-1000) with a step_size of ``20``, and refractory period of ``10``.
  Choose your neuron parameters such that you obtain a mean resting potential of ``300``, a membrane time constant of ``10 us``, and a threshold read of ``350``.
  You may use the values you determined in :ref:`lif_parameters`.
  Observe the rheobase (current threshold for firing) and the slope of the f-I curve.

- **Task 3:**
  Investigate the effect of the leak conductance on the f-I curve.
  Choose three values of the leak conductance with a difference of ``100``.
  Can you qualitatively explain the effects?
  Note, the range of the leak conductance is ``0-1022``.

- **Task 4:**
  Investigate the effect of the refractory period on the f-I curve.
  Choose three values of the refractory period with minimal difference of ``10``.
  What is the maximum theoretical firing rate?
  Make sure the refractory period is consistent with the duration of experiment for high refractory periods.
  Note, the range of the refractory period is ``10-1022``.

- **Task 5:**
  Examine the firing behavior of the neuron around the rheobase.
  Use your knowledge of the rheobase (current threshold) from task 2 to study the firing behavior of the neuron for a limited current range in the vicinity of the rheobase.
  For your f-I curve, sweep over a current range of (rheobase-20, rheobase+50) in a step-size of ``1``, a refractory period of ``10``, and an experiment duration of ``1 ms``.
  Comment on the type of the neuron following the LIF dynamics.
  Can you spot any imperfections in the plot?
  Which neuron models can result in different f-I curves?

- **Task 6:**
  Finally, let's discuss the significance of the f-I curve.
  What does the slope of the f-I curve tell about the encoding behavior?
  Can you forsee an advantage of f-I curves in computational neuroscience?
  Compare the f-I curve of the LIF model to the rectified linear unit (ReLU) activation function in machine learning.

.. only:: Solution

    **Solution:**

    - **Task 1:**

    .. code:: ipython3

        def fI_experiment(currents,
                          refractory_period,
                          leak_conductance,
                          experiment_duration):
            """
            Stimulate a neuron with a constant current and record its output firing rate.

            :param currents: values of current to test
            :param refractory_period: neuron refractory period in hardware units
            :param leak_conductance: neuron membrane leak conductance in hardware units
            :param experiment_duration: duration of a single experiment run for a considered
              current magnitude
            :return firing_rate: firing rate of the neuron corresponding to the given current
              range
            """
            pynn.setup()
            pop = pynn.Population(1, pynn.cells.HXNeuron(
                      leak_i_bias=leak_conductance,
                      leak_v_leak=530,
                      threshold_v_threshold=250,
                      reset_v_reset=300,
                      membrane_capacitance_capacitance=63,
                      refractory_period_refractory_time=refractory_period,
                      threshold_enable=True,
                      reset_i_bias=1022,
                      reset_enable_multiplication=True))
            firing_rate = np.zeros_like(currents)
            for i, current in enumerate(currents):
                pop.record(["spikes"])
                pop.set(constant_current_enable=True,
                        constant_current_i_offset=current)
                pynn.run(experiment_duration)
                spikes = pop.get_data("spikes").segments[-1].spiketrains[0]
                if (len(spikes) > 1):
                  firing_rate[i] = 1 / np.mean(np.diff(spikes))
                pynn.reset()
            pynn.end()
            return firing_rate

    - **Task 2:**
      The threshold or rheobase is the value of the current when the neuron starts to spike, or the firing rate is higher than 0.
      The resulting f-I curve is linear, where the firing rate is almost proportional to the current beyond the threshold (rheobase).
      The steepness of the curve depends on the choice of parameters.
      For example, higher leak conductance and refractory period lead to smaller slope (less steep curve).

    .. code:: ipython3

        current_range = np.linspace(0, 1000, 50)
        firing_rate = fI_experiment(currents=current_range, refractory_period=10,
                                    leak_conductance=150, experiment_duration=1)
        fig, ax = plt.subplots()
        ax.scatter(current_range, firing_rate)
        ax.set(xlabel="Current (a.u.)", ylabel="Firing rate (KHz)")

    - **Task 3:**
      An increase in the leak conductance causes the membrane to leak faster and consequently decreases the firing rate for a given current.
      Also, the rheobase or the current threshold at which the neuron starts to fire increases.
      Theoretically, two effects should be seen: a shift in the threshold and a less steep f-I curve.
      The shift in the rheobase should be observed clearly from the plots.
      Due to the limited supplied current, the observation of a less steep curve is not shown.
      The beginning of the f-I curve shows a linear behavior which is similar across different values of conductance.
      However, it should be visible that the highest firing rate reached for a given current decreases compared to lower conductance.
      The theoretical asymptotic firing rate remains the same.

    .. code:: ipython3

        current_range = np.linspace(0, 1000, 50)
        conductance_range = [150, 300, 450]
        firing_rate = np.zeros((len(conductance_range), len(current_range)))
        fig, ax = plt.subplots()
        for c, conductance in enumerate(conductance_range):
          firing_rate[c] = fI_experiment(currents=current_range, refractory_period=10,
                                         leak_conductance=conductance, experiment_duration=1)
          ax.scatter(current_range, firing_rate[c], label=f"Conductance={conductance}")
        ax.legend()
        ax.set(xlabel="Current (a.u.)", ylabel="Firing rate (KHz)")

    - **Task 4:**
      An increase in refractory period leads to a decrease in the firing rate and therefore a less steep f-I curve for a given current.
      The theoretical asymptotic firing rate decreases with an increase in the refractory period.
      Ideally, the rheobase should remain the same if a sufficient duration is chosen.
      Both effects are clearly shown in the plots.

    .. code:: ipython3

        current_range = np.linspace(0, 1000, 50)
        refractory_range = [10, 30, 50]
        firing_rate = np.zeros((len(refractory_range), len(current_range)))
        fig, ax = plt.subplots()
        for r, ref_period in enumerate(refractory_range):
          firing_rate[r] = fI_experiment(currents=current_range, refractory_period=ref_period,
                                         leak_conductance=150, experiment_duration=1)
          ax.scatter(current_range, firing_rate[r], label=f"Ref-period={ref_period}")
        ax.legend()
        ax.set(xlabel="Current (a.u.)", ylabel="Firing rate (KHz)")

    - **Task 5:**
      Ideally, a type I neuron should be observed in which the firing rate increases slowly beyond the rheobase, and small frequency values near zero are achievable.
      Some low frequency values are not represented in the f-I curve, mainly due to the analog circuit and the small currents of BrainScaleS-2.
      In the presence of an adaptation current, a type II neuron can be obtained.
      The Hodgkin-Huxley model also yields type II neurons.

    .. code:: ipython3

        current_range = np.arange(200, 270)
        fig, ax = plt.subplots()
        firing_rate = fI_experiment(currents=current_range, refractory_period=10,
                                    leak_conductance=150, experiment_duration=1)
        ax.scatter(current_range, firing_rate)
        ax.set(xlabel="Current (a.u.)", ylabel="Firing rate (KHz)")

    - **Task 6:**
      The slope of the f-I curve is a measure of the neuron sensitivity to an input stimulus.
      The higher the slope, the more responsive the neuron is to a specific stimulus at a given intensity.
      This is useful for understanding, for example, the perception of sensory inputs.
      In computational neuroscience, the f-I curves of neuron models can be compared to those from experimental data.
      This can ensure that the neuron model is realistic when the goal is to model biologically-plausible neural networks.

      The f-I curve of the LIF model can be approximated by a threshold-linear function up to a certain range of stimulus.
      The ReLU is a threshold-linear function used as an activation function in machine learning, thus the analogy between the two.
