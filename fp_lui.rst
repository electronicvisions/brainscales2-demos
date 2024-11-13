Lu.i experiment
===============

To get accustomed to the concept of neuromorphic computation, you will work with *Lu.i*.
Lu.i is an electronic neuron circuit mimicking and illustrating the basic dynamics of real, biological neurons.
The printed circuit board (PCB) features a configurable, fully analog implementation of the leaky integrate-and-fire model and visualizes the internal state, the membrane potential, through a VU-meter-style chain of LEDs.
The neuron emits a short pulse whenever the membrane potential crosses a predefined threshold voltage.
Neurons communicate by exchanging these spikes because multiple boards can be connected via jumper wires to form networks.

.. image:: _static/common/lui-with-coin_lq_cropped.jpeg
   :width: 50%
   :align: center

You may find additional information including schematics on the project's homepage, `giantaxon.org <http://giantaxon.org>`_.

Exercises
~~~~~~~~~

Getting familiar with Lu.i's parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before you begin, connect Lu.i's membrane to the oscilloscope.

- Familiarize yourself with the parameters of the neuron emulation.
  What happens if you increase the leak potential?
  Can you observe the membrane time constant?
  Note the impact of the individual potentiometers.
- Connect two or multiple neurons to form a simple network and note how signals propagate across neurons, e.g., while playing with the synaptic weights and the synaptic time constant.
- Pick a configuration where you can approximately measure :math:`\tau_\text{ref}` and the threshold :math:`\vartheta`.
  Take notes of your results and your methods of determining the individual parameters.

.. note::
   Keep the setting for the membrane time constant :math:`\tau_\text{mem}` fixed for the following task.


Measuring of an f-I curve
^^^^^^^^^^^^^^^^^^^^^^^^^

Next, you will record your first f-I curve, i.e., the dependence of a neuron's firing rate on its stimulating current.
For this purpose, you will rely on a voltage-controlled current source.

.. image:: _static/fp/lui_current_source.png
   :width: 45%
   :align: center

The output current of this circuit is governed by the following equation:

.. math::
    I(U) = \frac{V_\text{CC}-U}{R_\text{sense}}

In the present case, :math:`R_\text{sense}` is 10 kΩ.

Make sure that Lu.i's membrane is connected to both the oscilloscope and the current source.
A second set of cables allows you to read out the voltage at the lower node of :math:`R_\text{sense}` at the same time such that you can calculate the voltage drop across the resistor and thus the current.

- Now measure about 20 firing frequencies as a function of the voltage drop in a range from :math:`1.75V` to :math:`5V` (here referring to the absolute potential).
- Plot your results.
- Derive an equation for :math:`f_\text{theo}(\vartheta, \tau_m, R, I)`.
  You might need to define your integration limits.
  (Hint 1: Consult equation :eq:`eq:lif` as a starting point;
  hint 2: You'll find :math:`R\cdot I > \vartheta` as a condition)
- Fit the function to your measurements.

.. only:: Solution

    **Solution:**

    .. code:: ipython3

        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.optimize import curve_fit


        def get_data(filename=None):
            if filename is None:
                # Use dummy results
                return np.array([[5.1, 0.],
                                 [5., 0.],
                                 [4.96, 1.],
                                 [4.83, 2.24],
                                 [4.68, 3.47],
                                 [4.54, 4.46],
                                 [4.3, 6.1],
                                 [4.16, 7.14],
                                 [4., 8.06],
                                 [3.89, 8.6],
                                 [3.78, 9.26],
                                 [3.62, 10.],
                                 [3.55, 10.6],
                                 [3.3, 11.4],
                                 [3., 13.9],
                                 [2.62, 15.6],
                                 [2.44, 16.7],
                                 [2.05, 17.8],
                                 [1.75, 17.8]])
            else:
                # Load results from file
                with open(filename, "r") as f:
                    data = f.readlines()
                return np.array([[float(da) for da in d.split()] for d in data])


        data = get_data()
        V_CC = 5.2
        x = data[:, 0]
        x = (V_CC - x) / 10000
        y = data[:, 1]


        def func(I, tau_m, V_thres, R, E_l, V_reset):
            tau_ref = 0.0012
            return np.nan_to_num(
                1 / (tau_ref + tau_m * np.log((V_reset - E_l - R * I) / (V_thres - E_l - R * I))))


        bounds = {
            "min": {
                "tau_m": 0,
                "V_thres": 0,
                "R": 0,
                "E_l": 0,
                "V_reset": 0},
            "max": {
                "tau_m": 10,
                "V_thres": 3.5,
                "R": np.inf,
                "E_l": 3.5,
                "V_reset": 0.2}
        }
        start = {
            "tau_m": 1,
            "V_thres": 1.5,
            "R": 10000,
            "E_l": 0.5,
            "V_reset": 0.
        }

        param_bounds = [list(b.values()) for b in bounds.values()]
        p0 = list(start.values())
        fitted_parameters, pcov = curve_fit(func, x, y, p0=p0, bounds=param_bounds)

        print(
            f"tau_m = {fitted_parameters[0]}, V_thres = {fitted_parameters[1]}, R = {fitted_parameters[2]}, E_l = {fitted_parameters[3]}, V_reset = {fitted_parameters[4]}")
        plt.plot(x, y, label="measured")
        vals = np.arange(0, np.max(x), 1e-6)
        plt.plot(vals, func(vals, *fitted_parameters), label="fit")
        plt.legend()
        plt.savefig("result.png")
