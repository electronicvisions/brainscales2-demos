Introduction to the non-interactive queue runner
================================================

Welcome to this tutorial of using pyNN for the BrainScaleS-2 neuromorphic accelerator with the non-interactive queue runner.
We will guide you through all the steps necessary to interact with the system and help you explore the capabilities of the on-chip analog neurons and synapses.
A tutorial of the interactive usage can be found in :ref:`BrainScaleS-2 single neuron experiments`.

.. only:: not exclude_nmpi

    .. code-block:: ipython3

        !pip install -U hbp_neuromorphic_platform
        !pip install ebrains-drive
        %matplotlib inline

    .. code-block:: ipython3

       import nmpi
       client = nmpi.Client()
       import time
       import os
       import ebrains_drive
       import requests
       from ebrains_drive.client import DriveApiClient

The next cell is a workaround for a missing functionality. Just run and ignore...

.. only:: not exclude_nmpi

    .. include:: ./nmpi_queue_runner/helper.rst

Next, we define a function which is used to execute the experiments on the BrainScaleS-2 neuromorphic accelerator.

.. only:: not exclude_nmpi

    .. include:: ./nmpi_queue_runner/execute_on_hw.rst

First Experiment
----------------

Now we can start with the first experiment. Here, we will record the membrane of a single, silent
neuron on the analog substrate.
We save our experiment description to a python file and will send this file to a cluster in Heidelberg.
In Heidelberg the experiment is executed on the BrainScaleS-2 neuromorphic system, the results are collected and send back.

.. only:: not exclude_nmpi

    .. include:: ./nmpi_queue_runner/first_exp.txt
        :code: python

    .. code:: ipython3

        first_experiment_id = execute_on_hardware('first_experiment.py')

.. only:: exclude_nmpi

    .. include:: ./nmpi_queue_runner/first_exp.txt
        :code: python
        :start-line: 1

The following helper function plots the membrane potential as well as any spikes found in the result file of the experiment. It will be used throughout this tutorial.

.. include:: ./nmpi_queue_runner/plot_helper.rst

Plot the results of the first experiment.

.. only:: not exclude_nmpi

    .. code:: ipython3

        plot_membrane_dynamics(f"{outputDir}/job_{first_experiment_id}/first_experiment.dat")

.. only:: exclude_nmpi

    .. code:: ipython3

        plot_membrane_dynamics("first_experiment.dat")

Second Experiment
-----------------

As a second experiment, we will let the neurons on BrainScaleS-2 spike by setting a 'leak-over-threshold' configuration.

.. only:: not exclude_nmpi

    .. include:: ./nmpi_queue_runner/second_exp.txt
        :code: python

.. only:: exclude_nmpi

    .. include:: ./nmpi_queue_runner/second_exp.txt
        :code: python
        :start-line: 1

Execute the experiment on the neuromorphic hardware and plot the results.

.. only:: not exclude_nmpi

    .. code:: ipython3

        print("Starting 2nd experiment at ",time.ctime())
        second_experiment_id = execute_on_hardware('second_experiment.py')
        print("Start Plotting:")
        plot_membrane_dynamics(f"{outputDir}/job_{second_experiment_id}/second_experiment.dat")

.. only:: exclude_nmpi

    .. code:: ipython3

        plot_membrane_dynamics("second_experiment.dat")

Third Experiment: Fixed-pattern noise
-------------------------------------

Due to the analog nature of the BrainScaleS-2 platform, the inevitable mismatch of semiconductor fabrication results in inhomogeneous properties of the computational elements.
We will visualize these effects by recording the membrane potential of multiple neurons in leak-over-threshold configuration.
You will notice different resting, reset and threshold potentials as well as varying membrane time constants.

.. only:: not exclude_nmpi

    .. include:: ./nmpi_queue_runner/third_exp.txt
        :code: python

.. only:: exclude_nmpi

    .. include:: ./nmpi_queue_runner/third_exp.txt
        :code: python
        :start-line: 1

Execute the experiment on the neuromorphic hardware and plot the results.

.. only:: not exclude_nmpi

    .. code:: ipython3

        third_experiment_id = execute_on_hardware('third_experiment.py')
        print("Start Plotting:")
        plot_membrane_dynamics(f"{outputDir}/job_{third_experiment_id}/third_experiment.dat")

The plot shows the recorded membrane traces of multiple different neurons.
Due to the time-continuous nature of the system, there is no temporal alignment between the individual traces, so the figure shows multiple independent effects:
* Temporal misalignment: From the system's view, the recording happens in an arbitrary time frame during the continuously evolving integration. Neurons are not synchronized to each other.
* Circuit-level mismatch: Each individual neurons shows slightly different analog properties. The threshold is different for all traces; as is the membrane time constant (visible as slope) and the reset potentials (visible as plateaus during the refractory time).

.. only:: exclude_nmpi

    .. code:: ipython3

        plot_membrane_dynamics("third_experiment.dat")

Fourth Experiment: External stimulation
---------------------------------------

Up to now, we have observed analog neurons without external stimulus. In
this experiment, we will introduce the latter and examine post-synaptic
pulses on the analog neuron's membrane.

.. only:: not exclude_nmpi

    .. include:: ./nmpi_queue_runner/fourth_exp.txt
        :code: python

.. only:: exclude_nmpi

    .. include:: ./nmpi_queue_runner/fourth_exp.txt
        :code: python
        :start-line: 1

Execute the experiment on the neuromorphic hardware and plot the results.

.. only:: not exclude_nmpi

    .. code:: ipython3

        print("Starting 4th experiment at ",time.ctime())
        fourth_experiment_id = execute_on_hardware('fourth_experiment.py')
        print("Start Plotting:")
        plot_membrane_dynamics(f"{outputDir}/job_{fourth_experiment_id}/fourth_experiment.dat")

.. only:: exclude_nmpi

    .. code:: ipython3

        plot_membrane_dynamics("fourth_experiment.dat")
