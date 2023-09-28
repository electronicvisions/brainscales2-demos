Training an SNN on BrainScaleS-2 with PyTorch
=============================================

In this tutorial we create and train a spiking neural network (SNN) on
the neuromorphic hardware system BrainScaleS-2 (BSS-2) [1] to solve the
Yin-Yang dataset [2] using the PyTorch-based software framwork
``hxtorch.snn`` [3]. For training we rely on surrogate gradients [4].

This tutorial assumes you are familiar with the basics of
``hxtorch.snn`` and the training of SNNs with surrogate gradients.

For further reading, see the references below.

References and further reading
------------------------------

[1] Pehle, C., Billaudelle, S., Cramer, B., Kaiser, J., Schreiber, K.,
Stradmann, Y., Weis, J., Leibfried, A., Müller, E., and Schemmel, J. The
BrainScaleS-2 accelerated neuromorphic system with hybrid plasticity.
Frontiers in Neuroscience, 16, 2022. ISSN 1662-453X. `doi:
10.3389/fnins.2022.795876 <https://www.frontiersin.org/articles/10.3389/fnins.2022.795876/full>`__.

[2] Kriener, L., Göltz, J., and Petrovici, M. A. The yin-yang dataset.
In Neuro-Inspired Computational Elements Conference, NICE 2022,
pp. 107–111, New York, NY, USA, 2022. Association for Computing
Machinery. ISBN 9781450395595. `doi:
10.1145/3517343.3517380 <https://dl.acm.org/doi/10.1145/3517343.3517380>`__.

[3] Spilger, P., Arnold, E., Blessing, L., Mauch, C., Pehle, C., Müller,
E., and Schemmel, J. hxtorch.snn: Machine- learning-inspired spiking
neural network modeling on BrainScaleS-2. Accepted, 2023. `doi:
10.48550.2212.12210 <https://doi.org/10.48550/arXiv.2212.12210>`__

[4] Emre O. Neftci, Hesham Mostafa, and Friedemann Zenke. 2019.
Surrogate gradi- ent learning in spiking neural networks: Bringing the
power of gradient-based optimization to spiking neural networks. IEEE
Signal Processing Magazine 36, 6 (2019),51–63.
https://doi.org/10.1109/MSP.2019.2931595

[5] Wunderlich, T.C., Pehle, C. Event-based backpropagation can compute exact gradients
for spiking neural networks. Scientific Reports 11, 12829 (2021).
`doi: 10.1038/s41598-021-91786-z <https://doi.org/10.1038/s41598-021-91786-z>`__.

.. code:: ipython3

    %matplotlib inline
    from typing import Tuple
    import matplotlib.pyplot as plt
    import ipywidgets as w
    import numpy as np
    import torch
    from _static.common.helpers import setup_hardware_client, save_nightly_calibration
    from _static.tutorial.snn_yinyang_helpers import plot_data, plot_input_encoding, plot_training
    setup_hardware_client()

.. code:: ipython3

    # Some seeds
    torch.manual_seed(42)

Yin-Yang Dataset
~~~~~~~~~~~~~~~~

The Yin-Yang dataset is a small two-dimensional nonlinear dataset to be
solved with limited resources where the sharp boundaries between the
classes - despite the low input dimensionality - create a hard problem
that neatly separates SNNs according to their capabilities. It consists
of three classes: the yin, the yang and dots. Each data sample :math:`i` is
defined by its coordinates :math:`(x_i, y_i)`, assigned to one of the three
classes, depending on the area it is located in. Further information can
be found in [2].

First, we import the YinYangDataset and create a PyTorch DataLoader to
access and visualize the dataset.

.. code:: ipython3

    from hxtorch.snn.datasets.yinyang import YinYangDataset
    from torch.utils.data import DataLoader

    R_BIG = 0.5  # Radius of yin-yang sign
    R_SMALL = 0.1  # Radius of yin-yang eyes

    def get_data_loaders(
            trainset_size: int, testset_size: int = 1, batch_size: int = 75):
        """
        Create a training and a test Yin-Yang dataset and return corresponding PyTorch
        data loaders.
        :param trainset_size: Number of samples in the training set.
        :param testset_size: Number of samples in the testset.
        :param batch_size: The batch size, used for both, the train and the test loader.
        :return: Returns a train and test data loader.
        """
        trainset = YinYangDataset(r_big=R_BIG, r_small=R_SMALL, size=trainset_size)
        testset = YinYangDataset(r_big=R_BIG, r_small=R_SMALL, size=testset_size)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)
        return train_loader, test_loader

.. code:: ipython3

    train_loader, _ = get_data_loaders(1000, 10, batch_size=1000)
    train_loader, len(train_loader)

.. code:: ipython3

    # Get data and targets
    data, targets = next(iter(train_loader))
    data[:10], targets[:10]

.. code:: ipython3

    # One random example for which we want to look at its spike encoding
    example = data[np.random.randint(0, len(data))]
    example

.. code:: ipython3

    plot_data(example, data, targets)

SNN Model
~~~~~~~~~

We now define an SNNs which we want to train to classify the class of a
given sample. For that we use an SNN with one hidden leaky-integrate and
fire (LIF) layer projecting its spike events onto one leaky-integrator
(LI) readout layer, as in [3]. Each neuron in the output layer
corresponds to one of the three classes:

.. code:: ipython3

    from functools import partial
    import hxtorch
    import hxtorch.snn as hxsnn
    import hxtorch.snn.functional as F
    from hxtorch.snn.transforms import weight_transforms
    from dlens_vx_v3 import halco

    log = hxtorch.logger.get("grenade.backend")
    hxtorch.logger.default_config(level=hxtorch.logger.LogLevel.ERROR)

.. code:: ipython3

    class SNN(torch.nn.Module):
        """ SNN with one hidden LIF layer and one readout LI layer """

        def __init__(
                self,
                n_in: int,
                n_hidden: int,
                n_out: int,
                mock: bool,
                dt: float,
                tau_mem: float,
                tau_syn: float,
                alpha: float,
                trace_shift_hidden: int,
                trace_shift_out: int,
                weight_init_hidden: Tuple[float, float],
                weight_init_output: Tuple[float, float],
                weight_scale: float,
                trace_scale: float,
                input_repetitions: int,
                device: torch.device):
            """
            :param n_in: Number of input units.
            :param n_hidden: Number of hidden units.
            :param n_out: Number of output units.
            :param mock: Indicating whether to train in software or on hardware.
            :param dt: Time-binning width.
            :param tau_mem: Membrane time constant.
            :param tau_syn: Synaptic time constant.
            :param trace_shift_hidden: Indicates how many indices the membrane
                trace of hidden layer is shifted to left along time axis.
            :param trace_shift_out: Indicates how many indices the membrane
                trace of readout layer is shifted to left along time axis.
            :param weight_init_hidden: Hidden layer weight initialization mean
                and std value.
            :param weight_init_output: Output layer weight initialization mean
                and std value.
            :param weight_scale: The factor with which the software weights are
                scaled when mapped to hardware.
            :param input_repetitions: Number of times to repeat input channels.
            :param device: The used PyTorch device used for tensor operations in
                software.
            """
            super().__init__()

            # Neuron parameters
            lif_params = F.CUBALIFParams(
                1. / tau_mem, 1. / tau_syn, alpha=alpha)
            li_params = F.CUBALIParams(1. / tau_mem, 1. / tau_syn)
            self.dt = dt

            # Instance to work on

            if not mock:
                save_nightly_calibration('spiking2_cocolist.pbin')
                self.experiment = hxsnn.Experiment(mock=mock, dt=dt,
                        calib_path='spiking2_cocolist.pbin')
            else:
                self.experiment = hxsnn.Experiment(mock=mock, dt=dt)

            # Repeat input
            self.input_repetitions = input_repetitions

            # Input projection
            self.linear_h = hxsnn.Synapse(
                n_in * input_repetitions,
                n_hidden,
                experiment=self.experiment,
                transform=partial(
                    weight_transforms.linear_saturating, scale=weight_scale))

            # Initialize weights
            if weight_init_hidden:
                w = torch.zeros(n_hidden, n_in)
                torch.nn.init.normal_(w, *weight_init_hidden)
                self.linear_h.weight.data = w.repeat(1, input_repetitions)

            # Hidden layer
            self.lif_h = hxsnn.Neuron(
                n_hidden,
                experiment=self.experiment,
                func=F.cuba_lif_integration,
                params=lif_params,
                trace_scale=trace_scale,
                cadc_time_shift=trace_shift_hidden,
                shift_cadc_to_first=True)

            # Output projection
            self.linear_o = hxsnn.Synapse(
                n_hidden,
                n_out,
                experiment=self.experiment,
                transform=partial(
                    weight_transforms.linear_saturating, scale=weight_scale))

            # Readout layer
            self.li_readout = hxsnn.ReadoutNeuron(
                n_out,
                experiment=self.experiment,
                func=F.cuba_li_integration,
                params=li_params,
                trace_scale=trace_scale,
                cadc_time_shift=trace_shift_out,
                shift_cadc_to_first=True,
                placement_constraint=list(
                    halco.LogicalNeuronOnDLS(
                        hxsnn.morphology.SingleCompartmentNeuron(1).compartments,
                        halco.AtomicNeuronOnDLS(
                            halco.NeuronRowOnDLS(1), halco.NeuronColumnOnDLS(nrn)))
                    for nrn in range(n_out)))

            # Initialize weights
            if weight_init_output:
                torch.nn.init.normal_(self.linear_o.weight, *weight_init_output)

            # Device
            self.device = device
            self.to(device)

        def forward(self, spikes: torch.Tensor) -> torch.Tensor:
            """
            Perform a forward path.
            :param spikes: NeuronHandle holding spikes as input.
            :return: Returns the output of the network, i.e. membrane traces of the
            readout neurons.
            """
            # Remember input spikes for plotting
            self.s_in = spikes
            # Increase synapse strength by repeating each input
            spikes = spikes.repeat(1, 1, self.input_repetitions)
            # Spike input handle
            spikes_handle = hxsnn.NeuronHandle(spikes)

            # Forward
            c_h = self.linear_h(spikes_handle)
            self.s_h = self.lif_h(c_h)  # Keep spikes for fire reg.
            c_o = self.linear_o(self.s_h)
            self.y_o = self.li_readout(c_o)

            # Execute on hardware
            hxtorch.snn.run(self.experiment, spikes.shape[0])

            return self.y_o.v_cadc

.. code:: ipython3

    N_HIDDEN      = 120
    MOCK          = False
    DT            = 2.0e-06  # s

    # We need to specify the device we want to use on the host computer
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # The SNN
    snn = SNN(
        n_in=5,
        n_hidden=N_HIDDEN,
        n_out=3,
        mock=MOCK,
        dt=DT,
        tau_mem=6.0e-06,
        tau_syn=6.0e-06,
        alpha=50.,
        trace_shift_hidden=int(.0e-06/DT),
        trace_shift_out=int(.0e-06/DT),
        weight_init_hidden=(0.001, 0.25),
        weight_init_output=(0.0, 0.1),
        weight_scale=66.39,
        trace_scale=0.0147,
        input_repetitions=1 if MOCK else 5,
        device=device)
    snn

Since the SNN gets spike events as inputs and the samples from the
dataset are real-valued, we first need to translate them into a
spike-based representation by an ``encoder`` module before we can pass
them to the SNN. Additionally, the we need to define some decoder
functionallity that translates the output of the SNN, here the trace of
the LI layer, into class scores to infere a prediction from. This is
done by an ``decoder`` module. For easier handling, the ``encoder``, the
``snn``, and the ``decoder`` are wrapped into a ``Model`` module:

.. code:: ipython3

    class Model(torch.nn.Module):
        """ Complete model with encoder, network (snn) and decoder """

        def __init__(
                self,
                encoder: torch.nn.Module,
                network: torch.nn.Module,
                decoder: torch.nn.Module,
                readout_scale: float = 1.):
            """
            Initialize the model by assigning encoder, network and decoder
            :param encoder: Module to encode input data
            :param network: Network module containing layers and
                parameters / weights
            :param decoder: Module to decode network output
            """
            super().__init__()

            self.encoder = encoder
            self.network = network
            self.decoder = decoder

            self.readout_scale = readout_scale

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            """
            Perform forward pass through whole model, i.e.
            data -> encoder -> network -> decoder -> output
            :param inputs: tensor input data
            :returns: Returns tensor output
            """
            spikes = self.encoder(inputs)
            traces = self.network(spikes)
            self.scores = self.decoder(traces).clone()

            # scale outputs
            with torch.no_grad():
                self.scores *= self.readout_scale

            return self.scores

        def regularize(
                self,
                reg_readout: float = 0.0,
                reg_bursts: float = 0.0,
                reg_w_hidden: float = 0.0,
                reg_w_output: float = 0.0) -> torch.Tensor:
            """
            Get regularization terms for bursts and weights like
            factor * (thing to be regularized) ** 2.
            :param reg_bursts: prefactor of burst / hidden spike regulaization
            :param reg_weights_hidden: prefactor of hidden weight regularization
            :param reg_weights_output: prefactor of output weight regularization
            :returns: Returns sum of regularization terms
            """
            reg = torch.tensor(0., device=self.scores.device)
            # Reg readout
            reg += reg_readout * torch.mean(self.scores ** 2)
            # bursts (hidden spikes) regularization
            reg += reg_bursts * torch.mean(
                torch.sum(self.network.s_h.spikes, dim=1) ** 2.)
            # weight regularization
            reg += reg_w_hidden * torch.mean(self.network.linear_h.weight ** 2.)
            reg += reg_w_output * torch.mean(self.network.linear_o.weight ** 2.)
            return reg

If we want to use an SNN to classify a sample :math:`i` in the Yin-Yang
dataset, we have to translate the point :math:`(x_i, y_i)` to spikes. For
this, we translate the value in each dimension, as well as their
inverse, to a spike time :math:`t_n^i` of an input neuron :math:`n` into
a range :math:`[t_\text{early}, t_\text{late}]` [2]:

.. math::


   \begin{bmatrix}
       x_{i} \\
       y_{i} \\
       1 - x_{i} \\
       1 - y_{i} \\
   \end{bmatrix}
   \longrightarrow
   \begin{bmatrix}
       t^i_0 \\
       t^i_1 \\
       t^i_2 \\
       t^i_3
   \end{bmatrix}
   = t_\text{early} +
   \begin{bmatrix}
       x_{i} \\
       y_{i} \\
       1 - x_{i} \\
       1 - y_{i}
   \end{bmatrix}
   \left( t_\text{late} - t_\text{early} \right)

.

To increase activity in the network we add an additional input neuron
that has a constant firing time :math:`t^\text{bias}`, such
that sample :math:`i` is represented by the spike events :math:`(t^i_0,
t^i_1, t^i_2, t^i_3, t^\text{bias}_4)^\top`.

The dataset ``YinYangDataset`` returns each data point in the form
:math:`(x_i, y_i, 1-x_i, 1-y_i)`. To translate them into spike times we
use the encoder module ``CoordinatesToSpikes``.

.. code:: ipython3

    from hxtorch.snn.transforms.encode import CoordinatesToSpikes

    T_SIM   = 6.0e-05  # s
    T_EARLY = 2.0e-06  # s
    T_LATE  = 4.0e-05  # s
    T_BIAS  = 1.8e-05  # s

    # This encoder translates the points into spikes on a discrete time lattice
    encoder = CoordinatesToSpikes(
        seq_length=int(T_SIM / DT),
        t_early=T_EARLY,
        t_late=T_LATE,
        dt=DT,
        t_bias=T_BIAS)
    encoder

.. code:: ipython3

    spikes = encoder(example.unsqueeze(0)).squeeze(1)
    spikes

.. code:: ipython3

    plot_input_encoding(spikes.cpu(), T_EARLY, T_LATE, T_BIAS, T_SIM, DT)

As ``decoder`` we use the max-over-time function, which returns the
highest membrane value along the time for each output neuron in the LI
layer. Those max-over-time-values are interpreted as scores.

.. code:: ipython3

    from hxtorch.snn.transforms.decode import MaxOverTime
    decoder = MaxOverTime()
    decoder

.. code:: ipython3

    model = Model(encoder, snn, decoder, readout_scale=10.)
    model

Training
~~~~~~~~

We now create a training routine in a PyTorch fashion. We use the Adam
optimizer for weight optimization and the cross-entropy as loss
function.

.. code:: ipython3

    from tqdm.auto import tqdm

    def predict(model, data, target, loss_func):
        """ """
        scores = model(data)
        loss = model.regularize(reg_readout=0.0004)
        loss = loss_func(scores, target) + loss
        return scores, loss


    def stats(model, scores, target):
        """ """
        # Train accuracy
        pred = scores.cpu().argmax(dim=1)
        acc = pred.eq(target.view_as(pred)).float().mean().item()
        # Firing rates
        rate = model.network.s_h.spikes.sum().item() / scores.shape[0]
        return acc, rate


    def train(model: torch.nn.Module,
              loader: DataLoader,
              loss_func: torch.nn.CrossEntropyLoss,
              optimizer: torch.optim.Optimizer,
              epoch: int, update):
        """
        Perform training for one epoch.
        :param model: The model to train.
        :param loader: Pytorch DataLoader instance providing training data.
        :param optimizer: The optimizer used or weight optimization.
        :param epoch: Current epoch for logging.
        :returns: Tuple (training loss, training accuracy)
        """
        model.train()
        loss, acc = 0., 0.
        n_total = len(loader)

        pbar = tqdm(total=len(loader), unit="batch", leave=False)
        for data, target in loader:

            model.zero_grad()

            scores, loss_b = predict(model, data.to(device), target.to(device), loss_func)

            loss_b.backward()
            optimizer.step()

            acc_b, rate_b = stats(model, scores, target)

            acc += acc_b / n_total
            loss += loss_b.item() / n_total

            update(n_total, loss_b.item(), 100 * acc_b, rate_b)

            pbar.set_postfix(
                epoch=f"{epoch}", loss=f"{loss_b.item():.4f}", acc=f"{acc_b:.4f}",
                rate=f"{rate_b:.2f}", lr=f"{optimizer.param_groups[-1]['lr']}")
            pbar.update()
        pbar.close()

        return loss, acc


    def test(model: torch.nn.Module,
             loader: torch.utils.data.DataLoader,
             loss_func: torch.nn.CrossEntropyLoss,
             epoch: int, update):
        """
        Test the model.
        :param model: The model to test
        :param loader: Data loader containing the test data set
        :param epoch: Current trainings epoch.
        :returns: Tuple of (test loss, test accuracy)
        """
        model.eval()
        dev = model.network.device

        loss, acc, rate = 0., 0., 0
        data, target, scores = [], [], []
        n_total = len(loader)

        pbar = tqdm(total=len(loader), unit="batch", leave=False)
        for data_b, target_b in loader:
            scores_b, loss_b = predict(model, data_b.to(device), target_b.to(device), loss_func)
            scores.append(scores_b.detach())
            data.append(data_b.detach())
            target.append(target_b.detach())

            acc_b, rate_b = stats(model, scores_b, target_b)
            acc += acc_b / n_total
            loss += loss_b.item() / n_total
            rate += rate_b / n_total

            pbar.update()
        pbar.close()
        print(f"Test epoch: {epoch}, average loss: {loss:.4f}, test acc={100 * acc:.2f}%")

        scores = torch.stack(scores).reshape(-1, 3)
        data = torch.stack(data).reshape(-1, 4)
        target = torch.stack(target).reshape(-1)

        update(
            model.network.s_in.detach(),
            model.network.s_h.spikes.detach(),
            model.network.y_o.v_cadc.detach(),
            data, target, scores,
            loss, 100 * acc, rate)

        return loss, acc, rate

.. code:: ipython3

    # Training params
    LR            = 0.002
    STEP_SIZE     = 5
    GAMMA         = 0.9
    EPOCHS        = 4 # Adjust here for longer training...
    BATCH_SIZE    = 75
    TRAINSET_SIZE = 5025
    TESTSET_SIZE  = 1050

.. code-block:: ipython3
    :class: test, html-display-none

    # Training params
    LR            = 0.002
    STEP_SIZE     = 5
    GAMMA         = 0.9
    EPOCHS        = 1
    BATCH_SIZE    = 50
    TRAINSET_SIZE = 500
    TESTSET_SIZE  = 100

.. code:: ipython3

    # Just for plotting...
    assert TRAINSET_SIZE % BATCH_SIZE == 0

    # PyTorch stuff... optimizer, scheduler and loss like you normally do.
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    loss = torch.nn.CrossEntropyLoss()

    # Data loaders
    train_loader, test_loader = get_data_loaders(TRAINSET_SIZE, TESTSET_SIZE, BATCH_SIZE)

.. code:: ipython3

    # Functions to update plot
    update_plot, update_train_data, update_test_data = plot_training(N_HIDDEN, T_SIM, DT)
    plt.close()
    output = w.Output()
    display(output)

    # Initialize the hardware
    if not MOCK:
        hxtorch.init_hardware()

    # Train and test
    for epoch in range(0, EPOCHS + 1):
        # Test
        loss_test, acc_test, rate_test = test(
            model, test_loader, loss, epoch, update_test_data)

        # Refresh plot
        output.clear_output(wait=True)
        with output:
            update_plot()

        # Train epoch
        if epoch < EPOCHS:
            loss_train, acc_train = train(
                model, train_loader, loss, optimizer, epoch, update_train_data)

        scheduler.step()

    # Release the hardware connection
    hxtorch.release_hardware()

EventProp
~~~~~~~~~

In [5] Wunderlich and Pehle derived the EventProp algorithm, which provides
a set of equations to compute exact parameter gradients for spiking neural
networks with LIF neurons, single-exponential-shaped synpases and a quite
general loss function.

The background section below is meant to give an overview of the equations
in the EventProp algorthm and provide a basis to understand the
time-discretized implementation in PyTorch autograd functions below.
This is all based directly on [5] and for the detailed derivation of the
algorithm, you might look into the reference.

If you just want to use the functions and train the network using them, you
might skip directly to the training part.

Background
^^^^^^^^^^

The state of a neuron :math:`n` is given by its membrane potential
:math:`V_{n}` and synaptic current :math:`I_{n}`, and their dynamics are
governed by a set of coupled differential equations

.. math::
    \begin{align*}
        & \text{Free dynamics}                  && \quad \text{Transition condition}              && \quad \text{Jumps at transition}   \\
        &\tau_{\mathrm{m}} \dot{V} = - V + I    && \quad (V)_{n} - V_{\mathrm{th}} = 0 \text{, }(\dot{V})_{n} > 0    && \quad (V^{+})_{n} = 0              \\
        &\tau_{\mathrm{s}} \dot{I} = - I        && \quad \text{for any } n                        && \quad I^{+} = I^{-} + W e_{n}
    \end{align*}
    
where the superscripts :math:`+` and :math:`-` denote the right- and left-hand
limit to the post-synaptic spike time.

The loss, which is to be minimized, is of the form

.. math::
    L = l_{\mathrm{p}} (t^{\mathrm{post}}) + \int_{0}^{T} l_{V} (V, t) \mathrm{d}t,

where :math:`l_{\mathrm{p}}(t^{\mathrm{post}})` and :math:`l_{V} (V, t)` are
smooth loss functions depending on the membrane potentials :math:`V`, time
:math:`t` and set of post-synaptic spike times :math:`t^{\mathrm{post}}`.

The system's forward dynamics, defined in the table above, can be introduced
as constraints via Lagrange multipliers :math:`\lambda_{V}` and :math:`\lambda_{I}`,
referring to the equation of the respective state variable. From this, an adjoint
system of differential equations for the lagrange multipliers can be found and
solved in reverse time. They also undergo jumps at the spike times of neurons
found by solving (or in our case emulating) the forward dynamics. Using he
notation :math:`' = - \frac{\mathrm{d}}{\mathrm{d} t}`, the adjoint equations are

.. math::
    \begin{align*}
        & \text{Free dynamics} && \quad \text{Transition condition} && \quad \text{Jumps at transition} \\
        & \tau_{\mathrm{m}} \lambda^{\prime}_{V} = - \lambda_{V} + \frac{\partial l_{V}}{\partial V}
        && \quad t - t^{\mathrm{post}}_{k} = 0
        && \quad \left(\lambda_{V}^{-} \right)_{n(k)} = \left(\lambda_{V}^{+} \right)_{n(k)} + \frac{1}{\tau_{\mathrm{m}} (\dot{V}^{-})_{n(k)} } \bigg[ \vartheta \left(\lambda_{V}^{+} \right)_{n(k)} \\
        & \tau_{\mathrm{s}} \lambda^{\prime}_{I} = - \lambda_{I} + \lambda_{V}
        && \quad \text{for any } k
        && \quad\quad + \left( W^{\top} \left( \lambda_{V}^{+} - \lambda_{I} \right) \right) + \frac{\partial l_{\mathrm{post}}}{\partial t^{\mathrm{post}}_{k}} + l_{V}^{-} - l_{V}^{+} \bigg]
    \end{align*}

The gradient with respect to
the synaptic weight :math:`w_{ji}`, connecting pre-synaptic neuron :math:`i`
to post-synaptic neuron :math:`j`, then only depends on the syaptic time
constant :math:`\tau_{\mathrm{s}}` and the adjoint variable :math:`\lambda_{I}`
at spike times:

.. math::
    \frac{\mathrm{d} L}{\mathrm{d}w_{ji}} = - \tau_{\mathrm{s}}
    \sum_{\text{spikes from } i} (\lambda_{I})_{j}

Implementation
^^^^^^^^^^^^^^

The neuron and synapse modules in ``hxtorch.snn`` allow users to provide
custom functions and we use this ability to implement the EventProp
algorithm [5] as an alternative gradient estimator to the surrogate
gradients which are used in the part above.

To ensure appropriate backpropagation of the terms in the EventProp
equations between layers one has to provide two functions handling
the computation and propagation of gradients, one for ``Neuron`` layer
and one for the ``Synapse`` layer.

.. code:: ipython3

    from typing import NamedTuple, Optional

    class EventPropNeuron(torch.autograd.Function):
        """
        Gradient estimation with time-discretized EventProp using explicit Euler integration.
        """
        @staticmethod
        def forward(ctx, input: torch.Tensor,
                    params: F.CUBALIFParams, dt: float) -> Tuple[torch.Tensor]:
            """
            Forward function, generating spikes at positions > 0.

            :param input: Weighted input spikes in shape (2, batch, time, neurons).
                The 2 at dim 0 comes from stacked output in EventPropSynapse.
            :param params: CUBALIFParams object holding neuron parameters.

            :returns: Returns the spike trains and membrane trace.
                Both tensors are of shape (batch, time, neurons).
            """
            dev = input.device
            T, bs, ps = input[0].shape
            z = torch.zeros(bs, ps).to(dev)
            i = torch.zeros(bs, ps).to(dev)
            v = torch.empty(bs, ps).fill_(params.v_leak).to(dev)

            spikes, current, membrane = [z], [i], [v]
            for ts in range(T - 1):
                # Current
                i = i * (1 - dt * params.tau_syn_inv) + input[0][ts]

                # Membrane
                dv = dt * params.tau_mem_inv * (params.v_leak - v + i)
                v = dv + v

                # Spikes
                z = torch.gt(v - params.v_th, 0.0).to((v - params.v_th).dtype)

                # Reset
                v = (1 - z) * v + z * params.v_reset

                # Save data
                spikes.append(z)
                membrane.append(v)
                current.append(i)

            spikes = torch.stack(spikes)
            membrane = torch.stack(membrane)
            current = torch.stack(current)

            ctx.save_for_backward(input, spikes, membrane, current)
            ctx.extra_kwargs = {"params": params, "dt": dt}

            return spikes, membrane, current

        @staticmethod
        def backward(ctx, grad_spikes: torch.Tensor, grad_membrane: torch.Tensor,
                    grad_current: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
            """
            Implements 'EventProp' for backward.

            :param grad_spikes: Backpropagted gradient wrt output spikes.
            :param _: backpropagated gradient wrt to membrane trace (currently not used).

            :returns: Gradient given by adjoint function lambda_i of current.
            """
            # input and layer data
            input_current = ctx.saved_tensors[0][0]
            T, _, _ = input_current.shape
            z = ctx.saved_tensors[1]
            params = ctx.extra_kwargs["params"]
            dt = ctx.extra_kwargs["dt"]

            # adjoints
            lambda_v = torch.zeros_like(input_current)
            lambda_i = torch.zeros_like(input_current)

            # When executed on hardware, spikes and membrane voltage are injected but the synaptic
            # current is not recorded. Approximate it:
            if ctx.saved_tensors[3] is not None:
                i = ctx.saved_tensors[3]
            else:
                i = torch.zeros_like(z)
                # compute current
                for ts in range(T - 1):
                    i[ts + 1] = \
                        i[ts] * (1 - dt * params.tau_syn_inv) \
                        + input_current[ts]

            for ts in range(T - 1, 0, -1):
                dv_m = params.v_leak - params.v_th + i[ts - 1]
                dv_p = params.v_leak - params.v_reset + i[ts - 1]

                lambda_i[ts - 1] = lambda_i[ts] + dt * \
                    params.tau_syn_inv * (lambda_v[ts] - lambda_i[ts])
                lambda_v[ts - 1] = lambda_v[ts] * \
                    (1 - dt * params.tau_mem_inv)

                output_term = z[ts] / dv_m * grad_spikes[ts]
                output_term[torch.isnan(output_term)] = 0.0

                jump_term = z[ts] * dv_p / dv_m
                jump_term[torch.isnan(jump_term)] = 0.0

                lambda_v[ts - 1] = (
                    (1 - z[ts]) * lambda_v[ts - 1]
                    + jump_term * lambda_v[ts - 1]
                    + output_term
                )
            return torch.stack((lambda_i / params.tau_syn_inv,
                                lambda_v - lambda_i)), None, None


    class EventPropSynapse(torch.autograd.Function):
        """
        Synapse function for proper gradient transport when using EventPropNeuron.
        """
        @staticmethod
        def forward(ctx, input: torch.Tensor, weight: torch.Tensor,
                    _: torch.Tensor = None
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            This should be used in combination with EventPropNeuron. Multiply input
            with weight and use a stacked output in order to be able to return two
            tensors (separate terms in EventProp algorithm), one for previous layer
            and the other one for weights.

            :param input: Input spikes in shape (batch, time, in_neurons).
            :param weight: Weight in shape (out_neurons, in_neurons).
            :param _: Bias, which is unused here.

            :returns: Returns stacked tensor holding weighted spikes and
                tensor with zeros but same shape.
            """
            ctx.save_for_backward(input, weight)
            output = input.matmul(weight.t())
            return torch.stack((output, torch.zeros_like(output)))

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor,
                    ) -> Tuple[Optional[torch.Tensor],
                                Optional[torch.Tensor]]:
            """
            Split gradient_output coming from EventPropNeuron and return
            weight * (lambda_v - lambda_i) as input gradient and
            - tau_s * lambda_i * input (i.e. - tau_s * lambda_i at spiketimes)
            as weight gradient.

            :param grad_output: Backpropagated gradient with shape (2, batch, time,
                out_neurons). The 2 is due to stacking in forward.

            :returns: Returns gradients w.r.t. input, weight and bias (None).
            """
            input, weight = ctx.saved_tensors
            grad_input = grad_weight = None

            if ctx.needs_input_grad[0]:
                grad_input = grad_output[1].matmul(weight)
            if ctx.needs_input_grad[1]:
                grad_weight = \
                    grad_output[0].transpose(0, 1).transpose(1, 2).matmul(
                        input.transpose(0, 1))

            return grad_input, grad_weight, None

.. code:: ipython3

    class EventPropSNN(SNN):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # use EventProp in hidden (spiking) LIF layer
            self.linear_h.func = EventPropSynapse
            self.lif_h.func = EventPropNeuron
            # trace information is not used in EventProp ->  disable cadc recording
            # of hidden layer
            self.lif_h._enable_cadc_recording = False

.. code:: ipython3

    N_HIDDEN      = 120
    MOCK          = False
    DT            = 0.5e-06  # s

    # We need to specify the device we want to use on the host computer
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # The SNN using EventProp functions
    snn = EventPropSNN(
        n_in=5,
        n_hidden=N_HIDDEN,
        n_out=3,
        mock=MOCK,
        dt=DT,
        tau_mem=6.0e-06,
        tau_syn=6.0e-06,
        alpha=50.,
        trace_shift_hidden=int(.0e-06/DT),
        trace_shift_out=int(.0e-06/DT),
        weight_init_hidden=(0.15, 0.25),  # higher mean to ensure spiking
        weight_init_output=(0.0, 0.1),
        weight_scale=66.39,
        trace_scale=0.0147,
        input_repetitions=1 if MOCK else 5,
        device=device)
    snn

Training with EventProp
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    T_SIM   = 3.8e-05  # s
    T_EARLY = 0.2e-05  # s
    T_LATE  = 2.6e-05  # s
    T_BIAS  = 0.2e-05  # s

    # This encoder translates the points into spikes on a discrete time lattice
    encoder = CoordinatesToSpikes(
        seq_length=int(T_SIM / DT),
        t_early=T_EARLY,
        t_late=T_LATE,
        dt=DT,
        t_bias=T_BIAS)
    encoder

.. code:: ipython3

    model = Model(encoder, snn, decoder, readout_scale=10.)
    model

.. code:: ipython3

    # Training params
    LR            = 0.002
    STEP_SIZE     = 5
    GAMMA         = 0.9
    EPOCHS        = 4 # Adjust here for longer training...
    BATCH_SIZE    = 50
    TRAINSET_SIZE = 5000
    TESTSET_SIZE  = 1000

.. code-block:: ipython3
    :class: test, html-display-none

    # Training params
    LR            = 0.002
    STEP_SIZE     = 5
    GAMMA         = 0.9
    EPOCHS        = 1
    BATCH_SIZE    = 50
    TRAINSET_SIZE = 500
    TESTSET_SIZE  = 100

.. code:: ipython3

    # Just for plotting...
    assert TRAINSET_SIZE % BATCH_SIZE == 0

    # PyTorch stuff... optimizer, scheduler and loss like you normally do.
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    loss = torch.nn.CrossEntropyLoss()

    # Data loaders
    train_loader, test_loader = get_data_loaders(TRAINSET_SIZE, TESTSET_SIZE, BATCH_SIZE)

.. code:: ipython3

    # Functions to update plot
    update_plot, update_train_data, update_test_data = plot_training(N_HIDDEN, T_SIM, DT)
    plt.close()
    output = w.Output()
    display(output)

    # Initialize the hardware
    if not MOCK:
        hxtorch.init_hardware()

    # Train and test
    for epoch in range(0, EPOCHS + 1):
        # Test
        loss_test, acc_test, rate_test = test(
            model, test_loader, loss, epoch, update_test_data)

        # Refresh plot
        output.clear_output(wait=True)
        with output:
            update_plot()

        # Train epoch
        if epoch < EPOCHS:
            loss_train, acc_train = train(
                model, train_loader, loss, optimizer, epoch, update_train_data)

        scheduler.step()

    # Release the hardware connection
    hxtorch.release_hardware()
