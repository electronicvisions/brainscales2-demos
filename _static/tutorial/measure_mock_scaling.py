from typing import Optional, Tuple, NamedTuple
import pylogging as logger
from tqdm import tqdm

from scipy.optimize import curve_fit
import torch

from dlens_vx_v3 import halco, lola

# pylint: disable=import-error, no-name-in-module
import hxtorch
import hxtorch.spiking as hxsnn
from hxtorch.spiking.morphology import Morphology, SingleCompartmentNeuron


class ConstantCurrentMixin:
    current_type: lola.AtomicNeuron.ConstantCurrent.Type \
        = lola.AtomicNeuron.ConstantCurrent.Type.source
    enable_current: bool = True

    def configure_hw_entity(self, neuron_id: int,
                            neuron_block: lola.NeuronBlock,
                            coord: halco.LogicalNeuronOnDLS) \
            -> lola.NeuronBlock:
        super().configure_hw_entity(neuron_id, neuron_block, coord)
        if self.enable_current:
            for nrn in halco.iter_all(halco.AtomicNeuronOnDLS):
                neuron_block.atomic_neurons[nrn] \
                    .constant_current.i_offset = 1000
                neuron_block.atomic_neurons[nrn] \
                    .constant_current.enable = True
                neuron_block.atomic_neurons[nrn] \
                    .constant_current.type = self.current_type
        return neuron_block


class ConstantCurrentNeuron(ConstantCurrentMixin, hxsnn.Neuron):
    # pylint: disable=redefined-builtin
    def forward_func(self, input, hw_data):
        z_hw, v_cadc, v_madc = (
            data.to(input.graded_spikes.device) if data is not None
            else None for data in hw_data)
        return hxsnn.NeuronHandle(z_hw, v_cadc, None, v_madc)


class ConstantCurrentReadoutNeuron(ConstantCurrentMixin, hxsnn.ReadoutNeuron):
    pass


class Threshold:
    """ Class to measure thresholds on BSS-2 """

    def __init__(self, params: Optional = None,
                 calib_path: Optional[str] = None,
                 batch_size: int = 100,
                 neuron_structure: Morphology = SingleCompartmentNeuron(1)):
        """ """
        self.calib_path = calib_path
        self.batch_size = batch_size
        self.neuron_structure = neuron_structure
        self.params = params
        # Constants
        self.time_length = 100
        self.hw_neurons = None
        self.neuron = None
        self.traces = None
        self.thresholds = None
        self.thresholds_mean = None

        # TODO: This probably fails if morphology is complexer
        self.output_size = halco.AtomicNeuronOnDLS.size // sum(
            len(ans) for ans in
            neuron_structure.compartments.get_compartments().values())

        self.log = logger.get("much-demos-such-wow.utils.Thresholds")

    def build_model(self, inputs: torch.Tensor, exp: hxsnn.Experiment,
                    enable_current: bool = True) \
            -> hxsnn.TensorHandle:
        """ Build model to map to hardware """
        self.log.TRACE("Build model ...")

        # Layers
        synapse = hxsnn.Synapse(1, self.output_size, experiment=exp)
        self.neuron = ConstantCurrentNeuron(
            self.output_size, experiment=exp, **self.params,
            shift_cadc_to_first=False, neuron_structure=self.neuron_structure)
        self.neuron.enable_current = enable_current

        # forward
        inputs = hxsnn.NeuronHandle(spikes=inputs)
        currents = synapse(inputs)
        traces = self.neuron(currents)
        return traces

    # pylint: disable=invalid-name
    def run(self, dt: float = 1e-6):
        """ Execute forward """
        self.log.TRACE("Run experiment ...")
        hxtorch.init_hardware()  # pylint: disable=no-member

        # Run for baseline
        exp = hxsnn.Experiment(mock=False, dt=dt)
        # Load calib
        if self.calib_path is not None:
            exp.default_execution_instance.load_calib(self.calib_path)

        inputs = torch.zeros((self.time_length, self.batch_size, 1))
        baselines = self.build_model(inputs, exp, enable_current=False)
        hxsnn.run(exp, self.time_length)

        # Run for threshold
        exp = hxsnn.Experiment(mock=False, dt=dt)
        # Load calib
        if self.calib_path is not None:
            exp.default_execution_instance.load_calib(self.calib_path)

        # Explicitly load because we want to set initial config
        inputs = torch.zeros((self.time_length, self.batch_size, 1))
        traces = self.build_model(inputs, exp)

        # run
        hxsnn.run(exp, self.time_length)
        hxtorch.release_hardware()  # pylint: disable=no-member
        self.log.TRACE("Experiment ran.")

        return self.post_process(traces, baselines)

    def post_process(self, traces: torch.Tensor, baselines: torch.Tensor) \
            -> Tuple[torch.Tensor, ...]:
        """ post-process data """
        self.log.TRACE("Postprocessing ...")
        baselines = baselines.membrane_cadc.detach().mean(0).mean(0)
        self.traces = traces.membrane_cadc.detach() - baselines
        self.thresholds = torch.max(self.traces, 0)[0].mean(0)
        self.thresholds_mean = self.thresholds.mean()
        return self.thresholds_mean


class WeightScaling:
    """
    Class to measure weight scaling between SW weight and the weight on BSS-2
    """

    max_weight = lola.SynapseWeightMatrix.Value.max
    min_weight = -lola.SynapseWeightMatrix.Value.max

    # pylint: disable=too-many-arguments
    def __init__(self, params: NamedTuple, calib_path: str = None,
                 batch_size: int = 100, trace_scale: float = 1.,
                 neuron_structure: Morphology = SingleCompartmentNeuron(1)):
        """ """
        self.calib_path = calib_path
        self.batch_size = batch_size
        self.neuron_structure = neuron_structure
        self.trace_scale = trace_scale
        self.params = params

        # Constants
        self.time_length = 100
        self.hw_neurons = None

        # TODO: This probably fails if morphology is complexer
        self.output_size = halco.AtomicNeuronOnDLS.size // sum(
            len(ans) for ans in
            neuron_structure.compartments.get_compartments().values())

        self.log = logger.get("much-demos-such-wow.utils.WeightScaling")

    # pylint: disable=arguments-differ, attribute-defined-outside-init
    def execute(self, weight: int) -> torch.Tensor:
        """ Execute forward """
        self.log.TRACE(f"Run experiment with weight {weight} ...")

        # Instance
        inputs = torch.zeros(self.time_length, self.batch_size, 1)
        inputs[10, :, :] = 1
        self.synapse.weight.data.fill_(weight)

        # forward
        spikes = hxsnn.NeuronHandle(spikes=inputs)
        currents = self.synapse(spikes)
        traces = self.neuron(currents)

        hxsnn.run(self.exp, self.time_length)
        self.log.TRACE("Experiment ran.")

        return self.post_process(traces, weight)

    def post_process(self, traces: torch.Tensor, weight: float) \
            -> Tuple[torch.Tensor, ...]:
        """ post-process data """
        self.log.TRACE("Postprocessing ...")
        self.traces = traces.membrane_cadc.detach() - self.baselines
        # Get max / min PSP over time
        if weight >= 0:
            self.amp = self.traces.max(0)[0].mean(0)
        else:
            self.amp = self.traces.min(0)[0].mean(0)
        self.amp_mean = self.amp.mean()
        return self.amp_mean, self.amp, self.traces

    # pylint: disable=arguments-differ, too-many-arguments, too-many-locals
    def run(
            self, weight_step: int = 1) \
            -> Tuple[torch.Tensor, ...]:
        """ run all measurements and compute output """
        hxtorch.init_hardware()  # pylint: disable=no-member

        # Sweep weights
        self.log.INFO(f"Using weight step: {weight_step}")
        hw_weights = torch.linspace(
            self.min_weight, self.max_weight, weight_step, dtype=int)

        # HW layers
        self.exp = hxsnn.Experiment(mock=False, dt=1e-6)
        self.synapse = hxsnn.Synapse(1, self.output_size, experiment=self.exp)
        self.neuron = hxsnn.ReadoutNeuron(
            self.output_size, experiment=self.exp,
            neuron_structure=self.neuron_structure,
            trace_scale=self.trace_scale, shift_cadc_to_first=True,
            **self.params)

        # Load calib
        if self.calib_path is not None:
            self.exp.default_execution_instance.load_calib(self.calib_path)

        # Sweep
        hw_amps = torch.zeros(hw_weights.shape[0], self.output_size)
        self.baselines = 0
        self.log.INFO("Begin hardware weight sweep...")
        pbar = tqdm(total=hw_weights.shape[0])
        for i, weight in enumerate(hw_weights):
            # Measure
            _, hw_amps[i], _ = self.execute(weight)
            # Update
            pbar.set_postfix(
                weight=f"{weight}", mean_amp=float(hw_amps[i].mean()))
            pbar.update()
        pbar.close()
        self.log.INFO("Hardware weight sweep finished.")

        # Fit
        self.log.INFO("Fit hardware data...")
        hw_scales = torch.zeros(self.output_size)
        for nrn in range(self.output_size):
            popt, _ = curve_fit(  # pylint: disable=unbalanced-tuple-unpacking
                f=lambda x, a: a * x, xdata=hw_weights.numpy(),
                ydata=hw_amps[:, nrn].numpy())
            hw_scales[nrn] = popt[0]

        # Mock values
        self.exp = hxsnn.Experiment(mock=True, dt=1e-6)
        self.synapse = hxsnn.Synapse(1, self.output_size, experiment=self.exp)
        self.neuron = hxsnn.ReadoutNeuron(
            self.output_size, experiment=self.exp,
            trace_scale=self.trace_scale, shift_cadc_to_first=False,
            **self.params)

        self.log.INFO("Begin mock weight sweep...")
        sw_weights = torch.arange(-1, 1 + 0.1, 0.1)
        # Software amplitudes
        self.baselines = self.params["leak"].model_value
        sw_amps = torch.zeros(sw_weights.shape[0], self.output_size)
        pbar = tqdm(total=sw_weights.shape[0])
        for i, weight in enumerate(sw_weights):
            # Measure
            _, sw_amps[i], _ = self.execute(weight)
            pbar.set_postfix(
                weight=f"{weight}", mean_amp=float(sw_amps[i].mean()))
            pbar.update()
        pbar.close()

        # SW scale
        sw_scales = torch.zeros(self.output_size)
        for nrn in range(self.output_size):
            popt, _ = curve_fit(  # pylint: disable=unbalanced-tuple-unpacking
                f=lambda x, a: a * x, xdata=sw_weights.numpy(),
                ydata=sw_amps[:, nrn].numpy())
            sw_scales[nrn] = popt[0]

        # Resulting scales
        scales = sw_scales / hw_scales

        self.log.INFO(
            f"Mock scale: {sw_scales}, HW scale: {hw_scales.mean()}"
            + f" +- {hw_scales.std()}")
        self.log.INFO(f"SW -> HW translation factor: {scales.mean()}")

        hxtorch.release_hardware()  # pylint: disable=no-member

        return scales.mean()


def get_weight_scaling(
        params, weight_step: int = 1,
        neuron_structure: Morphology = SingleCompartmentNeuron(1)) -> float:

    scale_runner = WeightScaling(
        params, None, neuron_structure=neuron_structure)
    weight_scale = scale_runner.run(weight_step=weight_step)

    threshold_runner = Threshold(
        params, None, neuron_structure=neuron_structure)
    threshold_hw = threshold_runner.run()

    # Compute effective weight scaling between SW and HW weights
    return weight_scale.item() * (
        threshold_hw.item() / params["threshold"].model_value)


def get_trace_scaling(
        neuron_structure: Morphology = SingleCompartmentNeuron(1),
        params: Optional = None) -> float:
    threshold_runner = Threshold(
        params, None, neuron_structure=neuron_structure)
    threshold_hw = threshold_runner.run()
    return params["threshold"].model_value / threshold_hw.item()
