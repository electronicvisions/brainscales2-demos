# pylint: disable=too-many-lines
# A significant portion of lines is docstrings anyway...

"""
Helper functions and imports for the Pong demo notebook.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional
import enum
import textwrap

import ipycanvas
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pynn_brainscales.brainscales2 as pynn

from dlens_vx_v3 import hal


@dataclass
class Parameters:
    """
    Parameters for experiment.

    :ivar noise_range_epochs: Number of total epochs in which noise is
        reduced from start to end. Note that this really only refers to
        the scaling of noise, the actual training can use a different
        number of epochs.
    :ivar n_inputs: Number of input neurons.
    :ivar n_outputs: Number of output neurons.
    :ivar input_distribution: Select distribution of input events across
        multiple rows.
        In order to not depend on all correlation sensors etc. working
        perfectly, we distribute the input events across multiple synapse
        rows. This list contains the relative event rates for further
        rows. The length of this list is arbitrary.
        Example: A configuration `input_distribution = [1, 0.7, 0.3]`
        will send the full rate to the middle row, 0.7 times the full rate
        to the rows one above and below, and 0.3 times the full rate to the
        rows two above and below.
    :ivar n_events: Number of input events sent to middle ("target") input
        row.
    :ivar wait_between_events: Inter-spike interval for input events in
        middle input row.
    :ivar wait_between_noise: Inter-spike interval for noise events. The
        noise is applied for the same time-duration as the inputs.
    :ivar noise_range_start: Start value for standard deviation of noise
        weight distribution. The noise weights are drawn from an
        approximated normal distribution. The standard deviation is
        reduced linearly during training, from the configured start to
        end values.
    :ivar noise_range_end: Corresponding end value for noise range.
    :ivar rewards: Reward for each neuron depending on the distance from
        the target neuron. The number of spikes of reach neuron is
        multiplied with the factor given here, and the sum of these
        products is the reward for an experiment run. This variable is
        a dict mapping the distance from the target neuron to the reward.
        Example: ```{0: 4, 1: 2, 2: 0, "elsewhere": -1}``` indicates that
        the reward is 4 at the target neuron, 2 for the neighboring
        neurons, 0 for the neurons that are 2 left or right of the
        target, and -1 everywhere else. The reward for "elsewhere" must
        be below 0 to calculate the game success correctly. All rewards
        below zero are assumed to have missed the paddle, while zero and
        greater rewards are assumed to have hit the paddle.
    :ivar learning_rate: Learning rate.
    :ivar homeostasis_target: Number of target spikes per column in an
        epoch for homeostasis.
    :ivar homeostasis_rate: Update rate for homeostasis, in weight LSB.
    :ivar reward_decay: Update rate of average reward in each epoch.
        The reward is updated with the new, instantaneous reward in
        each epoch. A high decay means that the average reward closely
        follows the state in each epoch, a low decay means that the
        average reward changes only slowly with the new results in each
        epoch. The mean reward is updated based on the instantaneous
        reward r as follows:
        mean_reward = (1 - reward_decay) * mean_reward + reward_decay * r
    :ivar reward_initializaion_phase: Number of epochs that are executed
        initially without learning, in order to initialize the average
        reward.
    :ivar init_duration: Scheduled time for initialization, such as
        correlation resets. Values around 1 ms seem to work well.
    :ivar plasticity_duration: Scheduled time for handling plasticity
        kernel. This is mostly constrained by the CADC readout for each
        row. Values around 5 ms seem to work well.
    :ivar epochs_per_run: Number of epochs executed within one pynn.run()
        call. Note: The learning rate is only updated at compile-time,
        i.e. will not be updated during the epochs within a PyNN run.
    """

    noise_range_epochs: int = 2000
    n_inputs: int = 250
    n_outputs: int = 250
    input_distribution: List = field(default_factory=lambda: [1, 0.7, 0.3])
    n_events: int = 140
    wait_between_events: pq.Quantity = field(
        default_factory=lambda: 4.20 * pq.us)
    wait_between_noise: pq.Quantity = field(
        default_factory=lambda: 3.5 * pq.us)
    noise_range_start: int = 15
    noise_range_end: int = 4
    rewards: List = field(default_factory=lambda: {
        0: 4,  # at target neuron
        1: 2,  # 1 left or right of target
        2: 1,  # 2 left or right of target
        3: 1,  # ...
        4: 0,
        "elsewhere": -1})
    learning_rate: float = 0.01
    homeostasis_target: int = 750
    homeostasis_rate: float = 1. / homeostasis_target
    reward_decay: float = 0.5
    reward_initialization_phase: int = 10
    init_duration: pq.Quantity = field(default_factory=lambda: 1 * pq.ms)
    plasticity_duration: pq.Quantity = field(default_factory=lambda: 7 * pq.ms)
    epochs_per_run: int = 6

    @property
    def input_duration(self) -> pq.Quantity:
        """
        Time duration of input spiketrain.
        """

        return self.n_events * self.wait_between_events

    @property
    def row_duration(self) -> pq.Quantity:
        """
        Time required to process one input row.
        """

        return self.init_duration + self.input_duration \
            + self.plasticity_duration

    @property
    def epoch_duration(self) -> pq.Quantity:
        """
        Time required for a whole epoch.
        """

        return self.row_duration * self.n_inputs

    def get_noise_range(self, current_epoch: int) -> float:
        """
        Get current standard deviation for noise distribution.

        Noise is decreased linearly from start to end.

        :param current_epoch: Currently running epoch.
        :return: Range of weights to be applied in noise row.
        """

        epoch_progress = current_epoch / self.noise_range_epochs

        return (1 - epoch_progress) * self.noise_range_start + \
            epoch_progress * self.noise_range_end

    def get_reward_decay(self, current_epoch: int) -> float:
        """
        Get current reward decay.

        Reward decay is initially set to 1 and decreased linearly to the
        desired value within the first few epochs.

        :param current_epoch: Currently running epoch.
        :return: Reward decay.
        """

        if current_epoch >= self.reward_initialization_phase:
            return self.reward_decay

        return 1 - ((1 - self.reward_decay) * current_epoch
                    / self.reward_initialization_phase)

    def get_learning_rate(self, current_epoch: int) -> float:
        """
        Get current learning rate.

        Learning rate is zero in the first few epochs, when
        the reward array is still initialized.

        :param current_epoch: Currently running epoch.
        :return: Learning rate.
        """

        if current_epoch >= self.reward_initialization_phase:
            return self.learning_rate
        return 0.

    def diagonal_entries(self) -> np.ndarray:
        """
        Mask for obtaining diagonal entries from all synapse weights.

        Returns a boolean mask which is True for entries that are on the
        diagonal or one above or below, and False everywhere else.

        :return: Mask for diagonal entries.
        """

        diagonal_entries = np.any([
            np.eye(self.n_inputs, k=k, dtype=bool)
            for k in range(-1, 2)], axis=0)
        if self.n_outputs > self.n_inputs:
            diagonal_entries = np.pad(
                diagonal_entries,
                [(0, 0), (0, self.n_outputs - self.n_inputs)])
        elif self.n_inputs > self.n_outputs:
            diagonal_entries = np.pad(
                diagonal_entries,
                [(0, self.n_inputs - self.n_outputs), (0, 0)])

        return diagonal_entries

    def off_diagonal_entries(self) -> np.ndarray:
        """
        Mask for obtaining off-diagonal entries from all synapse weights.

        Returns a boolean mask that is False for entries that are on the
        diagonal or within twice the size of ball and paddle above or
        below it, and True everywhere else.

        :return: Mask for off-diagonal entries.
        """

        distance_to_diagonal = 2 * max(
            self.paddle_size(), len(self.input_distribution))
        off_diagonal_entries = np.all(
            [~np.eye(self.n_inputs, k=k, dtype=bool)
             for k in range(-distance_to_diagonal, distance_to_diagonal + 1)],
            axis=0)
        if self.n_outputs > self.n_inputs:
            off_diagonal_entries = np.pad(
                off_diagonal_entries,
                [(0, 0), (0, self.n_outputs - self.n_inputs)])
        elif self.n_inputs > self.n_outputs:
            off_diagonal_entries = np.pad(
                off_diagonal_entries,
                [(0, self.n_inputs - self.n_outputs), (0, 0)])
        return off_diagonal_entries

    def paddle_size(self) -> int:
        """
        Return size of the paddle, from the middle to the edges,
        based on the selected rewards.
        """

        distances = []
        for key, value in self.rewards.items():
            if value >= 0:
                distances.append(key)
        return max(distances)


@dataclass
class ExperimentData:
    """
    Archive for data accumulated during the training, and functions
    for plotting.
    """

    epochs_per_run: int

    # Success (i.e. did the pong ball hit the paddle?) during training
    success_archive: List = field(default_factory=lambda: [])

    # Reward during training
    reward_archive: List = field(default_factory=lambda: [])

    # Weight trend: diagonal and off-diagonal means
    mean_diagonal_weight: List = field(default_factory=lambda: [])
    mean_off_diagonal_weight: List = field(default_factory=lambda: [])

    @property
    def n_epochs_trained(self) -> int:
        """
        Return number of already trained epochs.
        """

        return len(self.success_archive)

    # pylint: disable=too-many-statements
    def generate_plot(self, plot_weights: np.ndarray):
        """
        Plot weights and other training metrics.

        :param plot_weights: Latest synapse weights after training.
        """

        if self.n_epochs_trained == 0:
            raise ValueError("Train at least one epoch before plotting!")

        fig, axes = plt.subplots(
            figsize=(6, 7), nrows=3,
            gridspec_kw={"height_ratios": [0.5, 0.25, 0.25]})
        divider = make_axes_locatable(axes[0])
        cax = divider.append_axes("right", size="10%", pad=0.1)
        image = axes[0].imshow(plot_weights, origin="upper")
        fig.colorbar(image, cax=cax)
        axes[0].set_xlabel("output neuron")
        axes[0].set_ylabel("input row")
        axes[0].set_title(
            f"Weights after epoch {self.n_epochs_trained}")

        axes[1].plot(np.mean(self.reward_archive, axis=1), color="black")
        axes[1].set_ylabel("Average reward")
        axes[1].set_xlabel("Epoch")

        ax12 = axes[1].twinx()
        ax12.plot(np.mean(self.success_archive, axis=1),
                  color="red")
        ax12.axhline(1, linewidth=0.5, color="red")
        ax12.yaxis.label.set_color('red')
        ax12.tick_params(axis='y', colors='red')
        ax12.set_ylabel("Average success", color="red")
        ax12.set_ylim(-0.05, 1.05)

        # We do not record the weights at the begining of the training ->
        # shift epochs by number of epochs per run; subtract 1 since we start
        # counting at 0
        epochs = np.arange(self.epochs_per_run - 1,
                           self.n_epochs_trained, self.epochs_per_run)
        axes[2].plot(epochs, self.mean_diagonal_weight,
                     label="diagonal weight", color="black")
        axes[2].plot(epochs, self.mean_off_diagonal_weight,
                     label="off-diagonal weight", color="red")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Average weight [LSB]")
        axes[2].legend()

        return fig, np.append(axes, ax12), image

    def update_plot(
            self,
            plot: Tuple[plt.Figure, plt.Axes, matplotlib.image.AxesImage],
            plot_weights: np.ndarray):
        """
        Update an existing figure with new data.
        """

        fig, axes, image = plot

        image.set_data(plot_weights)
        image.set_norm(matplotlib.colors.Normalize(
            np.min(plot_weights), np.max(plot_weights)))
        axes[0].set_title(
            f"Weights after epoch {self.n_epochs_trained}")

        axes[1].lines[0].set_xdata(np.arange(self.n_epochs_trained))
        axes[1].set_xlim(-1, self.n_epochs_trained + 1)
        new_data = np.mean(self.reward_archive, axis=1)
        axes[1].lines[0].set_ydata(new_data)
        axes[1].set_ylim(np.min(new_data) - 1, np.max(new_data) + 1)
        axes[3].lines[0].set_xdata(np.arange(self.n_epochs_trained))
        axes[3].set_xlim(-1, self.n_epochs_trained + 1)
        axes[3].lines[0].set_ydata(np.mean(self.success_archive, axis=1))

        # We do not record the weights at the begining of the training ->
        # shift epochs by number of epochs per run; subtract 1 since we start
        # counting at 0
        epochs = np.arange(self.epochs_per_run - 1,
                           self.n_epochs_trained, self.epochs_per_run)
        axes[2].lines[0].set_xdata(epochs)
        axes[2].set_xlim(-1, self.n_epochs_trained + 1)
        axes[2].lines[0].set_ydata(self.mean_diagonal_weight)
        axes[2].lines[1].set_xdata(epochs)
        axes[2].lines[1].set_ydata(self.mean_off_diagonal_weight)
        axes[2].set_ylim(
            -0.2, np.max([self.mean_diagonal_weight,
                          self.mean_off_diagonal_weight]) + 0.2)

        display(fig)  # pylint: disable=undefined-variable


@dataclass
class InferenceTiming:
    """
    Dataclass containing timing properties that are used in an inference
    run.

    They control the timing of PPU programs that only read out the spike
    counters, as we don't need any plasticity here.

    We use these timings during the animation of the pong game, where we
    do not add artificial noise.

    The default values were obtained by reducing the timings until the
    results looked broken (i.e. spike counts from the bottom synram were
    no longer sensible).

    :ivar input_duration: Duration for the inputs of a single ball
        position.
    :ivar init_duration: Estimated time for resetting the spike counters.
    :ivar readout_duration: Estimated time how long reading the spike
        counters takes.
    """

    input_duration: pq.Quantity
    init_duration: pq.Quantity = field(default_factory=lambda: 0.5 * pq.ms)
    readout_duration: pq.Quantity = field(default_factory=lambda: 3 * pq.ms)

    @property
    def row_duration(self) -> pq.Quantity:
        """
        Time required to process one input row.
        """

        return self.init_duration + self.input_duration \
            + self.readout_duration


class InferenceInitRule(pynn.PlasticityRule):
    """
    Reset spike counters.
    """

    def generate_kernel(self) -> str:
        """
        Generate plasticity rule kernel to be compiled into PPU program.

        :return: PPU-code of plasticity-rule kernel as string.
        """
        return textwrap.dedent("""
        #include "grenade/vx/ppu/synapse_array_view_handle.h"
        #include "grenade/vx/ppu/neuron_view_handle.h"
        #include "libnux/vx/reset_neurons.h"
        #include "stadls/vx/v3/ppu/write.h"

        using namespace grenade::vx::ppu;
        using namespace libnux::vx;

        void PLASTICITY_RULE_KERNEL(
            std::array<SynapseArrayViewHandle, 1>& /* synapses */,
            std::array<NeuronViewHandle, 0>& /* neurons */)
        {{
            reset_neurons();

            // reset spike counters
            for (auto coord : halco::common::iter_all<
                     halco::hicann_dls::vx::v3::SpikeCounterResetOnDLS>())
            {{
                stadls::vx::v3::ppu::write(
                    coord, haldls::vx::v3::SpikeCounterReset());
            }}
        }}
        """)


class InferenceReadoutRule(pynn.PlasticityRule):
    """
    Read out spike counters, record index of most-active neuron.

    :ivar neuron_ids: IDs (enum-values of AtomicNeuron coordinates) of
        neurons to read spike counter values from.
    """

    def __init__(self, timer: pynn.Timer, neuron_ids: np.ndarray):
        """
        Initialize plastic synapse with execution timing information,
        hyperparameters and initial weight.
        """

        observables = {
            "winner_neuron": pynn.PlasticityRule.ObservableArray(),
        }
        observables["winner_neuron"].type = \
            pynn.PlasticityRule.ObservableArray.Type.uint8
        observables["winner_neuron"].size = 1

        super().__init__(timer=timer, observables=observables)

        self.neuron_ids = neuron_ids

    def generate_kernel(self) -> str:
        """
        Generate plasticity rule kernel to be compiled into PPU program.

        :return: PPU-code of plasticity-rule kernel as string.
        """
        return textwrap.dedent(f"""
        #include <algorithm>
        #include "grenade/vx/ppu/synapse_array_view_handle.h"
        #include "grenade/vx/ppu/neuron_view_handle.h"
        #include "libnux/vx/reset_neurons.h"
        #include "stadls/vx/v3/ppu/write.h"
        #include "stadls/vx/v3/ppu/read.h"

        using namespace grenade::vx::ppu;
        using namespace libnux::vx;

        template <size_t N>
        void PLASTICITY_RULE_KERNEL(
            std::array<SynapseArrayViewHandle, N>& /* synapses */,
            std::array<NeuronViewHandle, 0>& /* neurons */,
            Recording& recording)
        {{
            // read out spike counters
            size_t spike_counts[{len(self.neuron_ids)}];
            size_t item_counter = 0;
            for (size_t i : {{ {', '.join(self.neuron_ids.astype(str))} }})
            {{
                auto coord = halco::hicann_dls::vx::v3::AtomicNeuronOnDLS(
                    halco::common::Enum(i)).toSpikeCounterReadOnDLS();
                haldls::vx::v3::SpikeCounterRead container =
                    stadls::vx::v3::ppu::read<
                    haldls::vx::v3::SpikeCounterRead>(coord);
                spike_counts[item_counter] = container.get_count();
                item_counter++;
            }}

            uint8_t winner_neuron =
                std::max_element(&spike_counts[0],
                                 &spike_counts[{len(self.neuron_ids)}])
                - &spike_counts[0];
            recording.winner_neuron[0] = winner_neuron;
        }}
        """)


# Dimensions of the pong game field, as used by the following classes:
GAME_WIDTH = 800
GAME_HEIGHT = 500


class PongBall:
    """
    Class representing the ball in a pong game.

    :cvar ball_speed: Baseline for the speed of the ball, there will be
        some randomness (roughly +- 3) added on top.
    :ivar x_position: Horizontal position of the ball (center).
    :ivar y_position: Vertical position of the ball (center).
    :ivar v_x: Horizontal speed of the ball.
    :ivar v_y: Vertical speed of the ball.
    :ivar size: Rendered size of the ball, corresponds to number of input
        rows supplied with spikes in the network.
    """

    ball_speed = 10

    def __init__(self, parameters: Parameters):
        self.x_position = np.random.randint(0, GAME_WIDTH)
        self.y_position = np.random.randint(0, GAME_HEIGHT)
        self.v_x = self.ball_speed \
            + 3 * np.random.random() * np.random.choice([1, -1])
        self.v_y = self.ball_speed \
            * np.random.random() * np.random.choice([1, -1])
        self.size = GAME_HEIGHT / parameters.n_inputs \
            * (len(parameters.input_distribution) - 0.5) / 2

    def step(self) -> int:
        """
        Move the ball by one step.

        The ball is animated between left and right walls, i.e. as if the
        paddle always was at the correct position. This allows animating
        the whole game in advance, and only checking the correct paddle
        positions later.

        :return: Number of collitions with the left or right wall/paddle.
        """

        n_collisions = 0

        # collide with left wall
        if self.x_position + self.v_x < PongPaddle.paddle_width:
            self.v_x = self.ball_speed + 2 * np.random.random()
            self.v_y += np.random.random()
            n_collisions += 1

        # collide with right wall
        if self.x_position + self.v_x > GAME_WIDTH - PongPaddle.paddle_width:
            self.v_x = -self.ball_speed - 2 * np.random.random()
            self.v_y += np.random.random()
            n_collisions += 1

        # collide with top/bottom wall
        if self.y_position + self.v_y > GAME_HEIGHT \
                or self.y_position + self.v_y < 0:
            self.v_y *= -1

        self.x_position += self.v_x
        self.y_position += self.v_y

        return n_collisions

    def draw(self, canvas: ipycanvas.Canvas,
             x_position: Optional[float] = None,
             y_position: Optional[float] = None):
        """
        Draw ball on canvas.
        """

        canvas.fill_circle(x_position or self.x_position,
                           y_position or self.y_position, self.size)


class PongPaddle:
    """
    Class representing a paddle in a pong game.

    :cvar paddle_width: Horizontal width of the paddle.

    :ivar x: Horizontal position of the paddle (left edge).
    :ivar y: Vertical position of the paddle (center).
    :ivar height_scaling: Scaling factor for game/canvas positions to
        neuron IDs, which are controlling the paddle.
    :ivar paddle_height: Vertical size (from center to edge) of the paddle.
        Corresponds to the number of non-negative rewards in the network.
    """

    paddle_width = 5

    def __init__(self, parameters: Parameters, x_position: int):
        self.x_position = x_position
        self.y_position = np.random.randint(0, GAME_HEIGHT)
        self.height_scaling = GAME_HEIGHT / parameters.n_inputs
        self.paddle_height = self.height_scaling * parameters.paddle_size()

    def move_to(self, neuron_id: int):
        """
        Move paddle to the position corresponding to the given neuron.

        :param neuron_id: ID of neuron to move to.
        """

        self.y_position = self.height_scaling * neuron_id

    def collide(self, ball_y: float) -> bool:
        """
        Check whether the ball has hit the paddle.

        :param ball_y: Vertical position of the ball.

        :return: Boolean decision whether the ball has hit the paddle.
        """

        return self.y_position - self.paddle_height <= ball_y \
            <= self.y_position + self.paddle_height

    def draw(self, canvas: ipycanvas.Canvas):
        """
        Draw paddle on canvas.
        """

        canvas.fill_rect(self.x_position,
                         self.y_position - self.paddle_height,
                         self.paddle_width, 2 * self.paddle_height)


class Result(enum.Enum):
    """
    Possible results from a single pong game.
    """

    TIE = enum.auto()
    LEFT_WON = enum.auto()
    RIGHT_WON = enum.auto()


class PongGame:
    """
    Class containing all necessities for the animation (and emulation) of
    a pong game.

    :ivar demo: Select a demo mode, where an ideal game is displayed
        without interacting with the chip.
    :ivar parameters: Parameters for the experiment.
    :ivar pynn_config: Parameters for pynn.setup call, containing
        calibration.
    :ivar spiketrains_per_input: List of input spiketrains for each input
        row, per ball position. Generated for training in the notebook
        already, supplied as a variable here to save runtime.
    :ivar _score_left: Score of left player.
    :ivar _score_right: Score of right player.
    :ivar _score_updated: Stores whether the scores have been updated, as
        this happens during the animation and it can be played repeatedly.
    :ivar ball: Instance of a pong ball.
    :ivar paddle_left: Instance of a pong paddle, for left player.
    :ivar paddle_right: Instance of a pong paddle, for right player.
    :ivar n_steps: Number of animation steps to run the game for.
    :ivar ball_positions: Array of ball positions, containing its
        coordinates for the whole game.
    :ivar collisions: List of time steps where collisions between ball
        and left or right walls happen, i.e. collisions where there
        could be a paddle or not.
    :ivar inference_steps: Array of time steps where inference network
        runs are requested. Inference runs happen regularly in a specific
        interval and at collisions.
    :ivar inference_results: Paddle positions resulting from the
        inference runs.
    :ivar canvases: Ipython Multicanvas to draw and animate the game.
    """

    def __init__(self, demo: bool = False,
                 parameters: Optional[Parameters] = None,
                 pynn_config: Optional[Dict] = None,
                 spiketrains_per_input:
                 Optional[List[List[np.ndarray]]] = None):
        self.demo = demo
        self.parameters = parameters or Parameters()
        self.pynn_config = pynn_config or {}
        self.spiketrains_per_input = [] if spiketrains_per_input is None \
            else np.array(spiketrains_per_input, dtype=object)
        self._score_left = 0
        self._score_right = 0
        self._score_updated = False
        self.ball = PongBall(self.parameters)
        self.paddle_left = PongPaddle(self.parameters, 0)
        self.paddle_right = PongPaddle(
            self.parameters, GAME_WIDTH - PongPaddle.paddle_width)
        self.n_steps = 4000
        self.ball_positions = np.empty((self.n_steps + 1, 2))
        self.collisions = []
        self.inference_steps: Optional[np.ndarray] = None
        self.inference_results: Optional[np.ndarray] = None
        self.canvases: Optional[ipycanvas.MultiCanvas] = None

        if not self.demo:
            if len(self.spiketrains_per_input) != self.parameters.n_inputs:
                raise ValueError(
                    "Given parameters and input spiketrains are incompatible!")

    def reset(self):
        """
        Reset state, ahead of a new game. The score is not reset.
        """

        self.ball = PongBall(self.parameters)
        self.inference_steps: Optional[np.ndarray] = None
        self.inference_results: Optional[np.ndarray] = None
        self.collisions = []
        self._score_updated = False

    def simulate_ball(self):
        """
        Simulate positions of the ball throughout the game.

        Fills timesteps for collisions and requests inference steps
        accordingly.
        """

        for step_id in range(self.n_steps + 1):
            self.ball_positions[step_id] = [
                self.ball.x_position, self.ball.y_position]
            n_collisions = self.ball.step()
            if n_collisions > 0:
                self.collisions.append(step_id)

        self.inference_steps = np.unique(np.concatenate(
            (self.collisions, np.arange(0, self.n_steps + 1, 20))))

    def simulate_perfect_game(self):
        """
        Calculate paddle positions for a perfect game.
        """

        # treat y-positions of the ball as outputs, repeat array
        # to move both paddles
        self.inference_results = np.repeat(
            self.ball_positions[self.inference_steps, 1][np.newaxis]
            / (GAME_HEIGHT / self.parameters.n_outputs),
            2, axis=0)

        # stop paddle if the ball moves away from it
        for collision_step, next_collision_step in zip(
                self.collisions[:-1], self.collisions[1:]):
            ball_moves_left = (self.ball_positions[next_collision_step, 0]
                               - self.ball_positions[collision_step, 0]) < 0
            if ball_moves_left:
                self.inference_results[
                    1, np.searchsorted(
                        self.inference_steps, collision_step
                    ):np.searchsorted(
                        self.inference_steps, next_collision_step)] = \
                    self.inference_results[
                        1, np.searchsorted(self.inference_steps,
                                           collision_step)]
            else:
                self.inference_results[
                    0, np.searchsorted(
                        self.inference_steps, collision_step
                    ):np.searchsorted(
                        self.inference_steps, next_collision_step)] = \
                    self.inference_results[
                        0, np.searchsorted(self.inference_steps,
                                           collision_step)]

    def _generate_input_spiketrains(self) -> List[np.ndarray]:
        """
        Generate a list of spiketrains, that send spikes to the correct
        input rows depending on the ball positions for inference.

        :return: List of spiketrains for each input row.
        """

        inference_timing = InferenceTiming(
            input_duration=self.parameters.input_duration)
        spiketrains_per_step = []
        for input_id, input_position in enumerate(
                self.ball_positions[self.inference_steps, 1]):
            spiketrains_per_step.append(
                self.spiketrains_per_input[int(
                    input_position / GAME_HEIGHT * self.parameters.n_inputs)]
                + input_id
                * inference_timing.row_duration.rescale(pq.ms).magnitude)

        spiketrains = []
        for row in range(self.parameters.n_inputs):
            spiketrains.append(np.concatenate(
                [spiketrains_per_step[i][row]
                 for i in range(len(self.inference_steps))]))

        return spiketrains

    def run_network(self, learned_weights: np.ndarray):
        """
        Run the network to obtain paddle positions for all inference
        steps.

        The left paddle is controlled by an ideal matrix (maximum weight
        along the diagonal, zero otherwise), the right paddle is
        controlled by the given learned weights.

        :param learned_weights: Weight matrix for right player.
        """

        if self.inference_steps is None:
            raise ValueError(
                "Must simulate ball movement before running the network!")
        if learned_weights.shape != (self.parameters.n_inputs,
                                     self.parameters.n_outputs):
            raise ValueError(
                "Supplied learned weights have a bad shape: "
                + f"{learned_weights.shape}. Expected shape: ",
                f"({self.parameters.n_inputs}, {self.parameters.n_outputs})")

        inference_timing = InferenceTiming(
            input_duration=self.parameters.input_duration)
        inference_init_timer = pynn.Timer(
            start=0,
            period=inference_timing.row_duration.rescale(pq.ms).magnitude,
            num_periods=len(self.inference_steps))
        inference_readout_timer = pynn.Timer(
            start=(self.parameters.input_duration
                   + inference_timing.init_duration).rescale(pq.ms).magnitude,
            period=inference_timing.row_duration.rescale(pq.ms).magnitude,
            num_periods=len(self.inference_steps))

        pynn.setup(**self.pynn_config)

        spiketrains = self._generate_input_spiketrains()

        # setup input and output populations and projections
        pop_output_right = pynn.Population(
            self.parameters.n_outputs, pynn.cells.HXNeuron())
        pop_output_left = pynn.Population(
            self.parameters.n_outputs, pynn.cells.HXNeuron())

        pop_input = pynn.Population(
            self.parameters.n_inputs,
            pynn.cells.SpikeSourceArray(spike_times=spiketrains))
        pop_init = pynn.Population(
            1, pynn.cells.SpikeSourceArray(spike_times=[]))

        synapses_right = pynn.standardmodels.synapses.PlasticSynapse(
            plasticity_rule=InferenceReadoutRule(
                timer=inference_readout_timer,
                neuron_ids=np.array([
                    int(pynn.simulator.state.neuron_placement
                        .id2logicalneuron(index)
                        .get_atomic_neurons()[0].toEnum())
                    for index in range(pop_output_right.first_id,
                                       pop_output_right.last_id + 1)])),
            weight=learned_weights)
        pynn.Projection(
            pop_input, pop_output_right, pynn.AllToAllConnector(),
            synapse_type=synapses_right, receptor_type="excitatory")

        ideal_weights = np.zeros(
            (self.parameters.n_inputs, self.parameters.n_outputs), dtype=int)
        ideal_weights[self.parameters.diagonal_entries()] = \
            hal.SynapseQuad.Weight.max

        synapses_left = pynn.standardmodels.synapses.PlasticSynapse(
            plasticity_rule=InferenceReadoutRule(
                timer=inference_readout_timer,
                neuron_ids=np.array([
                    int(pynn.simulator.state.neuron_placement
                        .id2logicalneuron(index)
                        .get_atomic_neurons()[0].toEnum())
                    for index in range(pop_output_left.first_id,
                                       pop_output_left.last_id + 1)])),
            weight=ideal_weights)
        pynn.Projection(
            pop_input, pop_output_left, pynn.AllToAllConnector(),
            synapse_type=synapses_left, receptor_type="excitatory")

        synapses_init = pynn.standardmodels.synapses.PlasticSynapse(
            plasticity_rule=InferenceInitRule(timer=inference_init_timer),
            weight=0)
        pynn.Projection(
            pop_init, pop_output_right, pynn.AllToAllConnector(),
            synapse_type=synapses_init, receptor_type="excitatory")

        pynn.run(inference_timing.row_duration.rescale(pq.ms).magnitude
                 * len(self.inference_steps))

        self.inference_results = np.empty(
            (2, len(self.inference_steps)), dtype=int)
        for step_id in range(len(self.inference_steps)):
            self.inference_results[0, step_id] = \
                synapses_left.plasticity_rule.get_observable_array(
                    "winner_neuron")[step_id].data[0]
            self.inference_results[1, step_id] = \
                synapses_right.plasticity_rule.get_observable_array(
                    "winner_neuron")[step_id].data[0]

        pynn.end()

    def update_scores(self, result: Result) -> None:
        """
        Update the scores based on the given result, and announce
        the result on screen.

        :param result: Result from playing a pong game.
        """

        if result == Result.TIE:
            if not self.demo:
                self.canvases[0].fill_text(
                    "Game ended in a tie!", GAME_WIDTH // 2, 20)
        elif result == Result.LEFT_WON:
            if not self._score_updated:
                self._score_left += 1
                self._score_updated = True
            if not self.demo:
                self.draw_scores(self.canvases[0])
                self.canvases[0].fill_text(
                    "Left won!", GAME_WIDTH // 2, 20)
        elif result == Result.RIGHT_WON:
            if not self._score_updated:
                self._score_right += 1
                self._score_updated = True
            if not self.demo:
                self.draw_scores(self.canvases[0])
                self.canvases[0].fill_text(
                    "Right won!", GAME_WIDTH // 2, 20)

    def animate(self, play_speed: int = 0) -> Result:
        """
        Start the animation of the game on a canvas.

        The scores are counted depending on which player lost, i.e.
        which player's paddle missed the ball first. The new scores
        and result are shown immediately, to facilitate running many
        games, without having to wait minutes for the animation to see
        which player has won.

        :param play_speed: Control the playback speed of the animation.
            Positive values accelerate the playback, negative values
            slow the playback down. Sensible values range roughly
            from -8 to 8, but could also be higher (dropping even more
            frames).

        :return: Result of the game.
        """

        if self.inference_results is None:
            raise ValueError("Must run the network before calling animate!")

        self.draw()

        # interpolate paddle positions between inference results,
        # to have a smoother animation
        paddle_positions_left = np.concatenate(
            [np.linspace(start, stop, num, endpoint=False) for start, stop, num
             in zip(self.inference_results[0, :-1],
                    self.inference_results[0, 1:],
                    np.diff(self.inference_steps))])
        paddle_positions_right = np.concatenate(
            [np.linspace(start, stop, num, endpoint=False) for start, stop, num
             in zip(self.inference_results[1, :-1],
                    self.inference_results[1, 1:],
                    np.diff(self.inference_steps))])

        if self.demo:
            self.canvases[0].clear()

        with ipycanvas.hold_canvas(self.canvases):
            # Some frames are dropped in case the play_speed is set to
            # 4 or higher. Below that, we show all frames but change the
            # wait between frames.
            draw_steps = np.arange(0, self.n_steps, 1 + max(play_speed - 3, 0))
            for step_id in range(self.n_steps):
                self.paddle_left.move_to(paddle_positions_left[step_id])
                self.paddle_right.move_to(paddle_positions_right[step_id])

                if step_id in self.collisions:
                    # Check if ball hit the paddle, terminate game otherwise
                    if self.ball_positions[step_id, 0] < GAME_WIDTH // 2:
                        if not self.paddle_left.collide(
                                self.ball_positions[step_id, 1]):
                            result = Result.RIGHT_WON
                            break
                    else:
                        if not self.paddle_right.collide(
                                self.ball_positions[step_id, 1]):
                            result = Result.LEFT_WON
                            break

                if step_id in draw_steps:
                    self.canvases[1].clear()
                    self.ball.draw(
                        self.canvases[1], *self.ball_positions[step_id])
                    self.paddle_left.draw(self.canvases[1])
                    self.paddle_right.draw(self.canvases[1])
                    self.canvases[1].sleep(
                        16.6 * (1.3 ** -play_speed))  # ms
            else:
                result = Result.TIE

        self.update_scores(result)
        return result

    def run(self, learned_weights: Optional[np.ndarray] = None):
        """
        Run a pong game.

        Draw a canvas, simulate the ball positions, run the inference
        network.

        To animate the results on the canvas, use self.animate()
        afterwards.

        :param learned_weights: Weight matrix for right player,
            running inference of these learned weights versus an ideal
            matrix on chip. If None, a perfect game is calculated for
            both players, without using the BSS-2 chip.
        """

        self.reset()
        self.simulate_ball()

        if self.demo:
            self.simulate_perfect_game()
        else:
            self.run_network(learned_weights)

    def draw_scores(self, canvas: ipycanvas.Canvas):
        """
        Draw scores on a canvas.

        :param canvas: Canvas to draw scores on.
        """

        canvas.clear()
        canvas.fill_text(self._score_left, GAME_WIDTH // 4, 20)
        canvas.fill_text(self._score_right, 3 * GAME_WIDTH // 4, 20)

    def draw(self):
        """
        Create and show a canvas to animate the game in, draw the elements,
        and return the canvas for animation.
        """

        # Do not create a new canvas if there exists one already, since that
        # would make the animation disappear until the previous has finished
        if self.canvases is None:
            self.canvases = ipycanvas.MultiCanvas(
                layers=2, width=GAME_WIDTH, height=GAME_HEIGHT)
            self.canvases.layout.width = "75%"
            self.canvases.layout.height = "auto"
            self.canvases[0].font = "32px sans"
            self.canvases[0].text_align = "center"
            self.canvases[0].text_baseline = "top"
        else:
            self.canvases.clear()

        # draw middle line
        self.canvases[0].fill_styled_rects(
            GAME_WIDTH // 2, np.arange(0, GAME_HEIGHT, 50), 2, 20, color=150)

        self.draw_scores(self.canvases[0])
        self.paddle_left.draw(self.canvases[1])
        self.paddle_right.draw(self.canvases[1])
        self.ball.draw(self.canvases[1])

        display(self.canvases)  # pylint: disable=undefined-variable
