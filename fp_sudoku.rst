Sudoku
======

Another task a spiking neural network can learn is solving a sudoku.
Since the number of available neurons on chip is limited, we’ll consider
just a 4x4 sudoku.

.. image:: _static/common/sudoku.png
    :align: center
    :width: 500px

(Source: Alexander Kugele, “Solving the Constraint Satisfaction Problem
Sudoku on Neuromorphic Hardware”,
https://www.kip.uni-heidelberg.de/Veroeffentlichungen/download.php/6118/temp/3666.pdf)

Each field contains four neurons that represent the possible solutions.
When a number is fixed, it prohibits certain combinations according to
the Sudoku rules. For example, consider the gray 3. Since it is fixed,
the numbers 1, 2 or 4 cannot be present in the same field (orange). In
addition, no more 3 may appear in the same row (purple) or in the same
column (green). The same applies to the lower left block (blue). A
fixed number is represented by a continuous firing of the associated
neuron. The forbidden combinations shall be realized by the means of
inhibitory synapses. Consequently, when a number is fixed, all resulting
forbidden possibilities are strongly suppressed by the inhibitory
connections, such that the respective neurons do not fire. In order to
make the remaining allowed neurons fire, all neurons receive weak input
from random background noise. In addition, all neurons are excitatorily
connected to themselves to maintain possible activity.

.. code:: ipython3

    %%capture
    !pip install ipycanvas

.. code:: ipython3

    from math import sqrt
    import ipywidgets as w
    import matplotlib.pyplot as plt
    %matplotlib inline
    from itertools import product
    from functools import partial
    IntSlider = partial(w.IntSlider, continuous_update=False)
    from ipycanvas import Canvas, hold_canvas

    import numpy as np
    import pynn_brainscales.brainscales2 as pynn
    from pynn_brainscales.brainscales2.standardmodels.synapses import StaticSynapse
    from pynn_brainscales.brainscales2.standardmodels.cells import \
                SpikeSourceArray, HXNeuron,SpikeSourcePoisson


    from _static.common.helpers import setup_hardware_client, get_nightly_calibration

    runtime = 0.5
    dimension = 4

    # set the environment
    setup_hardware_client()
    calib = get_nightly_calibration()
    pynn.setup(initial_config=calib)

    # create a neuron for every number in each field
    # -> we need 4 (rows) * 4 (columns) * 4 (numbers) = 4^3 neurons
    print("Creating neurons... (1/4)")
    pop = pynn.Population(4**3, HXNeuron())
    pop.record(["spikes"])

    # to define the connections easier, we save a "view" of each neuron in a list
    pops_collector = []
    for row in range(dimension):
        pops_row = []
        for field_in_row in range(dimension):
            pops_field = []
            for number_in_field in range(dimension):
                neuron = pynn.PopulationView(
                    pop,
                    [row * dimension**2 + field_in_row * dimension
                     + number_in_field])
                pops_field.append(neuron)
            pops_row.append(pops_field)
        pops_collector.append(pops_row)

    # each neuron gets individual input from a Poisson distribution
    print("Generating background noise... (2/4)")
    poisson_source = pynn.Population(dimension**3,
        SpikeSourcePoisson(duration=runtime - 0.01, rate=5e5, start=0.01))

    # connect random sources with neurons
    # additionally each neuron is connected to itself excitatorily to
    # sustain possible activity
    print("Implementing connections from background and self-connections... (3/4)")
    pynn.Projection(pop,
                    pop,
                    pynn.OneToOneConnector(),
                    synapse_type=StaticSynapse(weight=20),
                    receptor_type='excitatory')
    pynn.Projection(poisson_source,
                    pop,
                    pynn.OneToOneConnector(),
                    synapse_type=StaticSynapse(weight=30),
                    receptor_type='excitatory')

    # create stimulation for clues and connect to according neurons
    print("Preparing clues... (4/4)")
    stim_given_numbers = pynn.Population(
        2, SpikeSourceArray(spike_times=np.linspace(0.0, runtime, 500)))
    clue_projections = []
    for row in range(4):
        clues_row = []
        for column in range(4):
            clues_field = []
            for number in range(4):
                clues_field.append(pynn.Projection(
                    stim_given_numbers,
                    pops_collector[row][column][number],
                    pynn.AllToAllConnector(),
                    synapse_type=StaticSynapse(weight=0),
                    receptor_type='excitatory'))
            clues_row.append(clues_field)
        clue_projections.append(clues_row)

We define some functions to solve and display the sudoku.

.. code:: ipython3

    # functions to solve the sudoku:

    def set_clues(clues=None):
        """Sets the clues in the network."""
        if clues is None:
            clues = np.zeros((4, 4), dtype=int)
        for row, row_clues in enumerate(clue_projections):
            for col, field_clues in enumerate(row_clues):
                for number, clue_projection in enumerate(field_clues, start=1):
                    for connection in clue_projection:
                        connection.weight = 63. if clues[row,col] == number else 0.

    def hide_solution(grid, num_clues, seed=None):
        """Hides the solution and only leaves `num_clues` hints."""
        indices = np.argwhere(np.logical_and(grid > 0, grid <= 4))
        if len(indices) < num_clues:
            raise RuntimeError(
                f"The sudoku has less than {num_clues} clues, which is the number of required clues :(")
        np.random.seed(seed)
        indices = indices[np.random.choice(len(indices), num_clues, replace=False)]
        clues = np.zeros_like(grid)
        clues[(indices.T[0], indices.T[1])] = grid[(indices.T[0], indices.T[1])]
        return clues

    def get_solution(clues):
        """Executes the network ad returns the current solution."""
        set_clues(clues)
        grid = np.zeros((4, 4), dtype=int)
        # emulate the network
        pynn.run(runtime)
        # read back solution
        for row, row_populations in enumerate(pops_collector):
            for col, field_populations in enumerate(row_populations):
                num_spikes = [
                    len(num_population.get_data("spikes").segments[-1].spiketrains[0])
                    for num_population in field_populations
                ]
                grid[row, col] = np.argmax(num_spikes) + 1
        pynn.reset()
        return grid

    # functions to display the sudoku:

    def canvas_empty(N=4, size=50, canvas=None):
        """Creates an emtpy canvas for the sudoku."""
        if canvas is None:
            canvas = Canvas(
                width=size*N, height=size*N,
                layout=w.Layout(margin='5px'))
            canvas.scale(size)
        canvas.clear()
        canvas.layout.border=f'solid {size/15}px black'
        canvas.font = '0.7px sans-serif'
        canvas.text_align = 'center';
        canvas.text_baseline = 'middle'
        return canvas

    def canvas_sudoku_empty(N=4, size=50, canvas=None):
        """Creates an empty sudoku. Only the numbers are missing."""
        Ns = int(sqrt(N))
        canvas = canvas_empty(N, size, canvas=canvas)
        with hold_canvas(canvas):
            for i in range(0, N+1):
                canvas.line_width = 1/15 if i % Ns == 0 else 1/30
                canvas.stroke_line(0, i, N, i)
                canvas.stroke_line(i, 0, i, N)
        return canvas

    def display_clues(canvas, grid):
        """Displays the clues."""
        with hold_canvas(canvas):
            for row, row_fields in enumerate(grid):
                for col, field in enumerate(row_fields):
                    if field > 0:
                        canvas.fill_style = '#00000022'
                        canvas.fill_rect(col, row, 1, 1)

    def check_solution(grid, N=4):
        """Checks if the sudoku rules are fulfilled."""
        Ns = int(sqrt(N))
        for i in range(N):
            # j, k index top left hand corner of each 3x3 tile
            j, k = (i // Ns) * Ns, (i % Ns) * Ns
            if len(set(grid[i,:])) != N or len(set(grid[:,i])) != N\
                       or len(set(grid[j:j+Ns, k:k+Ns].ravel())) != N:
                return False
        return True

    plot_output = w.Output()
    def display_activity(grid, clues):
        """Displays the activity of each neuron over time."""
        # Check if grid is a valid solution which satisfies all clues
        valid_solution = check_solution(grid)
        valid_clues = np.all((grid == clues)[clues != 0])
        if valid_solution and valid_clues:
            num_correct = dimension**2
        else:
            num_correct = (grid == sudoku).sum()
        #if num_correct < 16:
        numbers = np.arange(dimension) + 1
        labels = [f'{row}{column}' for row, column in
                  product(numbers, numbers)]
        plot_output.clear_output()
        with plot_output:
            colors = plt.get_cmap('tab10').colors[:dimension]
            fig, ax = plt.subplots(figsize=(15, 10))
            space_between_numbers = 2
            space_between_cells = 4
            current_y = 0
            for index, spikes in enumerate(pop.get_data().segments[0].spiketrains):
                # Actions between different cells
                if index % dimension == 0 and index > 0:
                    ax.axhline(current_y + space_between_cells / 2,
                               color='k',
                               alpha=0.5)
                    current_y += space_between_cells
                # Only add labels in first cell
                label = index % dimension + 1 if index < dimension else None
                ax.scatter(spikes, [current_y] * len(spikes),
                           label=label,
                           color=colors[index % dimension],
                           s=10)
                current_y += space_between_numbers
            ax.legend()
            # Set y labels at center of cells
            first_label = dimension * space_between_numbers / 2
            space_between_labels = dimension * space_between_numbers + \
                    space_between_cells
            ticks = np.arange(dimension**2) * space_between_labels + first_label
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)
            ax.set_xlabel('Time [ms]')
            ax.set_ylabel('Sudoku Coordinates [row, column]')
            plt.show()
        #num_clues = np.count_nonzero(clues)

    def display_solution(canvas, grid):
        """Displays the current solution."""
        with hold_canvas(canvas):
            for row, row_fields in enumerate(grid):
                for col, field in enumerate(row_fields):
                    if field > 0:
                        canvas.fill_style = '#000000dd'
                        canvas.fill_text(int(field), col+.5, row+.5)
            canvas.layout.border = canvas.layout.border.rsplit(' ', 1)[0] \
                + (' green' if check_solution(grid) else ' darkred')

    def display_sudoku_solver(sudoku):
        """Displays the sudoku and sliders."""
        canvas = canvas_sudoku_empty()
        num_clues_slider = IntSlider(
            8, 0, len(np.argwhere(sudoku)), description="Number of clues")
        seed_slider = IntSlider(
            1234, 0, 3000, description="Random seed")
        run_button = w.Button(description="again",icon="play")

        def solve_sudoku(num_clues, seed):
            """Tries to solve the sudoku and displays the result."""
            with hold_canvas(canvas):
                canvas_sudoku_empty(canvas=canvas)
            clues = hide_solution(sudoku, num_clues, seed)
            display_clues(canvas, clues)
            display_solution(canvas, get_solution(clues))
            display_activity(get_solution(clues), clues)

        interactive = w.interactive(
            solve_sudoku, num_clues=num_clues_slider, seed=seed_slider)
        run_button.on_click(interactive.update)
        display(w.VBox([w.HBox([canvas, w.VBox([num_clues_slider, seed_slider, run_button])]), plot_output]))
        interactive.update()

**Here, you need to implement the prohibition rules.**

.. code:: ipython3

    print("Implementing rules...")

    # create inhibitory connections to neurons in the same field
    # representing different numbers
    print("  - there may only be one number per field")


    # create inhibitory connections to neurons in the same row
    # representing the same number
    print("  - each number may only occur once per row")


    # create inhibitory connections to neurons in the same column
    # representing the same number
    print("  - each number may only occur once per column")


    # create inhibitory connections to neurons in the same block
    # representing the same number
    # - which connections actually need to be still realized?
    print("  - each number may only occur once per block")

.. code:: ipython3

    # this sudoku shall be solved
    sudoku = np.array([
        [3, 2, 4, 1],
        [1, 4, 3, 2],
        [2, 3, 1, 4],
        [4, 1, 2, 3]
    ])

    display_sudoku_solver(sudoku)

**Question a:** What happens, if the network just runs like this? Is
this what you would expect?

**Task 1:** Now implement the prohibition rules. It might be helpful to
investigate the code used for setting up the network above.

You can try different sudokus by changing the random seed or specifying
them yourself. To do so change the sudoku in the code above and set the
unknown fields to zero. For now, keep the number of clues at eight.

Once you were able to let the neuromorphic chip solve the sudoku for you,
let’s investigate this sudoku solver more closely:

**Task 2:** What do you expect to happen, if you set the number of clues
to zero? Check your hypothesis. Can you explain your observation?

**Task 3:** Now, investigate how the success rate is related to the
number of clues. For this, vary the number of clues from four to ten.
Repeat each configuration ten times, while keeping the sudoku fixed.
Visualize your result.
