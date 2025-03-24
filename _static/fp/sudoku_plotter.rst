.. code:: ipython3

    import pynn_brainscales.brainscales2 as pynn
    from typing import Callable
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt


    class SudokuPlotter():
    
        def __init__(
            self,
            dimension: int,
            pop: pynn.populations.Population,
            set_clues: Callable[[np.ndarray], None],
            hide_solution: Callable[[np.ndarray, int, int], np.ndarray],
            get_solution: Callable[[float, np.ndarray], np.ndarray]
        ):
            # Initialize and contain sudoku solver and plotter
            """
            Initialize a Sudoku Plotter
    
            :param dimension : Size of Sudoku to solve
            :param pop : pynn.Population of Neurons which run emulation
            :param set_clues(clues:np.ndarray) : function which sets clues
                on population
            :param hide_solution(sudoku, num_clues, seed=None) : Chooses random
                hints from sudoku returns clues
            :param get_solution(runtime: float, clues:np.ndarray) : Runs pynn and
                return solved sudoku
            """
            self.dimension = dimension
            self.pop = pop
    
            # Sudoku Params
            self.clues = None
            self.solved_sudoku = np.zeros((dimension, dimension))
            self.solved_sudoku_correct = None
    
            # Required functions to solve sudoku
            self.set_clues = set_clues
            self.hide_solution = hide_solution
            self.get_solution = get_solution
    
            # Config grid itself
            self.grid_color = "gray"
            self.grid_width = 2
            self.grid_major = 4
    
            # Config numbers on grid
            self.sudoku_fontsize = 15
    
            # Config Activities
            self.space_between_numbers = 2
            self.space_between_cells = 4
    
            self.runtime = None
    
        def empty_sudoku(self, ax: plt.axes) -> None:
            """
            Draws onto an axis an empty sudoku grid with dimensions of
                self.dimension
    
            param: ax: plt.axis : Axis to draw on sudoku
            """
    
            # Draw outer box
            outer_box = [[.5, self.dimension + .5, self.dimension + .5, .5, .5],
                         [.5, .5, self.dimension + .5, self.dimension + .5, .5]]
    
            # Coloring outer edge
            color_outer = self.grid_color
    
            if self.solved_sudoku_correct:
                color_outer = "green"
    
            if not self.solved_sudoku_correct:
                color_outer = "red"
    
            ax.plot(outer_box[0], outer_box[1], linewidth=8, zorder=20,
                    color=color_outer)
            ax.set_xlim(min(outer_box[0]) * 0.85, max(outer_box[0]) * 1.0125)
            ax.set_ylim(min(outer_box[1]) * 0.85, max(outer_box[1]) * 1.0125)
    
            # Arange fine lines for squares
            grid_lines = np.arange(.5, self.dimension + .5, 1)
    
            # Draw squares on grid
            ax.vlines(grid_lines, min(outer_box[0]), max(outer_box[0]),
                      linewidth=self.grid_width, color=self.grid_color)
    
            ax.hlines(grid_lines, min(outer_box[0]), max(outer_box[0]),
                      linewidth=self.grid_width, color=self.grid_color)
    
            # If a propper sudoku draw thick lines onto grid
            if self.dimension in [2, 4, 9, 16, 25, 36, 49]:
                step = int(np.sqrt(self.dimension))
                ax.vlines(grid_lines[::step], min(outer_box[0]), max(outer_box[0]),
                          linewidth=self.grid_major, color=self.grid_color)
                ax.hlines(grid_lines[::step], min(outer_box[0]), max(outer_box[0]),
                          linewidth=self.grid_major, color=self.grid_color)
    
            # Align ticks as neurons are labeled
            ax.set_xticks(np.arange(1, self.dimension + 1, 1))
            ax.set_yticks(np.arange(1, self.dimension + 1, 1),
                          np.arange(self.dimension, 0, -1))
    
            ax.grid(False)
    
        def sudoku_populate(
            self, ax: plt.axes, sudoku: np.ndarray
        ) -> None:
            """
            Insert Numbers from a given sudoku. If field equal 0 no number
            is written
    
            :param ax: plt.axes : Insert numbers onto this axes.
                                Run before self.empty_sudoku() to create grid
            :param sudoku: numpy.ndarray : Sudoku with 0 for empty cells
            """
    
            for y_coord, values in enumerate(sudoku):
                y_coord = len(values) - y_coord
                for x_coord, value in enumerate(values):
                    x_coord = x_coord + 1
                    if value != 0:
                        ax.text(x_coord, y_coord, value,
                                ha="center", va="center",
                                fontsize=self.sudoku_fontsize)
    
        def sudoku_populate_clues(self, ax: plt.axes,
                                  sudoku: np.ndarray) -> None:
    
            for y_coord, values in enumerate(sudoku):
                y_coord = len(values) - y_coord
                for x_coord, value in enumerate(values):
                    x_coord = x_coord + 1
                    if value != 0:
    
                        ax.add_patch(Rectangle((x_coord - .5, y_coord - .5), 1, 1,
                                               fc=(.85, .85, .85), zorder=-11))
    
        def check_solution(self, sudoku) -> bool:
            """
            Checks if a sudoku is correct by checking sum in each row.
            Expected value is
            x_exp = 1 + ... + dimension
    
            :param sudoku:np.ndarray
    
            return bool
            """
            expected_result = np.sum(np.arange(1, self.dimension + 1))
    
            for row in sudoku:
                if expected_result != np.sum(row):
                    return False
            return True
    
        def solve_sudoku(self, sudoku, runtime, num_clues, seed=None) -> None:
            """
            Executes required steps to solve sudoku
            1. Defines clues -> SudokuPlotter.clues
            2. Applies clues with set_clues
            3. Runs emulation (get_solution) and stores in
                SudokuPlotter.solved_sudoku
            4. Checks if solution correct with SudokuPlotter.ckeck_solution
            """
    
            # Generate clues
            self.clues = self.hide_solution(sudoku, num_clues, seed=seed)
            self.set_clues(self.clues)
    
            # Solve sudoku
            self.solved_sudoku = self.get_solution(runtime, self.clues)
    
            self.solved_sudoku_correct = self.check_solution(self.solved_sudoku)
            self.runtime = runtime
    
        def plot_sudoku(self, ax=None, figsize=(4, 4)) -> None:
            """
            Plots sudoku
            If no axis is given, a new figure is created
            """
    
            if ax is None:
                _, ax = plt.subplots(figsize=figsize)
    
            self.empty_sudoku(ax)
            self.sudoku_populate(ax, self.solved_sudoku)
            self.sudoku_populate_clues(ax, self.clues)
    
        def plot_activities(self, ax=None, figsize=(15, 10)) -> None:
            """
            Plots the activity of each individual neuron
            If no axis is given, a new figure is created
            """
            if ax is None:
                _, ax = plt.subplots(figsize=figsize)
    
            spiketrains = self.pop.get_data().segments[-1].spiketrains
    
            colors = plt.get_cmap("tab10").colors[:self.dimension]
    
            # Running vaiable for spacing
            current_y = 0
    
            for index, spikes in enumerate(spiketrains):
                # Action between different cells
                if index % self.dimension == 0 and index > 0:
                    # Draw a horizontal line to split cells
                    ax.axhline(current_y + self.space_between_cells / 2,
                               color="k", alpha=.5)
                    current_y += self.space_between_cells
    
                # Only add labels in first cell
                label = index % self.dimension + 1 if index < self.dimension \
                    else None
    
                # Plot the acitvity
                ax.scatter(spikes, [current_y] * len(spikes), label=label,
                           color=colors[index % self.dimension], s=10)
    
                current_y += self.space_between_numbers
    
            print(f"xlim Values: {ax.get_xlim()}")
            ax.set_xlim(ax.get_xlim()[0], self.runtime * 1.08)
            ax.legend()
    
            # Set y labels at center of cells
            first_label = self.dimension * self.space_between_numbers / 2
            space_between_labels = self.dimension * self.space_between_numbers + \
                self.space_between_cells
    
            ticks = np.arange(self.dimension**2) * space_between_labels + \
                first_label
            numbers = np.arange(self.dimension) + 1
            labels = [f'[{row},{column}]'
                      for row, column in product(numbers, numbers)]
    
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)
    
            ax.set_xlabel("Time [ms]")
            ax.set_ylabel("Coordinat [row, column]")
    
        def plot(self, grid=True, figsize=(15, 5)) -> None:
            """
            Plots the results
            if grid is True, makes one plot with grid and anctivities, else two
                separate images
            :param grid=True: plots a grid
            :param figsize=(15,5): standart size of figure, suggested to have a
                ratio 3:1
    
            """
            # self.solve_sudoku(sudoku, runtime, num_clues, seed=seed)
    
            if grid:
                fig = plt.figure(figsize=figsize)
    
                grid = GridSpec(3, 8, figure=fig, wspace=1.5)
    
                ax1 = fig.add_subplot(grid[:, :3])
                self.plot_sudoku(ax=ax1)
    
                ax2 = fig.add_subplot(grid[:, 3:])
                self.plot_activities(ax=ax2)
    
            else:
                self.plot_sudoku()
                self.plot_activities()

