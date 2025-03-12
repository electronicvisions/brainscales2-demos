from typing import List
import math

import torch
import torchvision

# pylint: disable=import-error
import hxtorch


# pylint: disable=no-member
logger = hxtorch.logger.get("partitioning_helper")


class RandomRandomRotation:

    # pylint: disable=too-many-arguments
    def __init__(self, dmin: int = -25, dmax: int = 25, prob: float = 0.5,
                 rescale: float = 1.2,
                 interpolation: torchvision.transforms.InterpolationMode
                 = torchvision.transforms.InterpolationMode.BILINEAR):
        self.degrees = (dmin, dmax)
        self.prob = prob
        self.rescale = rescale
        self.interpolation = interpolation
        self.random_rotation = torchvision.transforms.RandomRotation(
            degrees=(dmin, dmax), interpolation=interpolation)

    def __call__(self, tensor: torch.tensor):
        if torch.rand(1).item() < self.prob:
            rot_img = self.random_rotation(tensor)
            return torch.clamp(
                rot_img / torch.max(rot_img) * torch.tensor(self.rescale), 0,
                1)
        return tensor

    def __repr__(self):
        return self.__class__.__name__ \
            + f"(degrees={self.degrees}, prob={self.prob})"


def partition(layer_sizes: List[int], seperate_output: bool = True):
    """
    Calculates necessary partitions by means of hardware limitations on BSS-2
    for feed-forward fully connected architecture with signed weights

    :param layer_sizes: List holding number of units per layer (is interpreted
        as in direction of information flow)
    :return: Tuple of two lists, list 1 contains lists of number of partitions
        for the following layer (when hw-runs of consecutive layers can be
        combined they are grouped in one list) and list 2 the number of atomic
        neurons per single-compartment for each layer (except input layer).
    """

    n_layers = len(layer_sizes)
    part_arr = []
    comp_arr = []

    temp_size_arr = []
    temp_n_comp_arr = []
    temp_comb_part_arr = []
    temp_empty = True

    for i in range(1, n_layers):

        if layer_sizes[i] > 32 * 128 or layer_sizes[i] == 0:
            logger.WARN(
                f"Layer sizes below 1 and above 32*128 = {32 * 128} are not "
                + "possible.")
            return part_arr, comp_arr

        c_i = (int)(math.ceil(layer_sizes[i - 1] / 128))
        comp_arr += [c_i]
        n_ci = math.floor(512 / c_i)
        p_i = math.ceil(layer_sizes[i] / n_ci)

        if temp_empty:
            if p_i == 1:
                temp_n_comp_arr += [c_i]
                temp_size_arr += [layer_sizes[i]]
                temp_comb_part_arr += [1]
                temp_empty = False
            else:
                part_arr += [[p_i]]
        else:
            if p_i == 1:
                if not (seperate_output and i == n_layers - 1) and \
                    is_combinable(
                        temp_size_arr + [layer_sizes[i]],
                        temp_n_comp_arr + [c_i]):

                    temp_n_comp_arr += [c_i]
                    temp_size_arr += [layer_sizes[i]]
                    temp_comb_part_arr += [1]
                else:
                    part_arr += [temp_comb_part_arr]
                    temp_size_arr = [layer_sizes[i]]
                    temp_n_comp_arr = [c_i]
                    temp_comb_part_arr = [1]
            else:
                part_arr += [temp_comb_part_arr]
                part_arr += [p_i]

                temp_size_arr = []
                temp_n_comp_arr = []
                temp_comb_part_arr = []
                temp_empty = True

    if not temp_empty:
        part_arr += [temp_comb_part_arr]

    return part_arr, comp_arr


def is_combinable(layer_sizes: List[int], compartment_sizes: List[int]):
    """
    Decides if layer executions can be combined into one (bss-2)

    :param layer_sizes: List holding number of units per layer (is interpreted
        as in direction of information flow)
    :param compartment_sizes: List holding number of atomic neurons per
        compartment for each layer
    :returns: True if layers can be combined into one hardware execution,
        otherwise: False.
    """

    n_atm_neurons = 0
    for (i, n_l) in enumerate(layer_sizes):
        n_atm_neurons += n_l * compartment_sizes[i]
    if n_atm_neurons <= 512:
        return True

    return False
