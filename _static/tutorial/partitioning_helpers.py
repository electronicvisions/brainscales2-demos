import torch
import torchvision


class RandomRandomRotation:
    """
    Randomly applies a random rotation to an image tensor with a given
    probability
    """

    def __init__(self, dmin: int = -25, dmax: int = 25, prob: float = 0.5,
                 rescale: float = 1.2):
        """
        Randomly applies a random rotation to an image tensor with a given
        probability.

        :param dmin: Minimum rotation angle in degrees.
        :param dmax: Maximum rotation angle in degrees.
        :param prob: Probability of applying the rotation.
        :param rescale: Value to rescale the rotated image.
        """
        self.degrees = (dmin, dmax)
        self.prob = prob
        self.rescale = rescale
        self.random_rotation = torchvision.transforms.RandomRotation(
            degrees=(dmin, dmax),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR)

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
