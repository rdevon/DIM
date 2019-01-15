"""Module for networks used for computing MI.

"""

import numpy as np
import torch
import torch.nn as nn

from cortex_DIM.nn_modules.misc import Permute


class MIFCNet(nn.Module):
    """Simple custom network for computing MI.

    """
    def __init__(self, n_input, n_units):
        """

        Args:
            n_input: Number of input units.
            n_units: Number of output units.
        """
        super().__init__()

        assert(n_units >= n_input)

        self.linear_shortcut = nn.Linear(n_input, n_units)
        self.block_nonlinear = nn.Sequential(
            nn.Linear(n_input, n_units),
            nn.BatchNorm1d(n_units),
            nn.ReLU(),
            nn.Linear(n_units, n_units)
        )

        # initialize the initial projection to a sort of noisy copy
        eye_mask = np.zeros((n_units, n_input), dtype=np.uint8)
        for i in range(n_input):
            eye_mask[i, i] = 1

        self.linear_shortcut.weight.data.uniform_(-0.01, 0.01)
        self.linear_shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.)

    def forward(self, x):
        """

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: network output.

        """
        h = self.block_nonlinear(x) + self.linear_shortcut(x)
        return h


class MI1x1ConvNet(nn.Module):
    """Simple custorm 1x1 convnet.

    """
    def __init__(self, n_input, n_units):
        """

        Args:
            n_input: Number of input units.
            n_units: Number of output units.
        """

        super().__init__()

        self.block_nonlinear = nn.Sequential(
            nn.Conv2d(n_input, n_units, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_units),
            nn.ReLU(),
            nn.Conv2d(n_units, n_units, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.block_ln = nn.Sequential(
            Permute(0, 2, 3, 1),
            nn.LayerNorm(n_units),
            Permute(0, 3, 1, 2)
        )

        self.linear_shortcut = nn.Conv2d(n_input, n_units, kernel_size=1,
                                         stride=1, padding=0, bias=False)

        # initialize shortcut to be like identity (if possible)
        if n_units >= n_input:
            eye_mask = np.zeros((n_units, n_input, 1, 1), dtype=np.uint8)
            for i in range(n_input):
                eye_mask[i, i, 0, 0] = 1
            self.linear_shortcut.weight.data.uniform_(-0.01, 0.01)
            self.linear_shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.)

    def forward(self, x):
        """

            Args:
                x: Input tensor.

            Returns:
                torch.Tensor: network output.

        """
        h = self.block_ln(self.block_nonlinear(x) + self.linear_shortcut(x))
        return h