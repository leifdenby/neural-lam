# Standard library
import abc

# Third-party
from loguru import logger
from torch import nn


class BaseForecaster(nn.Module, abc.ABC):
    """
    A generic forecaster capable of mapping from a set of initial states,
    forcing and boundary forcing into a full forecast of the requested
    length
    """

    pass

    @abc.abstractmethod
    def forward(self, init_states, forcing_features, border_states):
        """
        Using the initial states, forcing features and true states, produce a
        forecast for the requested length

        Parameters
        ----------
        init_states : torch.Tensor, shape [B, 2, num_grid_nodes, d_f]
            The initial states
        forcing_features : torch.Tensor, shape [B, pred_steps, num_grid_nodes, d_static_f]
            The forcing features
        border_states : torch.Tensor, shape [B, pred_steps, num_grid_nodes, d_f]
            The border states
        """
        raise NotImplementedError("Forward method not implemented")
