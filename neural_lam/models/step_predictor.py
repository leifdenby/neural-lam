# Standard library
import abc

# Third-party
import torch
from torch import nn


class StepPredictor(nn.Module, abc.ABC):
    """
    A model mapping from the two previous time steps + forcing + boundary
    forcing to a prediction of the next state. Corresponds to the function in
    Oskarsson et al.
    """

    def forward(
        self,
        prev_state: torch.Tensor,
        prev_prev_state: torch.Tensor,
        forcing: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        prev_state: (B, num_grid_nodes, feature_dim), X_t prev_prev_state: (B,
        num_grid_nodes, feature_dim), X_{t-1} forcing: (B, num_grid_nodes,
        forcing_dim)

        Parameters
        ----------
        prev_state : torch.Tensor, shape [B, num_grid_nodes, feature_dim]
            The state at the previous time step
        prev_prev_state : torch.Tensor, shape [B, num_grid_nodes, feature_dim]
            The state at the time step before the previous time step
        forcing : torch.Tensor, shape [B, num_grid_nodes, forcing_dim]
            The forcing for the next time step
        """
        raise NotImplementedError("No prediction step implemented")


class PersistanceStepPredictor(StepPredictor):
    """
    A simple forecaster that just repeats the last state as the forecast
    """

    def forward(
        self,
        prev_state: torch.Tensor,
        prev_prev_state: torch.Tensor,
        forcing: torch.Tensor,
    ) -> torch.Tensor:
        """
        Step state one step ahead using prediction model, X_{t-1}, X_t -> X_t+1
        prev_state: (B, num_grid_nodes, feature_dim), X_t prev_prev_state: (B,
        num_grid_nodes, feature_dim), X_{t-1} forcing: (B, num_grid_nodes,
        forcing_dim)

        Parameters
        ----------
        prev_state : torch.Tensor, shape [B, num_grid_nodes, feature_dim]
            The state at the previous time step
        prev_prev_state : torch.Tensor, shape [B, num_grid_nodes, feature_dim]
            The state at the time step before the previous time step
        forcing : torch.Tensor, shape [B, num_grid_nodes, forcing_dim]
            The forcing
        """
        return prev_state
