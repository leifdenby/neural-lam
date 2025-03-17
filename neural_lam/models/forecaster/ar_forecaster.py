# Third-party
import torch
from loguru import logger

# Local
from ..step_predictor import BaseStepPredictor, PersistanceStepPredictor
from .base import BaseForecaster


class ARForecaster(BaseForecaster):
    """
    Subclass of Forecaster that uses an auto-regressive strategy to unroll a
    forecast. Makes use of a StepPredictor at each AR step.
    """

    _step_predictor: BaseStepPredictor

    def __init__(self, num_prediction_steps):
        logger.warning("Using persistance step predictor for now")
        self.step_predictor = PersistanceStepPredictor()
        self._num_prediction_steps = num_prediction_steps

    def forward(self, init_states, forcing_features, border_states):
        """
        Roll out prediction taking multiple autoregressive steps with the step-predictor

        Parameters
        ----------
        init_states : torch.Tensor, shape [B, 2, num_grid_nodes, d_f]
            The initial states
        forcing_features : torch.Tensor, shape [B, pred_steps, num_grid_nodes, d_static_f]
            The forcing features
        border_states : torch.Tensor, shape [B, pred_steps, num_grid_nodes, d_f]
            The border states
        """
        prev_prev_state = init_states[:, 0]
        prev_state = init_states[:, 1]
        prediction_list = []
        pred_std_list = []
        assert forcing_features.shape[1] == self._num_prediction_steps

        for i in range(self._num_prediction_steps):
            forcing = forcing_features[:, i]
            border_state = border_states[:, i]

            pred_state, pred_std = self.step_predictor(
                prev_state, prev_prev_state, forcing
            )
            # state: (B, num_grid_nodes, d_f) pred_std: (B, num_grid_nodes,
            # d_f) or None

            # Overwrite border with true state
            new_state = (
                self.boundary_mask * border_state
                + self.interior_mask * pred_state
            )

            prediction_list.append(new_state)
            if self.output_std:
                pred_std_list.append(pred_std)

            # Update conditioning states
            prev_prev_state = prev_state
            prev_state = new_state

        prediction = torch.stack(
            prediction_list, dim=1
        )  # (B, pred_steps, num_grid_nodes, d_f)
        if self.output_std:
            pred_std = torch.stack(
                pred_std_list, dim=1
            )  # (B, pred_steps, num_grid_nodes, d_f)
        else:
            pred_std = self.per_var_std  # (d_f,)

        return prediction, pred_std
