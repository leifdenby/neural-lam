# Standard library
import dataclasses
from functools import cached_property
from typing import List, Union

# Third-party
import parse
import pytorch_lightning as pl
import torch
import xarray as xr

# Local
from . import metrics, vis
from .datastore import BaseDatastore
from .loss_weighting import get_state_feature_weighting
from .models.forecaster.base import ARForecastor
from .weather_dataset import create_dataarray_from_tensor


@dataclasses.dataclass
class ForecasterConfig:
    input_dim: int
    output_dim: int
    hidden_dim: int
    num_layers: int
    num_past_forcing_steps: int
    num_future_forcing_steps: int
    loss_fn: str


@dataclasses.dataclass
class MetricWatchShorthand:
    split: str
    metric: str
    var_name: str
    ar_step: int
    FORMAT = "{split}_{metric}_{var_name}_step_{ar_step}"

    def __init__(self, s: str):
        parsed = parse.parse(self.FORMAT, s)
        self.split = parsed["split"]
        self.metric = parsed["metric"]
        self.var_name = parsed["var_name"]
        self.ar_step = parsed["ar_step"]


def expand_to_batch(x, batch_size):
    """
    Expand tensor with initial batch dimension
    """
    return x.unsqueeze(0).expand(batch_size, -1, -1)


@dataclasses.dataclass
class MetricTracking:
    """
    Track values of metrics over roll-out windows during training
    """

    watched_metrics: List[MetricWatchShorthand]

    def create_metric_rollout_plot(
        self, metric_tensor, metric_plot_id, save_to_disk
    ):
        """
        Create a plot of the metric values over the roll-out window

        metric_tensor: (pred_steps, d_f), metric values per time and variable
        prefix: string, prefix to use for logging
        metric_name: string, name of the metric

        Parameters
        ----------
        metric_tensor : torch.Tensor, shape [pred_steps, d_f]
            The metric values per time and variable
        metric_plot_id : str
            The identifier to use for logging
        save_to_disk : bool
            Whether to save the plot and data to disk, the filenames will be
            "{full_log_name}.pdf" and "{full_log_name}.csv"

        Returns
        -------
        plt.Figure
            The figure with the plot

        """
        metric_fig = vis.plot_error_map(
            errors=metric_tensor,
            datastore=self._datastore,
        )

        if save_to_disk:
            # Save pdf
            metric_fig.savefig(
                os.path.join(self.logger.save_dir, f"{metric_plot_id}.pdf")
            )
            # Save errors also as csv
            np.savetxt(
                os.path.join(self.logger.save_dir, f"{metric_plot_id}.csv"),
                metric_tensor.cpu().numpy(),
                delimiter=",",
            )

        return metric_fig

    def aggregate_and_plot_metrics(self, metrics_dict, prefix):
        """
        Aggregate and create error map plots for all metrics in metrics_dict

        metrics_dict: dictionary with metric_names and list of tensors
            with step-evals.
        prefix: string, prefix to use for logging
        """
        # XXX: use something more explicit than "prefix"
        save_figure_and_data_to_disk = prefix == "test"

        log_dict = {}
        for metric_name, metric_val_list in metrics_dict.items():
            metric_tensor = self.all_gather_cat(
                torch.cat(metric_val_list, dim=0)
            )  # (N_eval, pred_steps, d_f)

            if self.trainer.is_global_zero:
                metric_tensor_averaged = torch.mean(metric_tensor, dim=0)
                # (pred_steps, d_f)

                # Take square root after all averaging to change MSE to RMSE
                if "mse" in metric_name:
                    metric_tensor_averaged = torch.sqrt(metric_tensor_averaged)
                    metric_name = metric_name.replace("mse", "rmse")

                # NOTE: we here assume rescaling for all metrics is linear
                metric_rescaled = metric_tensor_averaged * self.state_std
                # (pred_steps, d_f)

                metric_plot_id = f"{prefix}_{metric_name}"
                log_dict[metric_plot_id] = self.create_metric_rollout_plot(
                    metric_tensor=metric_rescaled,
                    metric_plot_id=metric_plot_id,
                    save_to_disk=save_figure_and_data_to_disk,
                )

                for key, value in self.create_metric_log_entries(
                    metric_tensor=metric_rescaled,
                    split=prefix,
                    metric_name=metric_name,
                ):
                    log_dict[key] = value

        # Ensure that log_dict has structure for
        # logging as dict(str, plt.Figure)
        assert all(
            isinstance(key, str) and isinstance(value, plt.Figure)
            for key, value in log_dict.items()
        )

        if self.trainer.is_global_zero and not self.trainer.sanity_checking:

            current_epoch = self.trainer.current_epoch

            for key, figure in log_dict.items():
                # For other loggers than wandb, add epoch to key.
                # Wandb can log multiple images to the same key, while other
                # loggers, such as MLFlow need unique keys for each image.
                if not isinstance(self.logger, pl.loggers.WandbLogger):
                    key = f"{key}-{current_epoch}"

                if hasattr(self.logger, "log_image"):
                    self.logger.log_image(key=key, images=[figure])

            plt.close("all")  # Close all figs


class ForecasterModule(pl.LightningModule):
    """
    Takes over much of the responsibility of the old ARModel. Handles things
    not directly related to the nerual network components such as plotting,
    logging, moving batches to the right device. This inherits
    pytorch_lightning.LightningModule and have the different train/val/test
    steps. In each step (train/val/test), unpacks the batch of tensors and uses
    a Forecaster to produce a full forecast. Also responsible for computing the
    loss based in a produced forecast (could also be in Forecaster, not
    entirely sure about this).
    """

    def __init__(
        self,
        args,
        config: ForecasterConfig,
        datastore: BaseDatastore,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["datastore"])
        self.args = args
        self._datastore = datastore
        self._config = config
        self._include_std = False

        # Instantiate loss function
        self.loss = metrics.get_metric(config.loss_fn)

        self._register_buffers()

        # For making restoring of optimizer state optional
        self.restore_opt = args.restore_opt

        self.forecaster = ARForecastor(
            config=config,
            datastore=datastore,
            include_std=self._include_std,
        )

    def _register_buffers(self):
        # Load static features standardized
        da_static_features = self._datastore.get_dataarray(
            category="static", split=None, standardize=True
        )
        da_state_stats = self._datastore.get_standardization_dataarray(
            category="state"
        )
        da_boundary_mask = self._datastore.boundary_mask

        # Load static features for grid/data,
        self.register_buffer(
            "grid_static_features",
            torch.tensor(da_static_features.values, dtype=torch.float32),
            persistent=False,
        )

        state_stats = {
            "state_mean": da_state_stats.state_mean.values,
            "state_std": da_state_stats.state_std.values,
            # Note that the one-step-diff stats (diff_mean and diff_std) are
            # for differences computed on standardized data
            "diff_mean": da_state_stats.state_diff_mean_standardized.values,
            "diff_std": da_state_stats.state_diff_std_standardized.values,
        }

        for key, val in state_stats.items():
            val_tensor = torch.tensor(val, dtype=torch.float32)
            self.register_buffer(key, val_tensor, persistent=False)

        state_feature_weights = get_state_feature_weighting(
            config=self._config, datastore=self._datastore
        )
        self.feature_weights = torch.tensor(
            state_feature_weights, dtype=torch.float32
        )

        # Double grid output dim. to also output std.-dev.
        if not self._include_std:
            # Store constant per-variable std.-dev. weighting
            # NOTE that this is the inverse of the multiplicative weighting
            # in wMSE/wMAE
            self.register_buffer(
                "per_var_std",
                self.diff_std / torch.sqrt(self.feature_weights),
                persistent=False,
            )

        boundary_mask = torch.tensor(
            da_boundary_mask.values, dtype=torch.float32
        ).unsqueeze(
            1
        )  # add feature dim

        self.register_buffer("boundary_mask", boundary_mask, persistent=False)
        # Pre-compute interior mask for use in loss function
        self.register_buffer(
            "interior_mask", 1.0 - self.boundary_mask, persistent=False
        )  # (num_grid_nodes, 1), 1 for non-border

    @cached_property
    def num_grid_nodes(self) -> int:
        return self._datastore.num_grid_points

    @cached_property
    def grid_static_dim(self) -> int:
        return self.grid_static_features.shape[1]

    @cached_property
    def grid_dim(self):
        return (
            2 * self.grid_output_dim
            + self.grid_static_dim
            + self.num_forcing_vars
            * (
                self._config.num_past_forcing_steps
                + self._config.num_future_forcing_steps
                + 1
            )
        )

    @cached_property
    def grid_output_dim(self):
        if self._include_std:
            # Pred. dim. in grid cell
            self.grid_output_dim = 2 * self.num_state_vars
        else:
            # Pred. dim. in grid cell
            self.grid_output_dim = self.num_state_vars

    @cached_property
    def num_state_vars(self) -> int:
        return self._datastore.get_num_data_vars(category="state")

    @cached_property
    def num_forcing_vars(self) -> int:
        return self._datastore.get_num_data_vars(category="forcing")

    def forward(self, batch):
        """
        Predict on single batch batch consists of: init_states: (B, 2,
        num_grid_nodes, d_features) target_states: (B, pred_steps,
        num_grid_nodes, d_features) forcing_features: (B, pred_steps,
        num_grid_nodes, d_forcing),
            where index 0 corresponds to index 1 of init_states
        """
        # init, target, forcing, times
        (init_states, _, forcing_features, batch_times) = batch

        return self.forecaster(
            init_states=init_states,
            forcing_features=forcing_features,
            batch_times=batch_times,
        )

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.args.lr, betas=(0.9, 0.95)
        )
        return opt

    @property
    def interior_mask_bool(self):
        """
        Get the interior mask as a boolean (N,) mask.
        """
        return self.interior_mask[:, 0].to(torch.bool)

    def training_step(self, batch):
        """
        Train on single batch
        """
        prediction, target, pred_std, _ = self.forward(batch)

        # Compute loss
        batch_loss = torch.mean(
            self.loss(
                prediction, target, pred_std, mask=self.interior_mask_bool
            )
        )  # mean over unrolled times and batch

        log_dict = {"train_loss": batch_loss}
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )
        return batch_loss

    # newer lightning versions requires batch_idx argument, even if unused
    # pylint: disable-next=unused-argument
    def validation_step(self, batch, batch_idx):
        """
        Run validation on single batch
        """
        prediction, target, pred_std, _ = self.forward(batch)

        time_step_loss = torch.mean(
            self.loss(
                prediction, target, pred_std, mask=self.interior_mask_bool
            ),
            dim=0,
        )  # (time_steps-1)
        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        val_log_dict = {
            f"val_loss_unroll{step}": time_step_loss[step - 1]
            for step in self.args.val_steps_to_log
            if step <= len(time_step_loss)
        }
        val_log_dict["val_mean_loss"] = mean_loss
        self.log_dict(
            val_log_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )

        # Store MSEs
        entry_mses = metrics.mse(
            prediction,
            target,
            pred_std,
            mask=self.interior_mask_bool,
            sum_vars=False,
        )  # (B, pred_steps, d_f)
        self.val_metrics["mse"].append(entry_mses)

    def on_validation_epoch_end(self):
        """
        Compute val metrics at the end of val epoch
        """
        # Create error maps for all test metrics
        self.aggregate_and_plot_metrics(self.val_metrics, prefix="val")

        # Clear lists with validation metrics values
        for metric_list in self.val_metrics.values():
            metric_list.clear()

    # pylint: disable-next=unused-argument
    def test_step(self, batch, batch_idx):
        """
        Run test on single batch
        """
        # TODO Here batch_times can be used for plotting routines
        prediction, target, pred_std, batch_times = self.common_step(batch)
        # prediction: (B, pred_steps, num_grid_nodes, d_f) pred_std: (B,
        # pred_steps, num_grid_nodes, d_f) or (d_f,)

        time_step_loss = torch.mean(
            self.loss(
                prediction, target, pred_std, mask=self.interior_mask_bool
            ),
            dim=0,
        )  # (time_steps-1,)
        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        test_log_dict = {
            f"test_loss_unroll{step}": time_step_loss[step - 1]
            for step in self.args.val_steps_to_log
        }
        test_log_dict["test_mean_loss"] = mean_loss

        self.log_dict(
            test_log_dict,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch[0].shape[0],
        )

        # Compute all evaluation metrics for error maps Note: explicitly list
        # metrics here, as test_metrics can contain additional ones, computed
        # differently, but that should be aggregated on_test_epoch_end
        for metric_name in ("mse", "mae"):
            metric_func = metrics.get_metric(metric_name)
            batch_metric_vals = metric_func(
                prediction,
                target,
                pred_std,
                mask=self.interior_mask_bool,
                sum_vars=False,
            )  # (B, pred_steps, d_f)
            self.test_metrics[metric_name].append(batch_metric_vals)

        if self.output_std:
            # Store output std. per variable, spatially averaged
            mean_pred_std = torch.mean(
                pred_std[..., self.interior_mask_bool, :], dim=-2
            )  # (B, pred_steps, d_f)
            self.test_metrics["output_std"].append(mean_pred_std)

        # Save per-sample spatial loss for specific times
        spatial_loss = self.loss(
            prediction, target, pred_std, average_grid=False
        )  # (B, pred_steps, num_grid_nodes)
        log_spatial_losses = spatial_loss[
            :, [step - 1 for step in self.args.val_steps_to_log]
        ]
        self.spatial_loss_maps.append(log_spatial_losses)
        # (B, N_log, num_grid_nodes)

        # Plot example predictions (on rank 0 only)
        if (
            self.trainer.is_global_zero
            and self.plotted_examples < self.n_example_pred
        ):
            # Need to plot more example predictions
            n_additional_examples = min(
                prediction.shape[0],
                self.n_example_pred - self.plotted_examples,
            )

            self.plot_examples(
                batch,
                n_additional_examples,
                prediction=prediction,
                split="test",
            )

    def on_test_epoch_end(self):
        """
        Compute test metrics and make plots at the end of test epoch. Will
        gather stored tensors and perform plotting and logging on rank 0.
        """
        # Create error maps for all test metrics
        self.aggregate_and_plot_metrics(self.test_metrics, prefix="test")

        # Plot spatial loss maps
        spatial_loss_tensor = self.all_gather_cat(
            torch.cat(self.spatial_loss_maps, dim=0)
        )  # (N_test, N_log, num_grid_nodes)
        if self.trainer.is_global_zero:
            mean_spatial_loss = torch.mean(
                spatial_loss_tensor, dim=0
            )  # (N_log, num_grid_nodes)

            loss_map_figs = [
                vis.plot_spatial_error(
                    error=loss_map,
                    datastore=self._datastore,
                    title=f"Test loss, t={t_i} "
                    f"({self._datastore.step_length * t_i} h)",
                )
                for t_i, loss_map in zip(
                    self.args.val_steps_to_log, mean_spatial_loss
                )
            ]

            # log all to same key, sequentially
            for i, fig in enumerate(loss_map_figs):
                key = "test_loss"
                if not isinstance(self.logger, pl.loggers.WandbLogger):
                    key = f"{key}_{i}"
                if hasattr(self.logger, "log_image"):
                    self.logger.log_image(key=key, images=[fig])

            # also make without title and save as pdf
            pdf_loss_map_figs = [
                vis.plot_spatial_error(
                    error=loss_map, datastore=self._datastore
                )
                for loss_map in mean_spatial_loss
            ]
            pdf_loss_maps_dir = os.path.join(
                self.logger.save_dir, "spatial_loss_maps"
            )
            os.makedirs(pdf_loss_maps_dir, exist_ok=True)
            for t_i, fig in zip(self.args.val_steps_to_log, pdf_loss_map_figs):
                fig.savefig(os.path.join(pdf_loss_maps_dir, f"loss_t{t_i}.pdf"))
            # save mean spatial loss as .pt file also
            torch.save(
                mean_spatial_loss.cpu(),
                os.path.join(self.logger.save_dir, "mean_spatial_loss.pt"),
            )

        self.spatial_loss_maps.clear()

    def on_load_checkpoint(self, checkpoint):
        """
        Perform any changes to state dict before loading checkpoint
        """
        loaded_state_dict = checkpoint["state_dict"]

        # Fix for loading older models after IneractionNet refactoring, where
        # the grid MLP was moved outside the encoder InteractionNet class
        if "g2m_gnn.grid_mlp.0.weight" in loaded_state_dict:
            replace_keys = list(
                filter(
                    lambda key: key.startswith("g2m_gnn.grid_mlp"),
                    loaded_state_dict.keys(),
                )
            )
            for old_key in replace_keys:
                new_key = old_key.replace(
                    "g2m_gnn.grid_mlp", "encoding_grid_mlp"
                )
                loaded_state_dict[new_key] = loaded_state_dict[old_key]
                del loaded_state_dict[old_key]
        if not self.restore_opt:
            opt = self.configure_optimizers()
            checkpoint["optimizer_states"] = [opt.state_dict()]

    def all_gather_cat(self, tensor_to_gather):
        """
        Gather tensors across all ranks, and concatenate across dim. 0 (instead
        of stacking in new dim. 0)

        tensor_to_gather: (d1, d2, ...), distributed over K ranks

        returns: (K*d1, d2, ...)
        """
        return self.all_gather(tensor_to_gather).flatten(0, 1)


def plot_examples(batch, n_examples, split, prediction=None):
    """
    Plot the first n_examples forecasts from batch

    batch: batch with data to plot corresponding forecasts for n_examples:
    number of forecasts to plot prediction: (B, pred_steps, num_grid_nodes,
    d_f), existing prediction.
        Generate if None.
    """
    if prediction is None:
        prediction, target, _, _ = self.common_step(batch)

    target = batch[1]
    time = batch[3]

    # Rescale to original data scale
    prediction_rescaled = prediction * self.state_std + self.state_mean
    target_rescaled = target * self.state_std + self.state_mean

    # Iterate over the examples
    for pred_slice, target_slice, time_slice in zip(
        prediction_rescaled[:n_examples],
        target_rescaled[:n_examples],
        time[:n_examples],
    ):
        # Each slice is (pred_steps, num_grid_nodes, d_f)
        self.plotted_examples += 1  # Increment already here

        da_prediction = self._create_dataarray_from_tensor(
            tensor=pred_slice,
            time=time_slice,
            split=split,
            category="state",
        ).unstack("grid_index")
        da_target = self._create_dataarray_from_tensor(
            tensor=target_slice,
            time=time_slice,
            split=split,
            category="state",
        ).unstack("grid_index")

        var_vmin = (
            torch.minimum(
                pred_slice.flatten(0, 1).min(dim=0)[0],
                target_slice.flatten(0, 1).min(dim=0)[0],
            )
            .cpu()
            .numpy()
        )  # (d_f,)
        var_vmax = (
            torch.maximum(
                pred_slice.flatten(0, 1).max(dim=0)[0],
                target_slice.flatten(0, 1).max(dim=0)[0],
            )
            .cpu()
            .numpy()
        )  # (d_f,)
        var_vranges = list(zip(var_vmin, var_vmax))

        # Iterate over prediction horizon time steps
        for t_i, _ in enumerate(zip(pred_slice, target_slice), start=1):
            # Create one figure per variable at this time step
            var_figs = [
                vis.plot_prediction(
                    datastore=self._datastore,
                    title=f"{var_name} ({var_unit}), "
                    f"t={t_i} ({self._datastore.step_length * t_i} h)",
                    vrange=var_vrange,
                    da_prediction=da_prediction.isel(
                        state_feature=var_i, time=t_i - 1
                    ).squeeze(),
                    da_target=da_target.isel(
                        state_feature=var_i, time=t_i - 1
                    ).squeeze(),
                )
                for var_i, (var_name, var_unit, var_vrange) in enumerate(
                    zip(
                        self._datastore.get_vars_names("state"),
                        self._datastore.get_vars_units("state"),
                        var_vranges,
                    )
                )
            ]

            example_i = self.plotted_examples

            for var_name, fig in zip(
                self._datastore.get_vars_names("state"), var_figs
            ):

                # We need treat logging images differently for different
                # loggers. WANDB can log multiple images to the same key,
                # while other loggers, as MLFlow, need unique keys for
                # each image.
                if isinstance(self.logger, pl.loggers.WandbLogger):
                    key = f"{var_name}_example_{example_i}"
                else:
                    key = f"{var_name}_example"

                if hasattr(self.logger, "log_image"):
                    self.logger.log_image(key=key, images=[fig], step=t_i)
                else:
                    warnings.warn(
                        f"{self.logger} does not support image logging."
                    )

            plt.close("all")  # Close all figs for this time step, saves memory

        # Save pred and target as .pt files
        torch.save(
            pred_slice.cpu(),
            os.path.join(
                self.logger.save_dir,
                f"example_pred_{self.plotted_examples}.pt",
            ),
        )
        torch.save(
            target_slice.cpu(),
            os.path.join(
                self.logger.save_dir,
                f"example_target_{self.plotted_examples}.pt",
            ),
        )


# Standard library
import argparse
from dataclasses import dataclass
from typing import List, Set, Tuple

# Third-party
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import xarray as xr
from parse import parse
from torchmetrics import METRIC_REGISTRY


@dataclass(frozen=True)
class HeatmapConfig:
    """Configuration for heatmap metrics, defining the data split and metric."""

    STRING_FORMAT = "{split}:{metric}"
    split: str
    metric: str

    @staticmethod
    def from_string(s: str) -> "HeatmapConfig":
        """Parses a heatmap configuration string into a HeatmapConfig object."""
        result = parse(HeatmapConfig.STRING_FORMAT, s)
        if not result:
            raise ValueError(
                f"Invalid heatmap config format: '{s}'. Expected format is {HeatmapConfig.STRING_FORMAT}."
            )
        return HeatmapConfig(**result)


@dataclass(frozen=True)
class TraceConfig:
    """Configuration for trace metrics, defining the split, metric, variable, and timestep."""

    STRING_FORMAT = "{split}:{metric}:{variable}:{step}"
    split: str
    variable: str
    metric: str
    step: int

    @staticmethod
    def from_string(s: str) -> "TraceConfig":
        """Parses a trace configuration string into a TraceConfig object."""
        result = parse(TraceConfig.STRING_FORMAT, s)
        if not result:
            raise ValueError(
                f"Invalid trace config format: '{s}'. Expected format is {TraceConfig.STRING_FORMAT}."
            )
        return TraceConfig(**result)


@dataclass(frozen=True)
class MetricLoggingConfig:
    """Container for heatmap and trace metric configurations."""

    heatmaps: Set[HeatmapConfig]
    traces: Set[TraceConfig]

    @staticmethod
    def from_args(
        heatmaps: List[str], traces: List[str]
    ) -> "MetricLoggingConfig":
        """Creates a MetricLoggingConfig from command-line arguments."""
        heatmap_configs = {HeatmapConfig.from_string(s) for s in heatmaps}
        trace_configs = {TraceConfig.from_string(s) for s in traces}
        return MetricLoggingConfig(
            heatmaps=heatmap_configs, traces=trace_configs
        )


class MyModel(pl.LightningModule):
    """PyTorch Lightning model for metric logging with trace and heatmap visualization."""

    LOGGED_METRIC_KEY_FORMAT = TraceConfig.STRING_FORMAT

    def __init__(
        self, config: MetricLoggingConfig, ar_steps: int, variables: List[str]
    ):
        """Initializes the model with metric configurations, auto-regressive steps, and variables."""
        super().__init__()
        self.config = config
        self.ar_steps = ar_steps
        self.variables = variables
        self.layer = torch.nn.Linear(10, len(variables))
        self.metrics = self._setup_metrics()

    def _setup_metrics(self):
        """Initializes metrics based on the provided configurations."""
        metrics = torch.nn.ModuleDict()

        for cfg in self.config.traces:
            if cfg.metric in METRIC_REGISTRY:
                split_metrics = metrics.setdefault(
                    cfg.split, torch.nn.ModuleDict()
                )
                key = self.LOGGED_METRIC_KEY_FORMAT.format(
                    split=cfg.split,
                    metric=cfg.metric,
                    variable=cfg.variable,
                    step=cfg.step,
                )
                split_metrics[key] = METRIC_REGISTRY[cfg.metric]()

        for hm in self.config.heatmaps:
            for var in self.variables:
                for step in range(self.ar_steps):
                    key = self.LOGGED_METRIC_KEY_FORMAT.format(
                        split=hm.split,
                        metric=hm.metric,
                        variable=var,
                        step=step,
                    )
                    split_metrics = metrics.setdefault(
                        hm.split, torch.nn.ModuleDict()
                    )
                    split_metrics[key] = METRIC_REGISTRY[hm.metric]()

        return metrics

    def forward(self, x):
        """Performs a forward pass through the model."""
        return self.layer(x)

    def _update_metrics(self, split, preds, targets):
        """Updates metric calculations for a given split."""
        for key, metric in self.metrics.get(split, {}).items():
            _, _, var, step = key.rsplit(":", 3)
            step = int(step)
            metric.update(preds[:, step, :], targets[:, step, :])

    def _log_metrics(self, split):
        """Logs and visualizes metrics as heatmaps."""
        metrics = list({cfg.metric for cfg in self.config.heatmaps})
        heatmap_data = xr.DataArray(
            data=torch.zeros(
                (len(metrics), len(self.variables), self.ar_steps)
            ).numpy(),
            dims=["metric", "variable", "step"],
            coords={
                "metric": metrics,
                "variable": self.variables,
                "step": list(range(self.ar_steps)),
            },
        )

        for key, metric in self.metrics.get(split, {}).items():
            parsed = parse(self.LOGGED_METRIC_KEY_FORMAT, key)
            variable = parsed["variable"]
            step = parsed["step"]

            value = metric.compute()
            self.log(f"{split}_{key}", value, prog_bar=True)
            metric.reset()

            heatmap_data.loc[parsed["metric"], variable, step] = value

        for metric_name in heatmap_data.coords["metric"].values:
            plt.figure()
            plt.imshow(
                heatmap_data.sel(metric=metric_name).values,
                aspect="auto",
                cmap="viridis",
            )
            plt.xticks(
                ticks=range(self.ar_steps),
                labels=[f"Step {i}" for i in range(self.ar_steps)],
            )
            plt.yticks(ticks=range(len(self.variables)), labels=self.variables)
            plt.colorbar(label=metric_name)
            plt.title(f"{split} Heatmap for {metric_name}")
            plt.savefig(f"{split}_heatmap_{metric_name}.png")
            plt.close()

    def training_step(self, batch, batch_idx):
        """Defines a single training step."""
        x, y = batch
        y_hat = self(x)
        self._update_metrics("train", y_hat, y)
        return F.mse_loss(y_hat, y)

    def validation_step(self, batch, batch_idx):
        """Defines a single validation step."""
        x, y = batch
        y_hat = self(x)
        self._update_metrics("val", y_hat, y)
        return F.mse_loss(y_hat, y)

    def test_step(self, batch, batch_idx):
        """Defines a single test step."""
        x, y = batch
        y_hat = self(x)
        self._update_metrics("test", y_hat, y)
        return F.mse_loss(y_hat, y)

    def on_train_epoch_end(self):
        """Logs metrics at the end of a training epoch."""
        self._log_metrics("train")

    def on_validation_epoch_end(self):
        """Logs metrics at the end of a validation epoch."""
        self._log_metrics("val")

    def on_test_epoch_end(self):
        """Logs metrics at the end of a test epoch."""
        self._log_metrics("test")

    def configure_optimizers(self):
        """Configures the optimizer for training."""
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--heatmaps",
        nargs="*",
        default=[],
        help="Heatmap configs as split:metric",
    )
    parser.add_argument(
        "--traces",
        nargs="*",
        default=[],
        help="Trace configs as split:metric:variable:step",
    )
    parser.add_argument(
        "--variables",
        nargs="*",
        default=[],
        help="List of variables to track in heatmaps",
    )
    parser.add_argument(
        "--ar_steps",
        type=int,
        default=10,
        help="Number of auto-regressive steps for heatmap",
    )
    args = parser.parse_args()

    config = MetricLoggingConfig.from_args(args.heatmaps, args.traces)
    model = MyModel(config, args.ar_steps, args.variables)

    # Example training call would go here
