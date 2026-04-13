# Standard library
import dataclasses
from typing import List, Set

# Third-party
import matplotlib.pyplot as plt
import parse
import torch
import xarray as xr
from torchmetrics import METRIC_REGISTRY


@dataclasses.dataclass(frozen=True)
class HeatmapConfig:
    """Configuration for heatmap metrics, defining the data split and metric."""

    STRING_FORMAT = "{split}:{metric}"
    split: str
    metric: str

    @staticmethod
    def from_string(s: str) -> "HeatmapConfig":
        """Parses a heatmap configuration string into a HeatmapConfig object."""
        result = parse.parse(HeatmapConfig.STRING_FORMAT, s)
        if not result:
            raise ValueError(
                f"Invalid heatmap config format: '{s}'. Expected format is {HeatmapConfig.STRING_FORMAT}."
            )
        return HeatmapConfig(**result)


@dataclasses.dataclass(frozen=True)
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
        result = parse.parse(TraceConfig.STRING_FORMAT, s)
        if not result:
            raise ValueError(
                f"Invalid trace config format: '{s}'. Expected format is {TraceConfig.STRING_FORMAT}."
            )
        return TraceConfig(**result)


@dataclasses.dataclass(frozen=True)
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


class MetricLoggingMixin:
    LOGGED_METRIC_KEY_FORMAT = TraceConfig.STRING_FORMAT

    def _setup_metrics_logging(self):
        """Initializes logging of metrics based on the provided configurations."""
        metrics = torch.nn.ModuleDict()

        if not hasattr(self, "_logging_config"):
            return metrics

        for cfg in self._logging_config.traces:
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

        for hm in self._logging_config.heatmaps:
            for var in self._datastore.get_vars_names("state"):
                for step in range(
                    self._config.num_future_forcing_steps
                ):  # or self.ar_steps?
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

    def _update_metrics(self, split, preds, targets):
        """Updates metric calculations for a given split."""
        if not hasattr(self, "metrics"):
            return

        for key, metric in self.metrics.get(split, {}).items():
            _, _, var, step = key.rsplit(":", 3)
            step = int(step)
            metric.update(preds[:, step, :], targets[:, step, :])

    def _log_metrics(self, split):
        """Logs and visualizes metrics as heatmaps."""
        if not hasattr(self, "_logging_config") or not hasattr(self, "metrics"):
            return

        metrics = list({cfg.metric for cfg in self._logging_config.heatmaps})
        if not metrics:
            return

        variables = self._datastore.get_vars_names("state")
        ar_steps = self._config.num_future_forcing_steps

        heatmap_data = xr.DataArray(
            data=torch.zeros((len(metrics), len(variables), ar_steps)).numpy(),
            dims=["metric", "variable", "step"],
            coords={
                "metric": metrics,
                "variable": variables,
                "step": list(range(ar_steps)),
            },
        )

        for key, metric in self.metrics.get(split, {}).items():
            parsed = parse.parse(self.LOGGED_METRIC_KEY_FORMAT, key)
            variable = parsed["variable"]
            step = int(parsed["step"])

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
                ticks=range(ar_steps),
                labels=[f"Step {i}" for i in range(ar_steps)],
            )
            plt.yticks(ticks=range(len(variables)), labels=variables)
            plt.colorbar(label=metric_name)
            plt.title(f"{split} Heatmap for {metric_name}")

            # Save fig and log
            fig_path = f"{split}_heatmap_{metric_name}.png"
            plt.savefig(fig_path)
            if hasattr(self.logger, "log_image"):
                self.logger.log_image(
                    key=f"{split}_heatmap_{metric_name}", images=[fig_path]
                )

            plt.close()
