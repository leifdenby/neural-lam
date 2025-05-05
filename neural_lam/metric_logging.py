# Standard library
import dataclasses
from typing import List, Set

# Third-party
import parse
import torch




@dataclasses.dataclass(frozen=True)
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
        result = parse(TraceConfig.STRING_FORMAT, s)
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




    def _setup_metrics_logging(self):
        """Initializes logging of metrics based on the provided configurations."""
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