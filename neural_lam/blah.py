# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "parse",
# ]
# ///

# Standard library
import dataclasses

# Third-party
import parse


@dataclasses.dataclass
class MetricsWatchShorthand:
    split: str
    metric: str
    FORMAT = "{split}_{metric}"

    # create init that works from serialised format
    def __init__(self, s: str):
        parsed = parse.parse(self.FORMAT, s)
        self.split = parsed["split"]
        self.metric = parsed["metric"]


if __name__ == "__main__":
    print(MetricsWatchShorthand("train_mse"))
