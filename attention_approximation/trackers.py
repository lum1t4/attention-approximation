from pathlib import Path
from typing import Dict, List, Optional, Type

from attention_approximation.pytorch import rank_zero_only, RANK

try:
    import wandb
    # from wandb import Artifact  # uncomment if you want proper typing
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


class Tracker:
    """Base tracker that does nothing by default."""

    def __init__(self, config: Dict):
        self.config = config

    @rank_zero_only
    def log(self, x, y: Optional[float] = None, step: Optional[int] = None) -> None:
        """Log a key/value pair or dictionary of values."""
        return None

    @rank_zero_only
    def log_model(self, checkpoint: Path, aliases: Optional[List[str]] = None) -> None:
        """Log a model checkpoint."""
        return None

    @rank_zero_only
    def finish(self) -> None:
        """Clean up resources if needed."""
        return None


class WandbTracker(Tracker):
    """WandB implementation of the Tracker."""

    def __init__(self, config: Dict):
        if not WANDB_AVAILABLE:
            raise ImportError("wandb is not available. Please install it to use WandbTracker.")

        super().__init__(config)
        self.project: str = config["project"]
        self.run_name: str = config.get("name", None)


        if RANK in {-1, 0}:
            self.run = wandb.init(
                project=self.project,
                name=self.run_name,
                config=config,
                allow_val_change=True,
            )

            if "monitor" in config and "mode" in config:
                self.run.define_metric(config["monitor"], summary=config["mode"])

    @rank_zero_only
    def log(self, x, y: Optional[float] = None, step: Optional[int] = None) -> None:
        if isinstance(x, dict):
            self.run.log(x, step=step)
        else:
            self.run.log({x: y}, step=step)

    @rank_zero_only
    def log_model(self, checkpoint: Path, aliases: Optional[List[str]] = None) -> None:
        aliases = aliases or ["last"]
        artifact = wandb.Artifact(f"run_{wandb.run.id}_model", type="model")
        artifact.add_file(str(checkpoint), name=checkpoint.name)
        wandb.run.log_artifact(artifact, aliases=aliases)

    @rank_zero_only
    def finish(self) -> None:
        wandb.finish()


def AutoTracker(name: str = "wandb", config: dict = {}):
    if name == "wandb" and WANDB_AVAILABLE:
        return WandbTracker(config)
    return Tracker(config)
