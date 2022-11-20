import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from pytorch_lightning.cli import LightningCLI

from app.data import SleepDataModule
from app.model import SleepModel


def main() -> None:
    _ = LightningCLI(SleepModel, SleepDataModule)
