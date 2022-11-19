from pytorch_lightning.cli import LightningCLI

from app.data import SleepDataModule
from app.model import SleepModel


def main() -> None:
    _ = LightningCLI(SleepModel, SleepDataModule)
