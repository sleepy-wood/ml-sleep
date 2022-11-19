import pytorch_lightning as pl

from app.data import SleepDataModule


def main() -> None:
    pl.seed_everything(seed=1234)
    print("Hello World!")
