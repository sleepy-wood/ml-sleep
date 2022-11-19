import random
from collections import namedtuple
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


def read_data(sid):
    hrate = pd.read_csv(
        f"../data/heart_rate/{sid}_heartrate.txt",
        header=None,
        names=["tssec", "hrate"],
    )
    sleep = pd.read_csv(
        f"../data/labels/{sid}_labeled_sleep.txt",
        header=None,
        names=["tssec", "sleep"],
        sep=" ",
    )
    accel = pd.read_csv(
        f"../data/motion/{sid}_acceleration.txt",
        header=None,
        names=["tssec", "acc_x", "acc_y", "acc_z"],
        sep=" ",
    )
    hrate = hrate.sort_values(by="tssec")
    sleep = sleep.sort_values(by="tssec")
    accel = accel.sort_values(by="tssec")
    sleep["sleep"] = sleep["sleep"] > 0
    assert all(hrate.isna().sum() == 0)
    assert all(sleep.isna().sum() == 0)
    assert all(accel.isna().sum() == 0)
    return hrate, sleep, accel


def gen_row(
    row: namedtuple,
    hrate: pd.DataFrame,
    accel: pd.DataFrame,
    rng: np.random.Generator,
    nwin: int = 5,
) -> tuple:
    t = row.tssec
    i = accel["tssec"].searchsorted(t)
    if i == len(accel):
        i -= 1
    acc_x, acc_y, acc_z = accel.iloc[i, 1:]
    prev_t, prev_j = t, None
    hvs, hds = [], []
    for nw in range(nwin):
        low, high = (0, 6) if nw == 0 else (0.5, 7)
        diff = rng.uniform(low=low, high=high) * 60
        j = hrate["tssec"].searchsorted(prev_t - diff)
        if j == len(hrate):
            j -= 1
        if j == prev_j:
            j -= 1
        hrate_t, hrate_v = hrate.iloc[j, :]
        hvs += [hrate_v]
        hds += [prev_t - hrate_t]
        prev_t = hrate_t
        prev_j = j
    return [acc_x, acc_y, acc_z], hvs, hds, row.sleep


class SleepDataset(Dataset):
    def __init__(
        self,
        data: Sequence[Sequence[pd.DataFrame]],
        seed: int,
        train: bool,
        nwin: int = 5,
    ) -> None:
        self.data = data
        self.lens = [len(d[1]) for d in data]
        self.cumlen = np.cumsum(self.lens)
        self.seed = seed
        self.nwin = nwin
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return sum(self.lens)

    def __getitem__(self, idx: int) -> tuple:
        if not self.train:
            self.rng = np.random.default_rng(self.seed + idx)
        sid = np.searchsorted(self.cumlen, idx)
        if sid == len(self.cumlen):
            sid -= 1
        if sid > 0:
            idx -= self.cumlen[sid - 1]
        hrate, sleep, accel = self.data[sid]
        row = sleep.iloc[idx, :]
        accs, hvs, hds, label = gen_row(row, hrate, accel, self.rng, self.nwin)
        accs, hvs, hds = [
            torch.tensor(x, dtype=torch.float) for x in (accs, hvs, hds)
        ]
        return accs, hvs, hds, label


class SleepDataModule(LightningDataModule):
    # pylint: disable=unused-argument
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        val_num: int,
        val_seed: int,
        seed: int,
        nwin: int = 5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str = None) -> None:
        sids = sorted(
            int(p.stem.split("_")[0])
            for p in Path("../data/labels").glob("*.txt")
        )
        data = [read_data(sid) for sid in sids]
        random.Random(self.hparams.val_seed).shuffle(data)
        val_num = self.hparams.val_num
        train_data, val_data = data[:-val_num], data[-val_num:]
        if stage == "fit" or stage is None:
            # pylint: disable=attribute-defined-outside-init
            self.train_data = SleepDataset(
                train_data,
                self.hparams.seed,
                train=True,
                nwin=self.hparams.nwin,
            )
            self.val_data = SleepDataset(
                val_data,
                self.hparams.seed,
                train=False,
                nwin=self.hparams.nwin,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )
