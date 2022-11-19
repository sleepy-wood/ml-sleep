from typing import Any

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim.swa_utils import AveragedModel
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)


class SleepNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_hidden: int,
        dropout_p: float,
    ) -> None:
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(num_hidden - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p),
            ]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.layers = nn.Sequential(*self.layers)

    def forward(
        self, accs: torch.Tensor, hvs: torch.Tensor, hds: torch.Tensor
    ) -> torch.Tensor:
        hvs, hds = hvs / 60, hds / 60
        return self.layers(torch.cat([accs, hvs, hds], dim=-1)).squeeze(dim=-1)


class EMAModel(AveragedModel):
    def __init__(
        self,
        net: nn.Module,
        device: str,
        decay: float = 0.999,
    ) -> None:
        def ema_avg(avg_net_param, net_param, _num_averaged):
            return decay * avg_net_param + (1 - decay) * net_param

        super().__init__(net, device, ema_avg, use_buffers=True)


class SleepModel(LightningModule):
    # pylint: disable=unused-argument
    def __init__(
        self,
        nwin: int = 5,
        net_hidden_dim: int = 128,
        net_num_hidden: int = 3,
        net_dropout_p: float = 0.1,
        opt_lr: float = 1e-4,
        opt_wd: float = 0.05,
        opt_beta1: float = 0.9,
        opt_beta2: float = 0.999,
        ema_decay: float = 0.999,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.net = SleepNet(
            in_dim=3 + 2 * nwin,
            out_dim=1,
            hidden_dim=net_hidden_dim,
            num_hidden=net_num_hidden,
            dropout_p=net_dropout_p,
        )
        self.net_ema = EMAModel(self.net, self.device, ema_decay)
        self.loss = nn.BCEWithLogitsLoss()
        self.metrics = MetricCollection(
            dict(
                acc=BinaryAccuracy(),
                mp=BinaryPrecision(average="macro"),
                mr=BinaryRecall(average="macro"),
                mf1=BinaryF1Score(average="macro"),
                wp=BinaryPrecision(average="weighted"),
                wr=BinaryRecall(average="weighted"),
                wf1=BinaryF1Score(average="weighted"),
            ),
            prefix="val/",
        )
        self.metrics_ema = self.metrics.clone(prefix="val_ema/")

    # pylint: disable=arguments-differ
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        accs, hvs, hds, label = batch
        logits = self.net(accs, hvs, hds)
        loss = self.loss(logits, label)
        self.log("train/loss", loss)
        return loss

    def on_train_batch_end(
        self,
        outputs: Any,
        batch: tuple,
        batch_idx: int,
    ) -> None:
        self.net_ema.update_parameters(self.net)

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        accs, hvs, hds, label = batch
        logits = self.net(accs, hvs, hds)
        self.log_dict(self.metrics(logits, label))
        logits_ema = self.net_ema(accs, hvs, hds)
        self.log_dict(self.metrics_ema(logits_ema, label), prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.opt_lr,
            weight_decay=self.hparams.opt_wd,
            betas=(self.hparams.opt_beta1, self.hparams.opt_beta2),
        )
