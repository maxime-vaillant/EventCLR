import mlflow
import torch
from pytorch_lightning import LightningModule
from spikingjelly.activation_based import functional
from torch import optim

from losses.ntxent import TemporalNTXentLoss
from utils.metrics import encoder_representation_metrics


class LitSelfSupervised(LightningModule):
    def __init__(
            self,
            model,
            lr=1e-4,
            weight_decay=0.0,
            max_epochs=100,
            ntxent_temp=0.2,
            strategy='naive',
    ):
        super().__init__()

        self.model = model

        self.criterion = TemporalNTXentLoss(temperature=ntxent_temp, strategy=strategy)

        self.lr = lr
        self.weight_decay = weight_decay
        self.eta_min = lr * 0.01
        self.max_epochs = max_epochs

        self.representation_log = 20

        self._h_i_buffer = []
        self._h_j_buffer = []

    def forward(self, x_i, x_j):
        h_i, z_i = self.model(x_i)
        functional.reset_net(self.model)

        h_j, z_j = self.model(x_j)
        functional.reset_net(self.model)

        return h_i, h_j, z_i, z_j

    def _shared_step(self, batch, batch_idx, mode):
        (x_i, x_j), target = batch

        # (B, T, C, H, W) -> (T, B, C, H, W)
        x_i = torch.permute(x_i, (1, 0, 2, 3, 4))
        x_j = torch.permute(x_j, (1, 0, 2, 3, 4))

        h_i, h_j, z_i, z_j = self.forward(x_i, x_j)

        pretrain_loss = self.criterion(z_i, z_j)
        loss = pretrain_loss

        self.log(f'{mode}_pretrain_loss', pretrain_loss, prog_bar=True, on_epoch=True, sync_dist=True)

        if mode == 'train' and self.current_epoch % self.representation_log == 0:
            # h shape: [T, B, D] in multi-step mode — average over time
            self._h_i_buffer.append(h_i.mean(dim=0).detach())
            self._h_j_buffer.append(h_j.mean(dim=0).detach())

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, mode="train")

    def on_train_epoch_end(self) -> None:
        mlflow.log_metric("pretrain_loss", self.trainer.callback_metrics['train_pretrain_loss_epoch'].item(),
                          step=self.current_epoch)

        if self.current_epoch % self.representation_log == 0 and self._h_i_buffer:
            H_i = torch.cat(self._h_i_buffer, dim=0)  # [N, D]
            H_j = torch.cat(self._h_j_buffer, dim=0)  # [N, D]

            step = self.current_epoch
            for name, value in encoder_representation_metrics(H_i, H_j, epoch=step).items():
                mlflow.log_metric(f"enc_{name}", value, step=step)

            self._h_i_buffer.clear()
            self._h_j_buffer.clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=self.eta_min, T_max=self.max_epochs)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
