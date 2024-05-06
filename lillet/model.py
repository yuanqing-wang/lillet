import torch
import lightning as pl
from .layer import LilletLayer
from .mapping import InductiveMapping
from .radial import ExpNormalSmearing

class LilletModel(pl.LightningModule):
    def __init__(
            self,
            fine_grain_particles: int,
            coarse_grain_particles: int,
            heads: int,
            num_rbf: int,
            hidden_features: int,
            activation: torch.nn.Module = torch.nn.SiLU(),
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
            factor: float = 0.5,
            patience: int = 10,
            E_MEAN: float = 0.0,
            E_STD: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.layer = LilletLayer(
            mapping=InductiveMapping(
                fine_grain_particles=fine_grain_particles,
                coarse_grain_particles=coarse_grain_particles,
                heads=heads,
            ),
            smearing=ExpNormalSmearing(num_rbf=num_rbf),
            hidden_features=hidden_features,
            activation=activation,
        )

    def forward(
            self,
            x: torch.Tensor,
    ):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        R, E, F, Z = batch
        R.requires_grad_(True)
        E_hat = self(R)
        F_hat = -torch.autograd.grad(E_hat.sum(), R, create_graph=True)[0]
        loss_energy = torch.nn.functional.mse_loss(E_hat, E)
        self.log("train_loss_energy", loss_energy)
        loss_force = torch.nn.functional.mse_loss(F_hat, F)
        self.log("train_loss_force", loss_force)
        loss = 1e-2 * loss_energy + loss_force
        return loss
    
    def validation_step(self, batch, batch_idx):
        R, E, F, Z = batch
        R.requires_grad_(True)
        with torch.set_grad_enabled(True):
            E_hat = self(R) * self.E_STD + self.E_MEAN
            F_hat = -torch.autograd.grad(E_hat.sum(), R, create_graph=True)[0]
        loss_energy = torch.nn.functional.l1_loss(E_hat, E)
        loss_force = torch.nn.functional.l1_loss(F_hat, F)
        self.validation_step_outputs.append((loss_energy, loss_force))

    def on_validation_epoch_end(self):
        loss_energy, loss_force = zip(*self.validation_step_outputs)
        loss_energy = torch.stack(loss_energy).mean()
        loss_force = torch.stack(loss_force).mean()
        self.log("val_loss_energy", loss_energy)
        self.log("val_loss_force", loss_force)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="min", 
            factor=self.factor, 
            patience=self.patience, 
            min_lr=1e-6,
            verbose=True,
        )

        scheduler = {
            "scheduler": scheduler,
            "monitor": "val_loss_energy",
        }
    
        return [optimizer], [scheduler]
