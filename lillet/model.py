from re import L
import torch
import lightning as pl
from .layer import Linear, Outer

class SimpleOuterModel(torch.nn.Module):
    def __init__(
            self,
            in_particles: int,
            hidden_features: int,
            activation: torch.nn.Module = torch.nn.SiLU(),
    ):
        super().__init__()
        self.outer = Outer()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_particles ** 2, hidden_features),
            activation,
            torch.nn.Linear(hidden_features, 1),
        )

    def forward(
            self,
            X: torch.Tensor,
    ):
        X = self.outer(X)
        X = X.flatten(-2, -1)
        X = self.fc(X)
        return X


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
        self.layer = SimpleOuterModel(
            in_particles=fine_grain_particles,
            hidden_features=hidden_features,
            activation=activation,
        )
        self.validation_step_outputs = []
        self.test_step_outputs = []

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
        self.log("train_loss_energy", loss_energy.item() ** 0.5 * self.hparams.E_STD)
        loss_force = torch.nn.functional.mse_loss(F_hat, F)
        self.log("train_loss_force", loss_force.item() ** 0.5 * self.hparams.E_STD)
        loss = 1e-2 * loss_energy + loss_force
        return loss
    
    def validation_step(self, batch, batch_idx):
        R, E, F, Z = batch
        R.requires_grad_(True)
        with torch.set_grad_enabled(True):
            E_hat = self(R) * self.hparams.E_STD + self.hparams.E_MEAN
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

    def test_step(self, batch, batch_idx):
        self.testing = False
        R, E, F, Z = batch
        R.requires_grad_(True)
        with torch.set_grad_enabled(True):
            E_hat = self(R) * self.hparams.E_STD + self.hparams.E_MEAN
            F_hat = -torch.autograd.grad(E_hat.sum(), R, create_graph=True)[0]
        loss_energy = torch.nn.functional.l1_loss(E_hat, E)
        loss_force = torch.nn.functional.l1_loss(F_hat, F)
        self.test_step_outputs.append((loss_energy, loss_force))
    
    def on_test_epoch_end(self):
        loss_energy, loss_force = zip(*self.test_step_outputs)
        loss_energy = torch.stack(loss_energy).mean()
        loss_force = torch.stack(loss_force).mean()
        self.log("test_loss_energy", loss_energy)
        self.log("test_loss_force", loss_force)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="min", 
            factor=self.hparams.factor, 
            patience=self.hparams.patience, 
            min_lr=1e-8,
            verbose=True,
        )

        scheduler = {
            "scheduler": scheduler,
            "monitor": "val_loss_energy",
        }
    
        return [optimizer], [scheduler]
    
    def test_with_grad(self, data):
        losses_energy, losses_force = [], []
        for R, E, F, Z in data.test_dataloader():
            R.requires_grad = True
            if torch.cuda.is_available():
                R = R.cuda()
                E = E.cuda()
                F = F.cuda()
                Z = Z.cuda()
            with torch.set_grad_enabled(True):
                E_hat = self(R) * self.hparams.E_STD + self.hparams.E_MEAN
                F_hat = -torch.autograd.grad(E_hat.sum(), R, create_graph=True)[0]
            loss_energy = torch.nn.functional.l1_loss(E_hat, E)
            loss_force = torch.nn.functional.l1_loss(F_hat, F)
            losses_energy.append(loss_energy)
            losses_force.append(loss_force)
        return torch.stack(losses_energy).mean(), torch.stack(losses_force).mean()
