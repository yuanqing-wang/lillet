import numpy as np
import torch
import lightning as pl
from lightning.pytorch.loggers import CSVLogger

def run(args):
    from lillet.data.ccsd import CCSD
    data = CCSD(args.name, batch_size=args.batch_size, normalize=True)
    data.setup()

    from lillet.model import LilletModel
    model = LilletModel(
        fine_grain_particles=9,
        coarse_grain_particles=16,
        heads=4,
        hidden_features=args.hidden_features,
        num_rbf=args.num_rbf,
        lr=args.lr,
        weight_decay=args.weight_decay,
        E_MEAN=data.E_MEAN,
        E_STD=data.E_STD,
    )

    # model = torch.compile(model)

    from lightning.pytorch.callbacks import ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_energy",
        mode="max",
        verbose=True,
        save_weights_only=True,
        dirpath="checkpoints",
    )

    trainer = pl.Trainer(
        max_epochs=10000, 
        log_every_n_steps=1, 
        logger=CSVLogger("logs", name="junmai"),
        devices="auto",
        accelerator="auto",
        enable_progress_bar=False,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, data)

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run MD simulation")
    parser.add_argument("--name", type=str, default="ethanol")
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--num-rbf", type=int, default=100)
    parser.add_argument("--hidden-features", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    run(args)
