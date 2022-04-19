import argparse
import logging
import os
import time
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,4"
# os.environ["MASTER_ADDR"]="10.119.81.16"
# os.environ["MASTER_PORT"]="42069"
os.environ["NODE_RANK"]="0"

import json
import pytorch_lightning as pl
import torch
import wandb

from ipa_refine.config import model_config
from ipa_refine.data.data_modules import RefinementDataModule
from ipa_refine.model.refinement_model import RefinementModel
from ipa_refine.utils.callbacks import EarlyStoppingVerbose
from ipa_refine.utils.argparse import remove_arguments
from ipa_refine.utils.loss import RefinementModelLoss
from ipa_refine.utils.seed import seed_everything

hyper_config = dict(
    learning_rate=1e-3,
    trans_scale_factor=0.86,
    no_blocks=8,
    c_ipa=64
)

run = wandb.init(
    project="ipa_refine_tuning",
    config=hyper_config
)

###   Replace hyperparameters from Sweep in model config

def set_hyperparams_config(default_configs, hyperparams):
    for key in hyperparams.keys():
        if key in default_configs["model"]["structure_module"]:
            default_configs["model"]["structure_module"][key] = hyperparams[key]
    return default_configs


###    Model training loop

class RefinementModelWrapper(pl.LightningModule):

    def __init__(self, config):
        super(RefinementModelWrapper, self).__init__()
        self.config = config
        modelConfig = config.model
        self.model = RefinementModel(config, **modelConfig["structure_module"])
        self.loss = RefinementModelLoss(config.loss)

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss, loss_metrics = self.loss(outputs, batch)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # Calculate validation loss
        outputs = self(batch)
        loss, loss_metrics = self.loss(outputs, batch)
        return {"val_loss": loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('loss', avg_loss)
        wandb.log({"train_loss": avg_loss})
        return None

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)
        wandb.log({"val_loss": avg_loss})
        torch.save(self.model.state_dict(), os.path.join(run.dir, 'model_weights'))
        return {'val_loss': avg_loss, 'log': self.log}

    def configure_optimizers(self,
        learning_rate: float = wandb.config["learning_rate"],
        eps: float = 1e-8
    ) -> torch.optim.Adam:
        # Ignored as long as a DeepSpeed optimizer is configured
        return torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            eps=eps
        )


def main(args):

    if(args.seed is not None):
        seed_everything(args.seed)

    config = model_config(
        "model_1",
        train=True,
        low_prec=(args.precision == 16)
    )
    hyperparam_config = set_hyperparams_config(config, wandb.config)
    model_module = RefinementModelWrapper(hyperparam_config)
    data_module = RefinementDataModule(
        config=config.data,
        batch_seed=args.seed
    )

    data_module.prepare_data()
    data_module.setup()

    callbacks = []

    # Early Stopping
    if(args.early_stopping):
        es = EarlyStoppingVerbose(
            monitor="val_loss",
            min_delta=args.min_delta,
            patience=args.patience,
            verbose=False,
            mode="min",
            check_finite=True,
            strict=True,
        )
        callbacks.append(es)

    # Setup multi GPU Config
    if args.gpus > 1 or args.num_nodes > 1:
        strategy = "ddp"
    else:
        strategy = None

    trainer = pl.Trainer.from_argparse_args(
        args,
        strategy=strategy,
        callbacks=callbacks
    )

    trainer.fit(
        model_module,
        datamodule=data_module,
        ckpt_path=None,
    )


def bool_type(bool_str: str):
    bool_str_lower = bool_str.lower()
    if bool_str_lower in ('false', 'f', 'no', 'n', '0'):
        return False
    elif bool_str_lower in ('true', 't', 'yes', 'y', '1'):
        return True
    else:
        raise ValueError(f'Cannot interpret {bool_str} as bool')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--deepspeed_config_path", type=str, default=None,
        help="Path to DeepSpeed config. If not provided, DeepSpeed is disabled"
    )
    parser.add_argument(
        "--checkpoint_best_val", type=bool_type, default=True,
        help="""Whether to save the model parameters that perform best during
                validation"""
    )
    parser.add_argument(
        "--early_stopping", type=bool_type, default=True,
        help="Whether to stop training when validation loss fails to decrease"
    )
    parser.add_argument(
        "--min_delta", type=float, default=0,
        help="""The smallest decrease in validation loss that counts as an 
                improvement for the purposes of early stopping"""
    )
    parser.add_argument(
        "--patience", type=int, default=7,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--log_performance", action='store_true',
        help="Measure performance"
    )
    parser = pl.Trainer.add_argparse_args(parser)

    # Disable the initial validation pass
    parser.set_defaults(
        num_sanity_val_steps=0,
        gpus=1,
        num_nodes=1,
        max_epochs=400
    )

    # Remove some buggy/redundant arguments introduced by the Trainer
    remove_arguments(parser, ["--accelerator", "--resume_from_checkpoint"])

    args = parser.parse_args()

    if(args.seed is None and
        ((args.gpus is not None and args.gpus > 1) or
         (args.num_nodes is not None and args.num_nodes > 1))):
        raise ValueError("For distributed training, --seed must be specified")

    main(args)
