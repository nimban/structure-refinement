import argparse
import logging
import os
import time
import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
# os.environ["MASTER_ADDR"]="10.119.81.16"
# os.environ["MASTER_PORT"]="42069"
# os.environ["NODE_RANK"]="0"

import random
import time
import json

import numpy as np
import optuna
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.plugins.training_type import DeepSpeedPlugin
from pytorch_lightning.plugins.environments import SLURMEnvironment
import torch

from ipa_refine.config import model_config
from ipa_refine.data.data_modules import RefinementDataModule
from ipa_refine.model.refinement_model import RefinementModel
from ipa_refine.utils.callbacks import (
    EarlyStoppingVerbose,
)
from ipa_refine.utils.exponential_moving_average import ExponentialMovingAverage
from ipa_refine.utils.argparse import remove_arguments
from ipa_refine.utils.loss import RefinementModelLoss
from ipa_refine.old_scripts.eval_drmsd import eval_drmsd_loss_batch
from ipa_refine.utils.seed import seed_everything
from ipa_refine.utils.tensor_utils import tensor_tree_map
# from old_scripts.zero_to_fp32 import (
#     get_fp32_state_dict_from_zero_checkpoint
# )
from ipa_refine.utils.logger import PerformanceLoggingCallback
import wandb

directory = str(datetime.datetime.now())[0:-7]
base_path = os.path.join('/home/nn122/remote_runtime/newfold/ipa_refine/outputs', directory)
os.mkdir(base_path)
# wandb_logger = WandbLogger()

hyper_config = dict(
    learning_rate=1e-3,
    trans_scale_factor=1,
    no_blocks=8,
    c_ipa=128
)

run = wandb.init(
  project="ipa_refine",
  notes="first run",
  tags=[],
  config=hyper_config,
)

# Write Val and Train losses to file
def save_state_to_file(self, losses):
    with open(os.path.join(run.dir, 'loss_curve.json'), 'w') as convert_file:
        convert_file.write(json.dumps(losses))
    torch.save(self.model.state_dict(), os.path.join(run.dir, 'model_weights'))

global losses
losses = {
    'learning_rate': float,
    'num_blocks': int,
    'scale_factor': float,
    'train_fape': [],
    'train_drmsd': [],
    'val_fape': [],
    'val_drmsd': [],
    'train_fape_epoch': [],
    'train_drmsd_epoch': [],
    'val_drmsd_epoch': []
}


class RefinementModelWrapper(pl.LightningModule):

    def __init__(self, config):
        super(RefinementModelWrapper, self).__init__()
        self.config = config
        modelConfig = config.model
        losses['num_blocks'] = modelConfig['structure_module']['no_blocks']
        losses['scale_factor'] = modelConfig['structure_module']['trans_scale_factor']
        losses['learning_rate'] = hyper_config['learning_rate']
        modelConfig["structure_module"]["no_blocks"] = hyper_config['no_blocks']
        modelConfig["structure_module"]["c_ipa"] = hyper_config['c_ipa']
        modelConfig["structure_module"]["trans_scale_factor"] = hyper_config['trans_scale_factor']
        self.model = RefinementModel(config, **modelConfig["structure_module"])
        self.loss = RefinementModelLoss(config.loss)
        # self.cached_weights = None
        self.ema = ExponentialMovingAverage(
            model=self.model, decay=config.ema.decay
        )

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss, loss_metrics = self.loss(outputs, batch)
        drmsd_loss = eval_drmsd_loss_batch(outputs, batch['y_features'])
        losses['train_fape_epoch'].append(float(loss))
        losses['train_drmsd_epoch'].append(float(drmsd_loss))
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        # Calculate validation loss
        outputs = self(batch)
        loss, loss_metrics = self.loss(outputs, batch)
        drmsd_loss = eval_drmsd_loss_batch(outputs, batch['y_features'])
        losses['val_drmsd_epoch'].append(float(drmsd_loss))
        return {"val_loss": loss}

    def training_epoch_end(self, outputs):
        train_size = len(losses['train_fape_epoch'])
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        losses['train_fape'].append(sum(losses['train_fape_epoch'])/train_size)
        losses['train_drmsd'].append(sum(losses['train_drmsd_epoch'])/train_size)
        losses['train_fape_epoch'].clear()
        losses['train_drmsd_epoch'].clear()
        save_state_to_file(self, losses)
        self.log('train_loss', avg_loss)
        train_drmsd = float(losses['train_drmsd'][-1])
        wandb.log({"train_loss": avg_loss, "train_drmsd": train_drmsd})
        # wandb.log({"train_drmsd": losses['train_drmsd']})
        return None

    def validation_epoch_end(self, outputs):
        val_size = len(losses['val_drmsd_epoch'])
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        log = {'val_loss': avg_loss}
        losses['val_fape'].append(float(avg_loss))
        losses['val_drmsd'].append(sum(losses['val_drmsd_epoch'])/val_size)
        losses['val_drmsd_epoch'].clear()
        save_state_to_file(self, losses)
        self.log('val_loss', avg_loss)
        val_drmsd = float(losses['val_drmsd'][-1])
        wandb.log({"val_loss": avg_loss, "val_drmsd": val_drmsd})
        # wandb.log({"val_drmsd": losses['val_drmsd']})
        return {'val_loss': avg_loss, 'log': log}

    def configure_optimizers(self,
        learning_rate: float = hyper_config['learning_rate'],
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
    model_module = RefinementModelWrapper(config)
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
    # if args.gpus > 1 or args.num_nodes > 1:
    #     strategy = "ddp"
    # else:
    #     strategy = None

    strategy = None

    trainer = pl.Trainer.from_argparse_args(
        args,
        strategy=strategy,
        callbacks=callbacks,
        # logger=wandb_logger
    )

    trainer.fit(
        model_module,
        datamodule=data_module,
        ckpt_path=None,
    )

    save_state_to_file(trainer, losses)

    trainer.save_checkpoint(
        os.path.join(trainer.logger.log_dir, "checkpoints", "final.ckpt")
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
    # parser.add_argument(
    #     "train_data_dir", type=str,
    #     help="Directory containing training mmCIF files"
    # )
    parser.add_argument(
        "output_dir", type=str,
        help='''Directory in which to output checkpoints, logs, etc. Ignored
                if not on rank 0'''
    )
    # parser.add_argument(
    #     "--val_data_dir", type=str, default=None,
    #     help="Directory containing validation mmCIF files"
    # )
    # parser.add_argument(
    #     "--val_alignment_dir", type=str, default=None,
    #     help="Directory containing precomputed validation alignments"
    # )
    # parser.add_argument(
    #     "--train_mapping_path", type=str, default=None,
    #     help='''Optional path to a .json file containing a mapping from
    #             consecutive numerical indices to sample names. Used to filter
    #             the training set'''
    # )
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
        "--early_stopping", type=bool_type, default=False,
        help="Whether to stop training when validation loss fails to decrease"
    )
    parser.add_argument(
        "--min_delta", type=float, default=0,
        help="""The smallest decrease in validation loss that counts as an 
                improvement for the purposes of early stopping"""
    )
    parser.add_argument(
        "--patience", type=int, default=10,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--resume_from_ckpt", type=str, default=None,
        help="Path to a model checkpoint from which to restore training state"
    )
    parser.add_argument(
        "--resume_model_weights_only", type=bool_type, default=False,
        help="Whether to load just model weights as opposed to training state"
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
