import os
import sys
import wandb
import argparse

from fire import Fire
from loguru import logger

sys.path.append(os.path.dirname(os.path.abspath('./src')))
from src.__init__ import *

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def train(config_name=None, **kwargs):
    
    def _train(cfg: DictConfig):
        from src.lightning_model import COCO_EfficientDet, Laion400m_EfficientDet, VisGenome_EfficientDet, VisGenome_FuseDet

        from src.dataset.train_dataset import (
            COCO_Detection,
            Laion400M,
            VisualGenome,
            VisualGenomeFuseDet,
            refCOCO,
        )
        from src.dataset.val_dataset import Validate_Detection
        from src.dataset.bbox_augmentor import default_augmentor, bbox_safe_augmentor, eval_augmentor
        from src.dataset.utils import make_mini_batch
        from torch.utils.data import DataLoader

        from src.utils.config_trainer import Config_Trainer
        from src.utils.wandb_logger import Another_WandbLogger

        pl_modules = {
            "COCO_EfficientDet": COCO_EfficientDet,
            "Laion400m_EfficientDet": Laion400m_EfficientDet,
            "VisGenome_EfficientDet": VisGenome_EfficientDet,
            "VisGenome_FuseDet": VisGenome_FuseDet,
            "refCoco_FuseDet": VisGenome_FuseDet,
        }

        use_background_class = cfg.use_background_class if 'use_background_class' in cfg else True
        freeze_backbone = cfg.freeze_backbone if 'freeze_backbone' in cfg else False
        logger.info(f"freeze_backbone: {freeze_backbone}")
        logger.info(f"use_background_class: {use_background_class}")
        logger.info(f"cfg.trainer.pl_module: {cfg.trainer.Task.pl_module}")
        
        # lightning model
        pl_module = pl_modules[cfg.trainer.Task.pl_module]
        pl_model = pl_module(
            **cfg.model.model,
            **cfg.model.loss,
            **cfg.model.nms,
            **cfg.model.optimizer,
            # val_annFile=cfg.dataset.val.annFile, 
            background_class=use_background_class,
            freeze_backbone=freeze_backbone)
        
        # augmentor
        augmentor = bbox_safe_augmentor(pl_model.model.img_size)
        eval_transform = eval_augmentor(pl_model.model.img_size)

        # dataset and dataloader
        pl_data = {
            "COCO_EfficientDet": (COCO_Detection, Validate_Detection),
            "Laion400m_EfficientDet": (Laion400M, Laion400M),
            "VisGenome_EfficientDet": (VisualGenome, VisualGenome),
            "VisGenome_FuseDet": (VisualGenomeFuseDet, VisualGenomeFuseDet),
            "refCoco_FuseDet": (refCOCO, refCOCO),
        }
        train_set = pl_data[cfg.trainer.Task.pl_module][0](
            **cfg.dataset.train,
            bbox_augmentor=augmentor,
            background_class=use_background_class,
            split='train',
            offline_embed=False,
        )
        val_set = pl_data[cfg.trainer.Task.pl_module][1](
            **cfg.dataset.val,
            dataset_stat=cfg.dataset.dataset_stat,
            img_size=pl_model.model.img_size,
            bbox_augmentor=eval_transform,
            split='val',
            offline_embed=False,
        )

        train_loader = DataLoader(
            train_set, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
            collate_fn=train_set.collect_fn, num_workers=cfg.num_workers,
            multiprocessing_context='spawn' if cfg.num_workers > 0 else None
        )
        val_loader = DataLoader(
            val_set, batch_size=cfg.batch_size, drop_last=True, 
            num_workers=cfg.num_workers,
            multiprocessing_context='spawn' if cfg.num_workers > 0 else None
        )

        # trainer
        cfg_trainer = Config_Trainer(cfg.trainer)()

        if 'load_ckpt_weight' in cfg:
            logger.warning(f"load ckpt weight only: {cfg.load_ckpt_weight}")
            # ckpt = torch.load(cfg.load_ckpt_weight)
            # pl_model.load_state_dict(ckpt['state_dict'], strict=False)
            pl_model.load_finetune_checkpoint(cfg.load_ckpt_weight)

        with logger.catch(reraise=True):
            if 'overfit_batches' not in cfg.trainer.Trainer:
                wb_logger = Another_WandbLogger(**cfg.log)
                trainer = pl.Trainer(**cfg_trainer, logger=wb_logger, num_sanity_val_steps=1)

                wb_logger.watch(pl_model)
                # run training
                try:
                    if 'ckpt_path' in cfg:
                        trainer.fit(pl_model, train_loader, val_loader, ckpt_path=cfg.ckpt_path)
                    else:
                        trainer.fit(pl_model, train_loader, val_loader)
                except KeyboardInterrupt:
                    trainer.save_checkpoint("interrupt.ckpt")
            else:
                trainer = pl.Trainer(**cfg_trainer, num_sanity_val_steps=1, check_val_every_n_epoch=10)
                trainer.fit(pl_model, train_loader, val_loader)

        wandb.finish()
    
    NODE_RANK = os.environ['NODE_RANK'] if 'NODE_RANK' in os.environ else 0
    logger.info(f"args:  {config_name} + {kwargs}, NODE_RANK: {NODE_RANK}")

    if NODE_RANK == 0:
        _train = hydra.main(version_base=None, config_path='./config/', config_name=config_name)(_train)
        _train()
    else:
        from hydra import compose, initialize
        from omegaconf import OmegaConf

        initialize(config_path='./config/', job_name=f"ddp_{NODE_RANK}")
        cfg = compose(config_name=config_name, overrides=[])
        # print(OmegaConf.to_yaml(cfg))
        _train(cfg)


if __name__ == "__main__":
    Fire(train)

