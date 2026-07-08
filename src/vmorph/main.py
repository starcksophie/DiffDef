import os
import hydra
from omegaconf import DictConfig
from lightning.pytorch import loggers as pl_loggers
from lightning import Trainer
from data.ukbb_brain import UKBBDataModule

from model.lightning import LightningDLReg
from utils.misc import MyModelCheckpoint

import random
random.seed(7)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:

    # set via CLI hydra.run.dir
    model_dir = os.getcwd()
    # comment in for debugging purposes
    os.environ['WANDB_DISABLED'] = cfg.wandb_disable

    # use only one GPU
    if isinstance(cfg.gpu, int):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu)

    data_module = UKBBDataModule(data_dir=cfg.data.data_dir, train_csv=cfg.data.train_csv, val_csv=cfg.data.val_csv, batch_size=cfg.data.batch_size)
    data_module.setup('fit')

    # lightning model
    model = LightningDLReg(hparams=cfg)

    # configure logger
    # logger = TensorBoardLogger(model_dir, name='log')
    logger = pl_loggers.WandbLogger(save_dir=model_dir, project='VmorphUkbb')

    # model checkpoint callback with ckpt metric logging
    ckpt_callback = MyModelCheckpoint(save_last=True,
                                      dirpath=f'{model_dir}/checkpoints/',
                                      verbose=True
                                      )

    trainer = Trainer(default_root_dir=model_dir,
                      logger=logger,
                      callbacks=[ckpt_callback],
                      # gpus=gpus,
                      accelerator='gpu',
                      devices=1,
                      **cfg.training.trainer
                      )

    # run training
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
