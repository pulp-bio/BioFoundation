#*----------------------------------------------------------------------------*
#* Copyright (C) 2025 ETH Zurich, Switzerland                                 *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  Anna Tegon                                                        *
#* Author:  Thorir Mar Ingolfsson                                             *
#*----------------------------------------------------------------------------*
import os
import logging
import os.path as osp
from logging import Logger
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from util.train_utils import find_last_checkpoint_path
from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
from datetime import datetime
# Check the hostname to set the DATA_PATH accordingly
hostname = os.uname().nodename

os.environ['DATA_PATH'] = "#CHANGEME"
os.environ['CHECKPOINT_DIR'] = '#CHANGEME'

OmegaConf.register_new_resolver("env", lambda key: os.getenv(key))
OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)

logger: Logger = logging.getLogger(__name__)

def train(cfg: DictConfig):
    seed_everything(cfg.seed)

    
    date_format = "%d_%m_%H-%M"  

    # Create your version_name
    version= f"12seqbimamba35_unfrozen_mambaClass{datetime.now().strftime(date_format)}" 
    
    # tensorboard
    tb_logger = TensorBoardLogger(save_dir=osp.expanduser(cfg.io.base_output_path), name=cfg.tag, version=version)

    # DataLoader
    print("===> Loading datasets")
    data_module = hydra.utils.instantiate(cfg.data_module)

    # Pytorch Lightning module
    print("===> Start building model")
    model = hydra.utils.instantiate(cfg.task, cfg)
    print(model)

    # Load pretrained checkpoint
    if cfg.pretrained_checkpoint_path is not None:
        print(f"===> Loading pretrained_checkpoint from {cfg.pretrained_checkpoint_path}")
        model.load_pretrained_checkpoint(cfg.pretrained_checkpoint_path)
    
    else:
       print("No pretrained checkpoint provided. Proceeding without loading.")

    # New Checkpoint dipath  
    checkpoint_dirpath = cfg.io.checkpoint_dirpath
    checkpoint_dirpath = osp.join(checkpoint_dirpath, cfg.tag, version)
    print(f"Checkpoint path: {checkpoint_dirpath}")
    last_ckpt = None
    if cfg.resume:
        last_ckpt = find_last_checkpoint_path(checkpoint_dirpath)
        print(f"last_ckpt_{last_ckpt}")
    print("===> Checkpoint callbacks")
    model_checkpoint = ModelCheckpoint(dirpath=checkpoint_dirpath, **cfg.model_checkpoint)
    model_summary = pl.callbacks.ModelSummary(max_depth=4)
    callbacks = [model_checkpoint, model_summary]

    # Other Pytorch Lightning callbacks
    print("===> Instantiate other callbacks")
    for _, callback in cfg.callbacks.items():
        callbacks.append(hydra.utils.instantiate(callback))

    # Trainer
    print("===> Instantiate trainer")
    if cfg.trainer.strategy == "ddp":
        del cfg.trainer.strategy
        trainer = Trainer(
            **cfg.trainer,
            logger=tb_logger,
            callbacks=callbacks,
            strategy=DDPStrategy(find_unused_parameters=cfg.find_unused_parameters),
        )
    else:
        trainer = Trainer(
            **cfg.trainer,
            logger=tb_logger,
            callbacks=callbacks,
        )

    # Train the model
    if cfg.training:
        print("===> Start training")
        trainer.fit(model, data_module, ckpt_path=last_ckpt)

    best_ckpt = model_checkpoint.best_model_path
   
    print(f"Best checkpoint path: {best_ckpt}")
    print(f"Best model score: {model_checkpoint.best_model_score}")

    if cfg.final_validate:
        print("===> Start validation")
        trainer.validate(model, data_module,ckpt_path=best_ckpt)
    if cfg.final_test:
        print("===> Start testing")
        trainer.test(model, data_module, ckpt_path=last_ckpt)

    if not cfg.training:
        trainer.save_checkpoint(f"{checkpoint_dirpath}/last.ckpt")


@hydra.main(config_path="./config", config_name="defaults", version_base="1.1")
def run(cfg: DictConfig):
    print(f"PyTorch-Lightning Version: {pl.__version__}")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    train(cfg)


if __name__ == "__main__":
    # Ensure environment variables are set before Hydra processes the config
    os.environ['HYDRA_FULL_ERROR'] = '1'
    run()