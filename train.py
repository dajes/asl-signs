import logging
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.logger import DummyLogger

import constants
from dataset.data_module import LightData
from module import Module
from state import state
from utils import numerated_folder, seed_everything

seed_everything()

data = LightData(
    os.path.join(constants.DATASET_PATH, 'asl-signs', 'train.csv'), constants.N_COORDS, constants.MAX_LEN,
    constants.BATCH_SIZE, constants.STEPS_PER_EPOCH, constants.EPOCHS,
    constants.WORKERS,
)

working_dir = numerated_folder(os.path.join(constants.MODELS_PATH, 'asl'))
session_name = os.path.basename(working_dir)
module = Module(
    working_dir, constants.EPOCHS, constants.STEPS_PER_EPOCH, constants.LR, data.n_features, constants.N_FEATURES,
    data.n_outputs, constants.MAX_LEN, constants.DROPOUT,
    constants.N_LAYERS, constants.N_HEADS, constants.MLP_RATIO, model_type=constants.ARCHITECTURE,
)
if os.path.exists(constants.CHECKPOINT_PATH):
    module.load_state_dict(torch.load(constants.CHECKPOINT_PATH)['state_dict'])

pl._logger.setLevel(logging.WARNING)
trainer = pl.Trainer(
    auto_lr_find=True,
    enable_checkpointing=True,
    max_epochs=constants.EPOCHS,
    check_val_every_n_epoch=1,
    accelerator='gpu',
    devices=constants.DEVICES,
    logger=WandbLogger(session_name, project='ASL') if constants.WORKERS else DummyLogger(),
    # strategy='ddp_find_unused_parameters_false' if constants.DEVICES > 1 else None,
    precision='bf16',
    accumulate_grad_batches=constants.ACCUMULATE,
    gradient_clip_val=1.,
    callbacks=[
        pl.callbacks.LearningRateMonitor(logging_interval='epoch', log_momentum=True),
        pl.callbacks.ModelCheckpoint(
            working_dir, filename=session_name, monitor='val/acc', mode='max', save_top_k=1, save_last=True,
        )
    ]
)
if constants.TASK == 'find_lr':
    result = trainer.tune(module, data, lr_find_kwargs={'num_training': 100})
    if state['master']:
        result['lr_find'].plot(show=True)
        print(result['lr_find'].suggestion())
elif constants.TASK == 'train':
    trainer.fit(
        module, data,
        ckpt_path=constants.CHECKPOINT_PATH
        if constants.CHECKPOINT_PATH.endswith('.ckpt') and constants.RESUME else None,
    )
elif constants.TASK == 'val':
    trainer.validate(module, data)
else:
    raise ValueError(f'Unknown task: {repr(constants.TASK)}')
