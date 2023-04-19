import gc
import os

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers.logger import DummyLogger

import constants
from dataset.data_module import LightData
from module import Module
from state import state
from utils import numerated_folder, seed_everything

torch.set_float32_matmul_precision('medium')

for seed in range(1):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    seed_everything(seed)

    data = LightData(
        os.path.join(constants.DATASET_PATH, 'asl-signs', 'train.csv'), constants.N_COORDS, constants.MAX_LEN,
        constants.BATCH_SIZE, constants.STEPS_PER_EPOCH, constants.EPOCHS,
        constants.WORKERS, constants.EXTERNAL_DATASETS
    )

    working_dir = numerated_folder(os.path.join(constants.MODELS_PATH, 'asl'))
    session_name = os.path.basename(working_dir)
    module = Module(
        working_dir, constants.EPOCHS, constants.STEPS_PER_EPOCH, constants.LR, data.n_features, constants.N_FEATURES,
        data.n_outputs, constants.MAX_LEN, constants.DROPOUT,
        constants.N_LAYERS, constants.N_HEADS, constants.MLP_RATIO, constants.ARCHITECTURE, constants.CAUSAL_FORESIGHT,
        constants.EMA, constants.LABEL_SMOOTHING, constants.MIXUP, constants.N_COORDS
    )
    if constants.CHECKPOINT_PATH:
        module.load_state_dict(torch.load(constants.CHECKPOINT_PATH)['state_dict'], strict=False)

    if constants.TEACHER_PATH:
        module.teacher = Module.load_from_checkpoint(constants.TEACHER_PATH, 'cpu')

    trainer = pl.Trainer(
        enable_checkpointing=True,
        max_epochs=constants.EPOCHS,
        check_val_every_n_epoch=1,
        accelerator='gpu',
        devices=constants.DEVICES,
        logger=WandbLogger(session_name, project='ASL') if constants.WORKERS else DummyLogger(),
        strategy='ddp_find_unused_parameters_false' if isinstance(constants.DEVICES, list) or constants.DEVICES > 1
        else 'auto',
        precision='bf16',
        accumulate_grad_batches=constants.ACCUMULATE,
        gradient_clip_val=1.,
        callbacks=[
                      pl.callbacks.LearningRateMonitor(logging_interval='epoch', log_momentum=True),
                      pl.callbacks.ModelCheckpoint(
                          working_dir, filename=session_name, monitor='val/acc', mode='max', save_top_k=1,
                          save_last=True,
                      )
                  ] + ([pl.callbacks.QuantizationAwareTraining('qnnpack')] if constants.QAT else []),
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

    wandb.finish()
