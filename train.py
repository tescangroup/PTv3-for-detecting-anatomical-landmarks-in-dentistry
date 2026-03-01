"""
    Project: 3DTeethLand24

    Author(s): Tibor Kubik, Tomas Mojzis
    Email(s): tibor.kubik@tescan.com, tomas.mojzis@tescan.com

    Brief:
    This script sets up and trains a model for detecting landmarks in 3D teeth data using PyTorch Lightning.
    The method is based on feature extraction via PTv3 with lightweight decoder that predicts 6 geodesic maps
    for individual landmark classes.
    It includes functions to build datasets, a PyTorch Lightning module, and logging.
    The main function orchestrates the training process, utilizing the ClearML for experiment tracking.
"""

import os
import pickle

os.environ["TEETHLAND_MODE"] = "WITH_VISTA"

import hydra
import torch
import omegaconf
import lightning as L

from clearml import Task
from datetime import datetime
from socket import gethostname
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import LearningRateFinder, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from src.lightning_modules.TeethLandmarksDetector import TeethLandmarksDetector

from src.lightning_logging.TeethLandLogger import TeethLandLogger
from src.lightning_logging.callbacks import LoggingCallback
from src.data_proc.TLDataset import TLDataset

from src.modeling.build_encoder import build_encoder
from src.modeling.build_decoder import build_decoder

torch.set_float32_matmul_precision('medium')

DataLoaderTrain = DataLoader
DataLoaderVal = DataLoader


def build_datasets(cfg: omegaconf.DictConfig) -> tuple[DataLoaderTrain, DataLoaderVal]:
    """
    Builds and returns the training and validation data loaders based on the provided configuration.

    Args:
        cfg (omegaconf.DictConfig): Configuration object containing the paths and training parameters.

    Returns:
        tuple[DataLoaderTrain, DataLoaderVal]: A tuple containing the training data loader and
        the validation data loader.

    The function performs the following steps:
    1. Initializes the training dataset `dataset_trn` using the specified JSON path and the number of sampled points.
    2. Preprocesses the training dataset offline and saves the preprocessed data to the specified path.
    3. Creates a data loader `loader_trn` for the training dataset with the specified batch size,
       number of workers, and other parameters.
    4. Initializes the validation dataset `dataset_val` using the same JSON path and number of sampled points.
    5. Preprocesses the validation dataset offline and saves the preprocessed data to the specified path.
    6. Creates a data loader `loader_val` for the validation dataset with the specified batch size, number of workers,
       and other parameters.
    7. Returns a tuple containing the training and validation data loaders.

    Notes:
        - The `persistent_workers` parameter helps in keeping data loading workers alive, which can be beneficial for
          performance when using multiple workers.
        - The `drop_last` parameter ensures that the last incomplete batch is dropped if the number of samples
          is not divisible by the batch size.
    """

    dataset_trn = TLDataset(cfg.paths.dataset_json_path,
                            tmp_data_path=cfg.paths.preprocessed_data_path,
                            sampled_points_num=cfg.training.sampled_points_num,
                            num_morphed_meshes=cfg.training.dataset.offline_morphed_meshes_count,
                            slice_data=slice(cfg.training.dataset.training_slice_start,
                                             cfg.training.dataset.training_slice_end)
                            )
    # dataset_trn.offline_preprocess(cfg.paths.preprocessed_data_path)
    loader_trn = DataLoader(dataset=dataset_trn,
                            shuffle=True,
                            num_workers=cfg.training.num_workers,
                            persistent_workers=cfg.training.persistent_workers,
                            pin_memory=True,
                            batch_size=cfg.training.batch_size,
                            drop_last=True,
                            collate_fn=dataset_trn.collate_fn)

    dataset_val = TLDataset(cfg.paths.dataset_json_path,
                            tmp_data_path=cfg.paths.preprocessed_data_path,
                            sampled_points_num=None,
                            num_morphed_meshes=0,
                            slice_data=slice(cfg.training.dataset.validation_slice_start,
                                             cfg.training.dataset.validation_slice_end)
                            )
    # dataset_trn.offline_preprocess(cfg.paths.preprocessed_data_path)
    loader_val = DataLoader(dataset=dataset_val,
                            shuffle=False,
                            num_workers=cfg.training.num_workers,
                            pin_memory=True,
                            persistent_workers=cfg.training.persistent_workers,
                            batch_size=1,
                            drop_last=True,
                            collate_fn=dataset_val.collate_fn)

    dataset_test = TLDataset(cfg.paths.dataset_json_path,
                             tmp_data_path=None,
                             sampled_points_num=None,
                             num_morphed_meshes=0,
                             slice_data=slice(0, None),
                             only_with_landmarks=False
                             )
    # dataset_trn.offline_preprocess(cfg.paths.preprocessed_data_path)
    loader_test = DataLoader(dataset=dataset_test,
                             shuffle=False,
                             num_workers=cfg.training.num_workers,
                             pin_memory=True,
                             persistent_workers=cfg.training.persistent_workers,
                             batch_size=1,
                             drop_last=True,
                             collate_fn=dataset_val.collate_fn)

    return loader_trn, loader_val, loader_test


def build_lightning_module(cfg: omegaconf.DictConfig) -> L.LightningModule:
    """
    Builds and returns a PyTorch Lightning module based on the provided configuration.

    Args:
        cfg (omegaconf.DictConfig): Configuration object containing the model parameters and training settings.

    Returns:
        L.LightningModule: An instance of the TeethLandmarksDetector Lightning module.

    The function performs the following steps:
    1. Asserts that the specified model name in the configuration is one of the implemented models.
    2. Builds the encoder model using the specified model name, input channels, output channels, and patch size.
    2. Builds the decoder model.
    3. Constructs the TeethLandmarksDetector model using the built encoder, decoder, and the configuration.
    4. Returns the constructed TeethLandmarksDetector model.

    Raises:
        AssertionError: If the specified model name is not in ['ptv3_small', 'ptv3_medium', 'ptv3_large', 'conv'].
    """

    assert cfg.training.encoder.model_name in ['ptv3_small', 'ptv3_medium', 'ptv3_large', 'conv']
    assert cfg.training.decoder.model_name in ['mlp_small', 'mlp_medium', 'mlp_large']

    encoder = build_encoder(cfg.training.encoder.model_name,
                            input_channels=cfg.training.encoder.input_channels,
                            output_channels=cfg.training.encoder.output_channels,
                            patch_size=cfg.training.encoder.patch_size,
                            )

    decoder = build_decoder(cfg.training.decoder.model_name,
                            input_channels=cfg.training.encoder.output_channels,  # channels shared with encoder output
                            output_channels=cfg.training.decoder.output_channels,
                            hidden_channels=cfg.training.decoder.hidden_channels,
                            )

    model = TeethLandmarksDetector(encoder, decoder, cfg)

    return model


def build_logger(cfg: omegaconf.DictConfig,
                 task_name: str
                 ) -> TeethLandLogger:
    """
    Initializes and returns a TeethLandLogger based on the provided configuration.

    Args:
        cfg (omegaconf.DictConfig): Configuration object containing the training settings.
        task_name (str): The name of the task to be initialized.

    Returns:
        TeethLandLogger: An instance of TeethLandLogger associated with the initialized task.

    The function performs the following steps:
    1. Initializes a new ClearML task using the provided task name or a debug task name if debugging is enabled.
    2. Specifies the project name as "3DTeethLand".
    3. Sets `reuse_last_task_id` to `True` if debugging is enabled, otherwise sets it to `False`.
    4. Creates an instance of TeethLandLogger with the initialized task.
    5. Returns the created logger instance.

    Notes:
        - The `Task.init` function initializes a new task for tracking experiments and training runs.
        - The `TeethLandLogger` is assumed to be a custom logger for logging experiment details.
        - The `cfg.training.debug` flag determines whether the task is a debug run and whether to reuse the last task ID.
    """
    task = Task.init(
        task_name=task_name if not cfg.training.debug else "debug_run",
        project_name="3DTeethLand",
        reuse_last_task_id=True if cfg.training.debug else False
    )
    logger = TeethLandLogger(task, cfg)

    return logger


def build_callbacks_list(cfg: omegaconf.DictConfig, task_name: str) -> list[L.Callback]:
    """
    Builds and returns a list of callbacks based on the provided configuration.

    Args:
        task_name (str): The name of the task to be initialized.
        cfg (omegaconf.DictConfig): Configuration object containing the training settings.

    Returns:
        list: A list of callback instances to be used during training.

    The function performs the following steps:
    1. Initializes an empty list to hold the callbacks.
    2. Creates an instance of `LoggingCallback` using the sample IDs specified in
       the configuration and adds it to the list.
    3. If early stopping is enabled in the configuration, creates an instance of `EarlyStopping` with
       the specified parameters and adds it to the list.
    4. Returns the list of callbacks.

    Notes:
        - The `LoggingCallback` is assumed to be a custom callback for logging validation results.
        - The `EarlyStopping` callback monitors the validation loss and stops training early if there is no improvement.
    """
    callbacks = list()
    val_log_callback = LoggingCallback()
    callbacks.append(val_log_callback)
    if cfg.training.early_stopping:
        early_stop_callback = EarlyStopping(monitor="VAL/Loss", min_delta=0.00,
                                            patience=cfg.training.early_stopping_patience, verbose=True, mode="max")
        callbacks.append(early_stop_callback)

    if cfg.training.learning_rate_finder:
        lr_finder = LearningRateFinder()  # In base setting, tries to find an optimal lr at the beginning of training.
        callbacks.append(lr_finder)

    model_checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join('checkpoints', task_name),
        monitor="VAL/Loss",
        mode="min",
        save_top_k=5,
        filename="step-{step:07d}-val_loss-{VAL/Loss:.4f}",
        every_n_epochs=None,
        every_n_train_steps=cfg.training.checkpoint_every_n_train_steps,
        auto_insert_metric_name=False,
    )
    callbacks.append(model_checkpoint_callback)

    return callbacks


@hydra.main(config_path='config', config_name='config', version_base=None)
def main(cfg: omegaconf.DictConfig) -> None:
    now = datetime.now()
    task_name = (f'{now.strftime("%b")}{now.strftime("%d")}_{now.strftime("%H")}'
                 f'-{now.strftime("%M")}-{now.strftime("%S")}_{gethostname()}')

    logger = build_logger(cfg, task_name)  # Prepares logger of scalars and 3D scenes + initializes ClearML logging sesh
    loader_trn, loader_val, loader_test = build_datasets(cfg)  # Prepares datasets and dataloaders for training and validation
    model = build_lightning_module(cfg)  # Builds lightning module consisting of encoder, decoder, optimization details
    callbacks = build_callbacks_list(cfg, task_name)  # Creates callback list of actions used during training
    ckpt_callback = callbacks[-1]

    trainer = L.Trainer(accelerator=cfg.training.device,
                        max_steps=cfg.training.max_steps,
                        precision=cfg.training.precision,
                        log_every_n_steps=cfg.training.log_every_n_steps,
                        default_root_dir=os.path.join('checkpoints', task_name),
                        check_val_every_n_epoch=None,
                        val_check_interval=cfg.training.val_check_interval,
                        fast_dev_run=cfg.training.fast_dev_run_num_batches,
                        max_epochs=-1,
                        callbacks=callbacks,
                        logger=logger,
                        num_sanity_val_steps=0)

    trainer.fit(model=model,
                train_dataloaders=loader_trn,
                val_dataloaders=loader_val,
                ckpt_path=cfg.paths.pretrained_model_path)  # Set the value in yaml to null to train from scratch.

    logger.task.upload_artifact(name="last_model_checkpoint", artifact_object=ckpt_callback.last_model_path)

    # find optimal threshold and nms steps using val dataset
    model.start_calibration()
    trainer.test(model=model, dataloaders=loader_val)
    model.finish_calibration()
    model.log_calibration_outputs()

    # save the calibrated parameters as pickle and push to cml
    os.makedirs(ckpt_callback.dirpath, exist_ok=True)
    with open(os.path.join(ckpt_callback.dirpath, "calibrated_params.pkl"), 'wb') as f:
        pickle.dump({
            "calibrated_thresholds": model.calibrated_thresholds,
            "calibrated_nms_steps": model.calibrated_nms_steps
        },
            f
        )
    logger.task.upload_artifact(name="calibrated parameters",
                                artifact_object=os.path.join(ckpt_callback.dirpath, "calibrated_params.pkl"))

    # finally test with calibrated parameters
    trainer.test(model=model, dataloaders=loader_test)


if __name__ == '__main__':
    main()
