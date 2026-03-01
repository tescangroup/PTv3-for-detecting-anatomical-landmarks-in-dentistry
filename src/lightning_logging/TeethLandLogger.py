import os
from argparse import Namespace
from typing import Union, Dict, Any

import trimesh
import matplotlib.pyplot as plt
from clearml import Task
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only
import torch
import numpy as np
import pyvista
import omegaconf


class TeethLandLogger(Logger):

    def __init__(self,
                 clearml_task: Task,
                 cfg: omegaconf.DictConfig,
                 ):
        self.task = clearml_task
        self.cfg = cfg
        self.logging_cfg = cfg.training.logging

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace], *args: Any, **kwargs: Any) -> None:
        pass

    @property
    def name(self):
        return "TeethLandLogger"

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_metrics(self, metrics, step):
        for k, v in metrics.items():
            if k == 'epoch':
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()

            self.task.get_logger().report_scalar(title=k, series=k, value=v, iteration=step)

    @rank_zero_only
    def save(self):
        pass

    @rank_zero_only
    def finalize(self, status):
        self.task.flush()

    def save_sample_scene(
            self,
            iteration: int,
            name: str,
            scene_name: str,
            local_path: str,
    ) -> None:

        self.task.get_logger().report_media(
            title=name,
            series=scene_name,
            iteration=iteration,
            local_path=local_path
        )

    def save_batch_scenes(
            self,
            iteration: int,
            name: str,
            input_pcs: np.ndarray,
            labels: Any,
            output: Any,
            mesh_ids: tuple,
            mesh_paths: tuple
    ) -> None:
        # if input_pcs is not np.array convert
        if not isinstance(input_pcs, np.ndarray):
            input_pcs = np.array(input_pcs)
        for i, (pc, mesh_id, mesh_path) in enumerate(zip(input_pcs, mesh_ids, mesh_paths)):
            self.save_sample_scene(iteration, name, mesh_id, pc, None, None, mesh_path)

    def log_calibration(
            self,
            calibration_dict: dict,
            thresholds: list,
            steps: list,
    ) -> None:
        lms = calibration_dict.keys()
        for lm in lms:
            canvas = np.zeros((len(thresholds), len(steps)))
            for i, t in enumerate(thresholds):
                for j, s in enumerate(steps):
                    canvas[j, i] = np.mean(calibration_dict[lm][f"t_{t}-i_{s}"])

            ca = plt.imshow(canvas, cmap='hot', interpolation='nearest', )
            plt.xticks(np.arange(len(thresholds)), np.round(thresholds, 2), rotation=30)
            plt.xlabel("Detection threshold")
            plt.yticks(np.arange(len(steps)), steps)
            plt.ylabel("NMS steps")
            plt.colorbar(ca)
            self.task.get_logger().report_matplotlib_figure(
                title=f"Calibration for {lm}",
                series="Calibration",
                iteration=0,
                figure=plt,
                report_interactive=True
                )
            plt.close()

