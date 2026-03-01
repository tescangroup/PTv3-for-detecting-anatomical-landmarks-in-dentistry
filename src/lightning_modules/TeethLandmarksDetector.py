from typing import Dict, List, Tuple, Union

import torch
import omegaconf
import lightning as L
import torch.nn.functional as F
import numpy as np
import trimesh

from src.postprocessing.nms import postprocess_and_detect
from src.lightning_logging import metrics


class TeethLandmarksDetector(L.LightningModule):
    def __init__(self,
                 encoder: torch.nn.Module,
                 decoder: torch.nn.Module,
                 hyperparams: omegaconf.DictConfig
                 ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.postprocess = postprocess_and_detect

        self.hyperparams = hyperparams
        self._hparams = {'lr': self.hyperparams.training.lr}

        self.calibration_mode = False
        self.calibrated_thresholds = None
        self.calibrated_nms_steps = None

        self.nms_threshold_vals = np.linspace(0.1, self.hyperparams.training.nms_threshold * 4, 8)
        self.nms_iterations_vals = np.round(np.linspace(6, self.hyperparams.training.nms_iterations * 4, 8)).astype(int)
        self.accumulated_metrics = {}
        for c in range(1, 7):
            lm = f'{metrics.lm_to_class_idx[c]}'
            self.accumulated_metrics[lm] = {}
            for t in self.nms_threshold_vals:
                for i in self.nms_iterations_vals:
                    self.accumulated_metrics[lm][f"t_{t}-i_{i}"] = []

    def start_calibration(self):
        print("====== DETECTOR CALIBRATION MODE ON ======")
        self.calibration_mode = True

    def finish_calibration(self):
        print("====== DETECTOR CALIBRATION MODE OFF ======")
        self.calibration_mode = False
        self.calibrated_thresholds = {}
        self.calibrated_nms_steps = {}
        calibratation_scores = {}
        for c in range(1, 7):
            lm = f'{metrics.lm_to_class_idx[c]}'
            calibratation_scores[lm] = {}
            for t in self.nms_threshold_vals:
                for i in self.nms_iterations_vals:
                    calibratation_scores[lm][f"t_{t}-i_{i}"] = np.mean(self.accumulated_metrics[lm][f"t_{t}-i_{i}"])
            best_configuration = max(calibratation_scores[lm], key=calibratation_scores[lm].get)
            print(f"Best configuration for {lm}: {best_configuration}")
            self.calibrated_thresholds[lm] = float(best_configuration.split("-")[0].split("_")[1])
            self.calibrated_nms_steps[lm] = int(best_configuration.split("-")[1].split("_")[1])

    def log_calibration_outputs(self):
        self.logger.log_calibration(
            self.accumulated_metrics,
            self.nms_threshold_vals,
            self.nms_iterations_vals
        )

    def infer_mesh(self, mesh_path: str, min_pts, calibrated_params) -> np.ndarray:

        self.eval()

        # set no grad
        with torch.no_grad():
            mesh = trimesh.load(mesh_path)
            if len(mesh.vertices) < min_pts:
                mesh = mesh.subdivide()
            coords = torch.tensor(mesh.vertices)
            normals = torch.tensor(mesh.vertex_normals)
            v_subset_indices = torch.arange(0, len(coords))

            input_pcloud = torch.cat([coords, normals], dim=1).to(self.device).unsqueeze(0)
            v_subset_indices = v_subset_indices.to(self.device).unsqueeze(0)
            matrix = torch.eye(4).to(self.device)

            input_dict = {
                'input_pcloud': input_pcloud,
                'v_subset_indices': v_subset_indices,
                'labels_heatmaps': None,
                'lms_list': None,
                'case_ids': None,
                'file_paths': [mesh_path],
                'transform_matrices': [matrix]
            }

            enc_out = self.encoder(input_dict)
            torch.cuda.empty_cache()
            dec_out = self.decoder(enc_out)
            torch.cuda.empty_cache()
            batch_postprocessed = self.postprocess(  # Adds 'accumulated_dists' and 'detections_list'
                [dec_out],
                nms_iterations=calibrated_params["calibrated_nms_steps"],
                threshold=calibrated_params["calibrated_thresholds"]
            )
            torch.cuda.empty_cache()
            return batch_postprocessed

    def training_step(self,
                      batch_dict: Dict[str, Union[torch.Tensor,
                      List[Tuple[Tuple[float, float, float], int]], List[str]]],
                      batch_idx: int
                      ) -> dict:
        batch_dict = self.encoder(batch_dict)  # Adds key 'features' with shape (B x N x 32)
        batch_dict = self.decoder(batch_dict)  # Adds key 'dist_maps' with shape (B x N x 6)

        loss = F.mse_loss(batch_dict['dist_maps'],
                          batch_dict['labels_heatmaps'])  # gt (batch['labels_heatmaps']): (B x N x 6)

        self.log_dict({
            'TRN/Loss': loss,
        }, prog_bar=True, on_step=True, on_epoch=False, batch_size=batch_dict['features'].shape[0])

        return {
            'loss': loss,
        }

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hyperparams.training.lr)

        return optimizer

    def evaluate(self,
                 batch_dict: Dict[str, Union[torch.Tensor,
                 List[Tuple[Tuple[float, float, float], int]], List[str]]],
                 stage=None,
                 ) -> torch.Tensor:

        if stage == "VAL":
            ids_to_render = self.logger.logging_cfg.validation_log_html_ids
        elif stage == "TEST":
            ids_to_render = self.logger.logging_cfg.test_case_ids

        if stage == "TEST" and batch_dict['case_ids'][0] not in ids_to_render and not self.calibration_mode:
            return None

        batch_dict = self.encoder(batch_dict)  # Adds key 'features' with shape (B x N x 32)
        batch_dict = self.decoder(batch_dict)  # Adds key 'dist_maps' with shape (B x N x 6)

        nms_steps = self.calibrated_nms_steps if self.calibrated_nms_steps is not None else self.hyperparams.training.nms_iterations
        nms_threshold = self.calibrated_thresholds if self.calibrated_thresholds is not None else self.hyperparams.training.nms_threshold

        batch_postprocessed = self.postprocess(  # Adds 'accumulated_dists' and 'detections_list'
            [batch_dict],
            nms_steps,
            nms_threshold
        )

        if self.calibration_mode:
            for t in self.nms_threshold_vals:
                for i in self.nms_iterations_vals:

                    batch_postprocessed = self.postprocess(  # Adds 'accumulated_dists' and 'detections_list'
                        [batch_dict],
                        i,
                        t
                    )
                    result_dict = metrics.get_ar_ap(batch_postprocessed, np.arange(0, 2, 0.05))
                    for c in range(1, 7):
                        lm = f'{metrics.lm_to_class_idx[c]}'
                        self.accumulated_metrics[lm][f"t_{t}-i_{i}"].append(
                            result_dict["APs"][lm] * result_dict["ARs"][lm])

        if not "MORPH" in batch_postprocessed['file_paths'][0] and not self.calibration_mode:
            if batch_postprocessed['case_ids'][0] in ids_to_render:
                # visualize only selected validation cases
                predictions_local_path = metrics.visualize_dict(batch_postprocessed)
                features_local_path = metrics.visualize_features(batch_postprocessed)
                self.logger.save_sample_scene(
                    iteration=self.trainer.global_step,
                    name=f"{stage}_{batch_postprocessed['case_ids'][0]}",
                    scene_name=f"3D detections preview",
                    local_path=predictions_local_path
                )
                self.logger.save_sample_scene(
                    iteration=self.trainer.global_step,
                    name=f"{stage}_{batch_postprocessed['case_ids'][0]}",
                    scene_name=f"3D features preview",
                    local_path=features_local_path
                )

        if stage == "VAL" and not self.calibration_mode:
            result_dict = metrics.get_ar_ap(batch_postprocessed, np.arange(0, 2, 0.05))

            loss = F.mse_loss(batch_dict['dist_maps'],
                              batch_dict['labels_heatmaps'])  # gt (batch['labels_heatmaps']): (B x point_num x 6)

            if stage:
                dict_to_log = {
                    f'{stage}/Loss': loss,
                    f'{stage}/mAP': result_dict["mAP"],
                    f'{stage}/mAR': result_dict["mAR"]
                }
                for c in range(1, 7):
                    dict_to_log[f'{stage}/AP_{metrics.lm_to_class_idx[c]}'] = result_dict["APs"][
                        metrics.lm_to_class_idx[c]]
                    dict_to_log[f'{stage}/AR_{metrics.lm_to_class_idx[c]}'] = result_dict["ARs"][
                        metrics.lm_to_class_idx[c]]
                self.log_dict(dict_to_log, prog_bar=True, batch_size=batch_dict['features'].shape[0])
        elif stage == "TEST":
            loss = None

        return loss

    def test_step(self,
                  batch_dict: Dict[str, Union[torch.Tensor,
                  List[Tuple[Tuple[float, float, float], int]], List[str]]],
                  batch_idx: int
                  ) -> torch.Tensor:
        loss_tst = self.evaluate(batch_dict, "TEST")

        return loss_tst

    def validation_step(self,
                        batch_dict: Dict[str, Union[torch.Tensor,
                        List[Tuple[Tuple[float, float, float], int]], List[str]]],
                        batch_idx: int
                        ) -> torch.Tensor:
        loss_val = self.evaluate(batch_dict, "VAL")

        return loss_val
