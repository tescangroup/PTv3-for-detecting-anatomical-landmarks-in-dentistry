import os

import lightning as L
import trimesh
import torch
import sys
sys.path.append("src/lightning_logging")


class LoggingCallback(L.Callback):
    def __init__(self):
        super().__init__()

    def on_validation_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        # return
        # TODO finish test cases logging
        current_step = trainer.global_step
        log_test_every_n_steps = pl_module.logger.logging_cfg.log_test_cases_every_n_steps

        # Log test cases if current_step % log_test_every_n_steps
        if current_step % log_test_every_n_steps != 0:
            return

        test_cases_to_log = list(pl_module.logger.logging_cfg.test_case_ids)
        os.makedirs('test_cases', exist_ok=True)
        data_root_path = trainer.val_dataloaders.dataset.root_path
        stl_root_path = os.path.join(data_root_path, 'STLData')
        for id in test_cases_to_log:
            mesh_path = os.path.join(stl_root_path, f'{id}.obj')
            mesh = trimesh.load(mesh_path)
            indices = torch.arange(0, len(mesh.vertices))

            coords = torch.tensor(mesh.vertices[indices])
            normals = torch.tensor(mesh.vertex_normals[indices])

            input_pcloud = torch.cat((coords, normals), dim=1)

            input_dict = {
                'input_pcloud': input_pcloud,
                'case_ids': torch.tensor([id]),
                'file_paths': [mesh_path],
                'v_subset_indices': indices,
                'labels_heatmaps': None,
                'lms_list': None,
                'transform_matrices': None,
            }

            out_dict = pl_module.forward(input_dict)
            print("COMPUTED OUT DICT")
        pass