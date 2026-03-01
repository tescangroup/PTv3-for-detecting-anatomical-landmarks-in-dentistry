from copy import deepcopy

import numpy as np
import torch
import trimesh
import torch_geometric as tg
from typing import List, Dict

class_to_lm_idx = {'Mesial': 1, 'Distal': 2, 'Cusp': 3, 'InnerPoint': 4, 'OuterPoint': 5, 'FacialPoint': 6}
lm_to_class_idx = {v: k for k, v in class_to_lm_idx.items()}

class NMS(tg.nn.MessagePassing):
    def __init__(self):
        super().__init__(aggr='max')

    def forward(self, x, edge_index):
        edge_index, _ = tg.utils.add_self_loops(edge_index, num_nodes=x.size(0))
        out = -self.propagate(edge_index, x=-x)

        return out

    def message(self, x_j):
        return x_j


def accumulate_distances(
        dist_maps: List[torch.Tensor],
        v_subset_indices: List[torch.Tensor],
        mesh: trimesh.Trimesh
) -> torch.Tensor:

    accumulated_dist_values = torch.zeros(len(mesh.vertices), dist_maps[0].shape[-1], device="cuda")
    accumulated_dist_num = torch.zeros(len(mesh.vertices), dist_maps[0].shape[-1], device="cuda")

    for c in range(dist_maps[0].shape[-1]):
        for ids, dist_map in zip(v_subset_indices, dist_maps):
            accumulated_dist_values[..., c][ids] += dist_map[..., c]
            accumulated_dist_num[..., c][ids] += 1

    accumulated_dist_values /= accumulated_dist_num
    accumulated_dist_values[torch.isnan(accumulated_dist_values)] = 9999

    return accumulated_dist_values.cpu()


def non_maxima_suppression(
        dists: torch.Tensor,
        mesh: trimesh.Trimesh,
        iterations: int,
        threshold: float
) -> np.ndarray:

    ftm = tg.transforms.FaceToEdge(remove_faces=False)
    nms = NMS()

    mesh_tg = tg.utils.from_trimesh(mesh)
    mesh_tg = ftm(mesh_tg)

    dists_out = dists.clone()
    for _ in range(iterations):
        dists_out = nms(dists_out.view(-1, 1), mesh_tg.edge_index)[..., 0]

    return np.where((dists == dists_out) & (dists != 9999) & (dists < threshold))[0]


def postprocess_and_detect(
        inference_output_list: List[Dict],
        nms_iterations: int | dict,
        threshold: float | dict
) -> Dict:
    """
    nms_interations and threshold can either be scalars or a dict with string keys for each lm class
    """

    for sample_dict in inference_output_list:
        assert "file_paths" in sample_dict.keys(), "mesh file paths not present in sample dict"
        assert "dist_maps" in sample_dict, "dist_maps not present in sample dict"
        assert len(sample_dict["dist_maps"].shape) == 3, "dist maps need to have 3 dimensions (B, num_points, num_classes)"
        assert "v_subset_indices" in sample_dict, "v_subset_indices not present in sample dict"
        assert len(sample_dict["v_subset_indices"].shape) == 2, "v_subset_indices needs to have 2 dimensions (B, num_points)"

    dists = []
    batch_detections = []
    batch_scores = []

    postprocess_output = inference_output_list[0]

    for b in range(inference_output_list[0]["v_subset_indices"].shape[0]):
        sample_mesh = trimesh.load(inference_output_list[0]["file_paths"][b])
        sample_mesh.apply_transform(inference_output_list[0]["transform_matrices"][b].cpu())
        accumulated_dists = accumulate_distances(
            [x["dist_maps"][b] for x in inference_output_list],
            [x["v_subset_indices"][b] for x in inference_output_list],
            sample_mesh
        )

        detections = []
        scores = []
        for c in range(1, 7):
            if isinstance(nms_iterations, dict):
                c_nms_iterations = nms_iterations[lm_to_class_idx[c]]
            else:
                c_nms_iterations = nms_iterations
            if isinstance(threshold, dict):
                c_threshold = threshold[lm_to_class_idx[c]]
            else:
                c_threshold = threshold
            minima_position = non_maxima_suppression(
                accumulated_dists[..., c-1],
                sample_mesh,
                iterations=c_nms_iterations,
                threshold=c_threshold
            )
            class_scores = []
            for i in minima_position:
                detections.append((sample_mesh.vertices[i], c))
                class_scores.append(accumulated_dists[i, c-1].item())

            class_scores = 1/np.array(class_scores)
            class_scores = class_scores / np.max(class_scores)
            scores.extend(list(class_scores))

        dists.append(accumulated_dists)
        batch_detections.append(detections)
        batch_scores.append(scores)




    postprocess_output["detections_scores"] = batch_scores
    postprocess_output["dist_maps_accumulated"] = dists
    postprocess_output["detections_list"] = batch_detections

    return postprocess_output


if __name__ == "__main__":
    import sys
    sys.path.append("./src/logging")
    # from metrics import visualize_dict
    # produce dummy mesh
    sphere = trimesh.creation.icosphere(4, 1)
    sphere.export("./tmp.stl")
    sphere = trimesh.load("./tmp.stl")

    # produce several samples with distances to landmarks
    inference_output_list = []

    src_idx = [0, 100, 30, 699, 1700, 2000]
    cs = [1, 2, 3, 4, 5, 6]

    B = 4

    for i in range(1):
        sphere_tg = tg.utils.from_trimesh(sphere)
        dists = tg.utils.geodesic_distance(sphere_tg.pos, sphere_tg.face, src=torch.tensor(src_idx))
        dists = torch.min(dists, dim=0).values
        idx = np.random.choice(sphere_tg.pos.shape[0], sphere_tg.pos.shape[0], replace=False)
        dists_sample = dists[idx]
        verts_sample = sphere_tg.pos[idx]
        dists_sample = dists_sample + torch.rand(dists_sample.shape) * 0.1

        lms_list = []
        for src_id, c in zip(src_idx, cs):
            lms_list.append((sphere.vertices[src_id], c))

        verts = verts_sample.reshape(1, -1, 3)
        verts = torch.tile(verts, (B, 1, 1))

        v_subset_indices = idx.reshape(1, -1)
        v_subset_indices = np.tile(v_subset_indices, (B, 1))

        dist_map = dists_sample.reshape(1, -1, 1)
        dist_maps = torch.tile(dist_map, (1, 1, 6))
        dist_maps[:, :, 1:] = 1
        dist_maps = torch.tile(dist_maps, (B, 1, 1))

        inference_output_list.append({
            "file_paths": B*[f"./tmp.stl"],
            "input": verts,
            "dist_maps": dist_maps,
            "v_subset_indices": v_subset_indices,
            "lms_list": B*[lms_list]
        })

    # call NMS
    out_dict = postprocess_and_detect(inference_output_list)

    # visualize
    # visualize_dict(out_dict)
