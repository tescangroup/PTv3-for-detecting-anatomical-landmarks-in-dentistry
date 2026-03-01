from typing import List, Tuple, Dict
import os

import numpy as np
import pyvista
import trimesh
from sklearn.decomposition import PCA
import sklearn.preprocessing as skp

if __name__ == "__main__":
    os.environ["TEETHLAND_MODE"] = "WITH_VISTA"

# To suppress the warning about the missing DISPLAY environment variable
import sys
if sys.platform.startswith('linux'):
    try:
        display = os.environ['DISPLAY']
    except KeyError:
        if os.environ["TEETHLAND_MODE"] == "NO_VISTA":
            print(f"Running in no pyvista mode, not starting Xvfb.")
        else:
            pyvista.start_xvfb()


class_to_lm_idx = {'Mesial': 1, 'Distal': 2, 'Cusp': 3, 'InnerPoint': 4, 'OuterPoint': 5, 'FacialPoint': 6}
lm_to_class_idx = {v: k for k, v in class_to_lm_idx.items()}


def compute_class_precisions_and_recall(
        lms_list: List[Tuple[np.ndarray, int]],
        detections_list: List[Tuple[np.ndarray, int]],
        lm_class: int,
        thresholds: np.ndarray
) -> Tuple[List[float], List[float]]:
    """
    Computes the precision for a given landmark class
    :param lms_list: list of tuples (landmark coordinates, landmark class)
    :param detections_list: list of tuples (detection coordinates, detection class)
    :param lm_class: landmark class
    :param thresholds: list of distance thresholds
    :return: list of precisions, list of recalls
    """
    precisions = []
    recalls = []
    for threshold in thresholds:
        tp = 0
        fp = 0
        fn = 0
        for b in range(len(lms_list)):
            for lm in lms_list[b]:
                if lm[1] == lm_class:
                    for det in detections_list[b]:
                        if det[1] == lm_class:
                            if np.linalg.norm(lm[0] - det[0]) < threshold:
                                tp += 1
                                break
                    else:
                        fn += 1

            for det in detections_list[b]:
                if det[1] == lm_class:
                    for lm in lms_list[b]:
                        if lm[1] == lm_class:
                            if np.linalg.norm(lm[0] - det[0]) < threshold:
                                break
                    else:
                        fp += 1

        if tp + fp == 0:
            precisions.append(1.0)
        else:
            precisions.append(tp / (tp + fp))

        if tp + fn == 0:
            recalls.append(1.0)
        else:
            recalls.append(tp / (tp + fn))

    return precisions, recalls


def get_ar_ap(out_dict: Dict, thresholds: np.array) -> Dict:
    """
    Computes the average precision and average recall for each class and total mAP and mAR
    :param out_dict: dictionary containing the landmarks and detections lists
    :param thresholds: array of distance thresholds
    :return: dictionary containing the mAP, mAR, APs and ARs for each class
    """
    result_dict = {
        "mAP": 0,
        "mAR": 0,
        "APs": {},
        "ARs": {}
    }

    for c in range(1, 7):
        ps, rs = [], []
        p, r = compute_class_precisions_and_recall(
            out_dict["lms_list"],
            out_dict["detections_list"],
            c,
            thresholds
        )

        ps += p
        rs += r

        result_dict["APs"][lm_to_class_idx[c]] = np.mean(p)
        result_dict["ARs"][lm_to_class_idx[c]] = np.mean(r)

    result_dict["mAP"] = np.mean(list(result_dict["APs"].values()))
    result_dict["mAR"] = np.mean(list(result_dict["ARs"].values()))

    return result_dict

def visualize_submission_output(csv_path: str, meshes_dir: str, distmaps_dir: str):

    # load the csv file
    import csv
    with open(csv_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        lms_dict = {}
        for row in csv_reader:
            key = row['key']
            if key not in lms_dict:
                lms_dict[key] = []
            coord = np.array([float(row['coord_x']), float(row['coord_y']), float(row['coord_z'])])
            lms_dict[key].append((coord, class_to_lm_idx[row['class']]))

    for i, (key, lms) in zip(range(len(lms_dict.keys())), lms_dict.items()):
        mesh_path = os.path.join(meshes_dir, key + ".obj")
        mesh = trimesh.load(mesh_path)

        pl = pyvista.Plotter(shape=(1, 6))

        distmaps = np.load(os.path.join(distmaps_dir, "distmap_" + str(i) + ".npy"))

        for cls in range(1, 7):
            pl.subplot(0, cls-1)
            pl.add_text(lm_to_class_idx[cls])

            dists = distmaps[..., cls-1]
            if dists.shape[0] == len(mesh.vertices):
                pl.add_mesh(pyvista.wrap(mesh), scalars=-dists)
            else:
                pl.add_mesh(pyvista.wrap(mesh))
                pts_subset = pyvista.PolyData(mesh.vertices[out_dict["v_subset_indices"][b].cpu()])
                pl.add_points(
                    pts_subset,
                    render_points_as_spheres=True,
                    point_size=10,
                    scalars=-dists
                )

            c_lms = [xyz for xyz, c in lms if c==cls]
            if c_lms:
                for lm in c_lms:
                    pl.add_mesh(pyvista.Sphere(radius=0.4, center=lm), color="red", opacity=0.5)

        pl.show()


def visualize_dict(out_dict: Dict) -> str:
    B = len(out_dict["file_paths"])
    pl = pyvista.Plotter(shape=(B, 6))
    case_id = None

    for b in range(B):
        case_id = out_dict["case_ids"][b]
        mesh = trimesh.load(out_dict["file_paths"][b])
        mesh.apply_transform(out_dict["transform_matrices"][b].cpu())

        for cls in range(1, 7):
            pl.subplot(b, cls-1)
            pl.add_text(lm_to_class_idx[cls])

            if "dist_maps" in out_dict:
                dists = out_dict["dist_maps"][b].cpu()[..., cls-1].numpy()
                if dists.shape[0] == len(mesh.vertices):
                    pl.add_mesh(pyvista.wrap(mesh), scalars=-dists)
                else:
                    pl.add_mesh(pyvista.wrap(mesh))
                    pts_subset = pyvista.PolyData(mesh.vertices[out_dict["v_subset_indices"][b].cpu()])
                    pl.add_points(
                        pts_subset,
                        render_points_as_spheres=True,
                        point_size=10,
                        scalars=-dists
                    )

            lms = [xyz for xyz, c in out_dict["lms_list"][b] if c==cls]
            if lms:
                for lm in lms:
                    pl.add_mesh(pyvista.Sphere(radius=0.4, center=lm), color="red", opacity=0.5)

            if "detections_list" in out_dict:
                dets = [xyz for xyz, c in out_dict["detections_list"][b] if c==cls]
                if dets:
                    for det in dets:
                        pl.add_mesh(pyvista.Sphere(radius=0.4, center=det), color="white", opacity=1)

    pl.link_views()
    local_path = f'./tmp-3dview-{case_id}.html'
    pl.export_html(local_path)
    # pl.show()

    return local_path


def visualize_features(out_dict: Dict) -> str:
    B = len(out_dict["file_paths"])
    pl = pyvista.Plotter(shape=(B, 1))
    case_id = None

    for b in range(B):
        case_id = out_dict["case_ids"][b]
        mesh = trimesh.load(out_dict["file_paths"][b])
        mesh.apply_transform(out_dict["transform_matrices"][b].cpu())

        features = out_dict["features"][b].cpu().numpy()
        coords = out_dict['input_pcloud'][b][:, :3].cpu().numpy()
        pca = PCA(n_components=3)
        pca.fit(features)

        # Project the original points onto the PCA components
        projected_points = pca.transform(features)

        # Normalize the projections to [0, 1] for RGB mapping
        min_max_scaler = skp.MinMaxScaler()
        normalized_projections = min_max_scaler.fit_transform(projected_points)

        # Use the normalized projections as RGB values
        colors = normalized_projections

        # Create a PolyData object with the original points
        cloud = pyvista.PolyData(coords)
        cloud['colors'] = colors  # Add colors as a point data array

        # Create a Plotter object and add the colored point cloud
        if colors.shape[0] == len(mesh.vertices):
            # pl.add_mesh(mesh, scalars=colors)
            pl.add_mesh(cloud, scalars='colors', rgb=True, show_scalar_bar=True)

    local_path = f'./tmp-features-{case_id}.html'
    pl.link_views()
    pl.export_html(local_path)

    return local_path


if __name__ == "__main__":

    visualize_submission_output("E:/Hackaton/3DTeethLand-MICCAI24/output/predictions.csv",
                                "D:/Data/3DTeethLand/testdata/",
                                "E:/Hackaton/3DTeethLand-MICCAI24/output/")
    exit()

    out_dict = {
        "lms_list": [],
        "detections_list": []
    }

    for i in range(4):
        lms_list = []
        detections_list = []
        for j in range(np.random.randint(256)):
            xyz = np.random.rand(3)
            c = np.random.randint(1, 7)
            if np.random.rand() > 0.05:
                lms_list.append((xyz, c))
            if np.random.rand() > 0.05:
                detections_list.append((xyz + np.random.rand(3) * 0.01, c))

        out_dict["lms_list"].append(lms_list)
        out_dict["detections_list"].append(detections_list)

    result_dict = get_ar_ap(out_dict, thresholds=np.linspace(0, 0.1, 100))
    print(result_dict)
