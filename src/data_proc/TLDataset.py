########################################################################################################
# Project: 3DTeethland24
# Author(s): K. Travnickova, O. Vavro
#
# This file includes the data loader for the 3DTeethland24 project.
########################################################################################################
import json

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from concurrent.futures import ProcessPoolExecutor

from src.data_proc.data_helpers import TLSerializer
import trimesh
import os
from copy import deepcopy
import torch
import numpy as np
import pymeshlab
from tqdm import tqdm

from pygem.ffd import FFD

class TLDataset(Dataset):
    def __init__(self,
                 json_path: str,
                 tmp_data_path: str = None,
                 sampled_points_num: int = 64000,
                 scale_range: tuple = (0.8, 1.2),
                 translation_range: tuple = (5, 5),
                 rotation_range: tuple =(0.5, 0.5),
                 offline_morphing_range: tuple = (-5, 5),
                 offile_morphing_grid: tuple = (5, 5, 5),
                 num_morphed_meshes: int = 2,
                 max_geodesic_distance: float = 15.0,
                 slice_data: slice = None,
                 only_with_landmarks: bool = True
                 ) -> None:
        assert sampled_points_num is None or sampled_points_num > 0, f'Number of points to sample from mesh must be a positive integer.'
        self.num_preprocessing_changes = 0
        self.serializer = TLSerializer(json_path, only_with_landmarks=only_with_landmarks)
        self.root_path = self.serializer.root_path
        self.sampled_points_num = sampled_points_num
        self.slice_data = slice_data

        if self.slice_data is not None:
            self.serializer.body = self.serializer.body[self.slice_data]
            print(f'training or validation dataloader sliced to {self.slice_data}, length: {len(self.serializer.body)}')

        # reserve 0 for something?
        self._CLASS_LM_TO_INDEX = {'Mesial': 1,
                                   'Distal': 2,
                                   'Cusp': 3,
                                   'InnerPoint': 4,
                                   'OuterPoint': 5,
                                   'FacialPoint': 6}
        # geodesic distance
        self.max_geodesic_distance = max_geodesic_distance

        self.tmp_data_path = tmp_data_path
        self.offline_morphing_range = offline_morphing_range
        self.offile_morphing_grid = offile_morphing_grid
        self.num_morphed_meshes = num_morphed_meshes
        if self.tmp_data_path is not None:
            # we will use offline augmentations
            self.aug_serializer_path = os.path.join(self.tmp_data_path, 'TL24AUG.json')

            self.aug_serializer = TLSerializer(self.aug_serializer_path, drop_unavailable_files=False)
            if len(self.aug_serializer.body) == 0:
                # if the file does not exist or is empty, we need to preprocess the data
                self.offline_preprocess()

        # augmentation parameters
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.rotation_range = rotation_range

    def init_sample_iterator(self,
                             batch_size: int = 1,
                             num_workers: int = 1
                             ) -> None:
        """
        Initializes the iterator for the dataset. The iterator is used to get batches of data.
        Need to call this method before using get_batch method.

        :param batch_size: batch size which will be used in get_batch method.
        :param num_workers: number of workers to use for loading data in parallel.
        """

        if num_workers == 0:
            persistent_workers = False
        else:
            persistent_workers = True

        self.batch_size = batch_size
        self.loader = DataLoader(
            self, batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            drop_last=True,
            collate_fn=self.collate_fn)
        self.iterator = iter(self.loader)

    def preproc_morph(self, idx, verbose=False):

        # check if the mesh is already morphed
        try:
            morphed_count = self.aug_serializer.body[idx]['MorphedCount']
            if morphed_count >= self.num_morphed_meshes:
                if verbose:
                    print("Morphed mesh already exists: ", morphed_count)
                return
        except:
            # if not, morph the mesh
            print("Not morphed yet.")
            self.num_preprocessing_changes += 1
            self.aug_serializer.body[idx]['MorphedCount'] = 0
            self.aug_serializer.body[idx]['MorphedMeshPaths'] = []
            self.aug_serializer.body[idx]['MorphedLandmarks'] = []

        mesh_path = self.aug_serializer.body[idx]['MeshPath']
        meshname = os.path.basename(mesh_path)
        mesh = trimesh.load(mesh_path)

        while self.aug_serializer.body[idx]['MorphedCount'] < self.num_morphed_meshes:
            i = self.aug_serializer.body[idx]['MorphedCount']
            morphed_mesh, morphed_lms = self.get_morphed_mesh(mesh, self.aug_serializer.body[idx]['Landmarks'])
            morphed_mesh_path = os.path.join(self.tmp_data_path, "MORPH_" + str(i) + "_" + meshname)
            morphed_mesh.export(morphed_mesh_path)
            self.aug_serializer.body[idx]['MorphedMeshPaths'].append(morphed_mesh_path)
            self.aug_serializer.body[idx]['MorphedLandmarks'].append(morphed_lms)
            self.aug_serializer.body[idx]['MorphedCount'] += 1
            print("Saved morphed mesh: ", morphed_mesh_path)

    def get_morphed_mesh(self, mesh, lanmarks):

        lms_coords = np.array([lms['coord'] for lms in lanmarks])

        box_size = mesh.bounds[1][0] - mesh.bounds[0][0]
        ffd = FFD(self.offile_morphing_grid)
        ffd.box_length = [box_size, box_size, box_size]
        ffd.box_origin = [-box_size / 2, -box_size / 2, -box_size / 2]

        for x in range(self.offile_morphing_grid[0]):
            for y in range(self.offile_morphing_grid[1]):
                for z in range(self.offile_morphing_grid[2]):
                    # perform some random transformations
                    ffd.array_mu_x[x, y, z] = np.random.uniform(self.offline_morphing_range[0],
                                                                self.offline_morphing_range[1])/box_size
                    ffd.array_mu_y[x, y, z] = np.random.uniform(self.offline_morphing_range[0],
                                                                self.offline_morphing_range[1])/box_size
                    ffd.array_mu_z[x, y, z] = np.random.uniform(self.offline_morphing_range[0],
                                                                self.offline_morphing_range[1])/box_size

        centroid = np.mean(mesh.vertices, axis=0)
        new_coords = ffd(lms_coords - centroid) + centroid
        new_mesh_v = ffd(mesh.vertices - centroid) + centroid
        new_mesh = trimesh.Trimesh(vertices=new_mesh_v, faces=mesh.faces)

        new_lms = deepcopy(lanmarks)
        for i, lms in enumerate(new_lms):
            lms['coord'] = new_coords[i].tolist()

        return new_mesh, new_lms


    def preproc_subdivide(self, idx, verbose=False):

        meshname = os.path.basename(self.serializer.body[idx]['MeshPath'])
        if verbose:
            print("Preprocessing vcount: ", meshname)
        # try to get vertex count from the serializer
        mesh_path = os.path.join(self.serializer.body[idx]['MeshPath'])
        print(f"Vcount not in aug_serializer. idx: {idx}")
        mesh = trimesh.load(mesh_path)

        try:
            vcount = self.aug_serializer.body[idx]['VertexCountOrig']
            if verbose:
                print("vcount: ", vcount)
        except:
            vcount = len(mesh.vertices)

            self.aug_serializer.body[idx]['VertexCountOrig'] = vcount
            mesh_path_new = os.path.join(self.tmp_data_path, meshname)
            self.aug_serializer.body[idx]['MeshPath'] = mesh_path_new
            self.num_preprocessing_changes += 1
            print("Added vcount to aug_serializer: ", vcount)
        mesh.export(self.aug_serializer.body[idx]['MeshPath'])

        if self.sampled_points_num is not None and vcount < self.sampled_points_num:  # handle case when sampling is equal to mesh vertex count
            # check if the mesh is already subdivided
            try:
                vcount = self.aug_serializer.body[idx]['VertexCountSubdivide']
                if vcount > self.sampled_points_num:
                    if verbose:
                        print("Subdivided mesh already exists: ", vcount)
                    return
            except:
                # if not, subdivide the mesh
                print("Not subdivided yet.")

            print("Subdividing mesh: ", meshname)
            mesh_path = self.serializer.body[idx]['MeshPath']
            mesh = trimesh.load(mesh_path)
            while len(mesh.vertices) < self.sampled_points_num:
                mesh = mesh.subdivide()
            self.aug_serializer.body[idx]['VertexCountSubdivide'] = len(mesh.vertices)
            self.aug_serializer.body[idx]['MeshPath'] = os.path.join(self.tmp_data_path, "SUBDIV_" + meshname)
            mesh.export(self.aug_serializer.body[idx]['MeshPath'])
            self.num_preprocessing_changes += 1
            print("Saving subdivided mesh: ", self.aug_serializer.body[idx]['MeshPath'], " with vertices: ", len(mesh.vertices))

    # def preproc_geodesic(self, idx, meshpath, landmarks):
    def preproc_geodesic(self, idx, verbose=False):

        # check if distances are already computed
        try:
            distances = self.aug_serializer.body[idx]['GeodesicDistances']
            if len(distances) >= 1 + self.num_morphed_meshes:
                if verbose:
                    print("Geodesic distances already exist!")
                return
        except:
            # if not, compute the distances
            print("No geodesic distances computed yet.")
            self.num_preprocessing_changes += 1
            self.aug_serializer.body[idx]['GeodesicDistances'] = []

        for c in range(0, self.num_morphed_meshes + 1):
            meshpath, landmarks = self.get_case(idx, c)
            meshname = os.path.basename(meshpath)

            meshname, _ = os.path.splitext(meshname)
            geo_serial_path = os.path.join(self.tmp_data_path, os.path.basename(meshname)+'_GEO_DIST.json')

            print("Computing geodesic distances for mesh: ", meshname)
            mesh = trimesh.load(meshpath)

            lms_list = self.lms_to_tensor(landmarks)

            # compute distance maps
            q = trimesh.proximity.ProximityQuery(mesh)  # closest vertices to landmarks on mesh (precomp.)

            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))

            labels_heatmaps = torch.zeros(len(mesh.vertices),
                                          len(self._CLASS_LM_TO_INDEX.keys())).t()  # TRANSPOSED! to iterate over classes
            # loop over landmark classes
            landmark_coords_aug = torch.Tensor([lms[0] for lms in lms_list])
            landmark_classes_aug = torch.Tensor([lms[1] for lms in lms_list])
            for i, _ in enumerate(labels_heatmaps):
                # calculate closest vertices to landmarks - TODO: make more precise?
                mesh_aug_landmark_coords = landmark_coords_aug[
                    (landmark_classes_aug == (i + 1))]  # mask to select landmarks of only one class

                # skip heatmap generation when no landmarks exist
                if len(mesh_aug_landmark_coords) == 0:
                    labels_heatmaps[i] = torch.ones(len(mesh.vertices)) * self.max_geodesic_distance
                    continue

                # select vertices close to points of a certain landmark class
                _, idxs = q.vertex(mesh_aug_landmark_coords.cpu().numpy())

                sel_string = f"vi=={idxs[0]}"
                for s in range(1, len(idxs)):
                    sel_string += f"||vi=={idxs[s]}"
                ms.set_selection_none()
                ms.compute_selection_by_condition_per_vertex(condselect=sel_string)

                # for all vertices calculate distance to landmarks of a certain class
                ms.compute_scalar_by_geodesic_distance_from_selection_per_vertex(maxdistance=pymeshlab.PureValue(self.max_geodesic_distance))
                distances = ms.current_mesh().vertex_scalar_array()
                #distances[distances > self.max_geodesic_distance] = self.max_geodesic_distance
                distances[distances == 0.0] = self.max_geodesic_distance
                distances[idxs] = 0.0

                # pick only distances for wanted sample points from all computed
                labels_heatmaps[i] = torch.Tensor(distances)

            labels_heatmaps = labels_heatmaps.t()

            with open(geo_serial_path, 'w') as f:
                to_write = {}
                to_write["GeodesicDistances"] = labels_heatmaps.tolist()
                f.write(json.dumps(to_write, indent=2))
            print("Saving geodesic distances for mesh: ", meshname)
            self.aug_serializer.body[idx]['GeodesicDistances'].append(os.path.relpath(geo_serial_path, self.tmp_data_path))

    def unified_preprocess_call(self, idx):
        self.preproc_subdivide(idx)
        self.preproc_morph(idx)
        self.preproc_geodesic(idx)

        return idx

    def offline_preprocess(self):
        # do we have a file with additional metadata?
        if os.path.exists(self.aug_serializer_path):
            self.aug_serializer = TLSerializer(self.aug_serializer_path)
        else:
            self.aug_serializer = TLSerializer(self.serializer.input_json_path)
            if self.slice_data is not None:
                self.aug_serializer.body = self.aug_serializer.body[self.slice_data]
                print(f'aug serializer: training or validation dataloader sliced to {self.slice_data}, length: {len(self.aug_serializer.body)}')

        # TODO this deepcopy does not work, if the JSON does not exists after preprocessing the body variable is filled with preprocessed values
        # body = deepcopy(self.aug_serializer.body)

        # print(f'Offline preprocessing starts, using {os.cpu_count()} workers to prepare data.')
        # with ProcessPoolExecutor(max_workers=None) as executor:
        #     if verbose:
        #         for processed_id in executor.map(self.unified_preprocess_call, list(range(len(self)))):
        #             print(f'Case {processed_id} has been processed in parallel execution.')
        #     else:
        #         list(tqdm(executor.map(self.unified_preprocess_call, list(range(len(self)))), total=len(self)))
        #

        for i in tqdm(range(len(self))):
            self.unified_preprocess_call(i)

        print(f'Offline preprocessing finished, {self.num_preprocessing_changes} changes were made.')
        if self.num_preprocessing_changes == 0:
            print("No changes in the serializer.")
            return

        self.aug_serializer.serialize_json(self.aug_serializer_path)


    def __len__(self):
        return len(self.serializer.body)

    def lms_to_tensor(self,
                      lms: list[dict]
                      ) -> torch.Tensor:
        """
        Converts landmarks to tensor.
        """
        lms_list = []
        for i, landmark in enumerate(lms):
            coords = tuple(landmark['coord'])
            lm_class = self._CLASS_LM_TO_INDEX[landmark['class']]
            lms_list.append((coords, lm_class))

        return lms_list

    @staticmethod
    def get_random_transformation(pcloud_centroid: np.ndarray,
                                  angle_range: tuple[float, float],
                                  scale_range: tuple[float, float],
                                  translation_range: tuple[float, float]
                                  ) -> np.ndarray:
        """
        Generates a random transformation matrix with a random rotation, translation and scaling.
        """

        to_centroid = trimesh.transformations.translation_matrix(-pcloud_centroid)
        from_centroid = trimesh.transformations.translation_matrix(pcloud_centroid)

        R = trimesh.transformations.rotation_matrix(np.random.uniform(angle_range[0], angle_range[1]), np.random.uniform(-1, 1, 3))
        T = trimesh.transformations.translation_matrix(np.random.uniform(translation_range[0], translation_range[1], 3))
        random_scale = np.random.uniform(scale_range[0], scale_range[1])
        S = trimesh.transformations.scale_matrix(random_scale)

        transformation_matrix = trimesh.transformations.concatenate_matrices(from_centroid, T, R, S, to_centroid)

        return transformation_matrix, random_scale

    # def random_transform(self, mesh, landmark_coords, landmark_classes, deep_copy=False):
    def random_transform(self,
                         mesh: trimesh.Trimesh,
                         lms_list: list,
                         deep_copy: bool = False
                         ) -> tuple[trimesh.Trimesh, list, float]:
        """
        Randomly transforms the mesh and landmarks.
        """
        centroid = np.mean(mesh.vertices, axis=0)
        m, scale = self.get_random_transformation(centroid, angle_range=self.rotation_range,
                                       scale_range=self.scale_range,
                                       translation_range=self.translation_range)
        matrix = torch.tensor(m)

        if deep_copy:
            mesh_aug = deepcopy(mesh)
            lms_list_aug = deepcopy(lms_list)
        else:
            mesh_aug = mesh
            lms_list_aug = lms_list

        mesh_aug.apply_transform(matrix.cpu().numpy())

        # transform landmarks
        coords = [lms[0] for lms in lms_list_aug]
        coords_transformed = trimesh.transform_points(coords, matrix.cpu().numpy())
        lms_list_aug = [(coords_transformed[i], lms[1]) for i, lms in enumerate(lms_list_aug)]

        return mesh_aug, lms_list_aug, matrix, scale

    def get_case(self, idx, choice = None):
        # TODO: case serialiezr[idx], branch to get by case_id...
        if self.tmp_data_path is not None:
            possible_meshes = [self.aug_serializer.body[idx]['MeshPath']]
            possible_meshes.extend([m for m in self.aug_serializer.body[idx]['MorphedMeshPaths']])
            possible_landmarks = [self.aug_serializer.body[idx]['Landmarks']]
            possible_landmarks.extend(self.aug_serializer.body[idx]['MorphedLandmarks'])

            if choice is None:
                choice = np.random.randint(0, len(possible_meshes))
            # return random choice from possible meshes
            return possible_meshes[choice], possible_landmarks[choice]
        else:
            return self.serializer.body[idx]['MeshPath'], self.serializer.body[idx]['Landmarks']

    def __getitem__(self,
                    idx: int
                    ) -> tuple:

        choice = None
        # in validation morphed meshes count is 0
        # if self.num_morphed_meshes == 0:
        #     choice = 0
        file_path, case_landmarks = self.get_case(idx, choice=choice)
        case_id = self.serializer.body[idx]['Id']
        lms_list = self.lms_to_tensor(case_landmarks)

        mesh = trimesh.load(file_path)

        # online augmentation
        mesh_aug, lms_list_aug, matrix, scale = self.random_transform(mesh, lms_list, deep_copy=False)

        if self.tmp_data_path is not None:
            # get geodesic distances
            fname, _ = os.path.splitext(file_path)
            geo_serial_path = os.path.join(self.tmp_data_path, os.path.basename(fname) + '_GEO_DIST.json')

            distances = None
            with open(geo_serial_path, 'r') as f:
                geo_serializer = json.loads(f.read())
                distances = geo_serializer['GeodesicDistances']

            # import vedo
            # landmarks_coords_aug = [lms[0] for lms in lms_list_aug]
            # landmarks_coords = [lms[0] for lms in lms_list]
            # mesh0 = vedo.Mesh(mesh, c='b')
            # mesh1 = vedo.Mesh(mesh_aug, c='r')
            # lms0 = vedo.Points(landmarks_coords, r=5, c='r')
            # lms1 = vedo.Points(landmarks_coords_aug, r=5, c='b')
            # vedo.show(mesh0, mesh1, lms0, lms1, axes=1).close()
            # vedo.show(mesh0, lms0)
            # vedo.show(mesh1, lms1)

            # random indices of the vertices or whole mesh
            v_subset_indices = torch.randint(0, len(mesh_aug.vertices), (self.sampled_points_num,)) \
                            if self.sampled_points_num is not None \
                            else torch.arange(0, len(mesh_aug.vertices))

            dist = torch.Tensor(distances)
            dist = dist * scale

            labels_heatmaps = dist[v_subset_indices]

        else:
            v_subset_indices = torch.arange(0, len(mesh_aug.vertices))
            labels_heatmaps = torch.zeros(len(mesh_aug.vertices), len(self._CLASS_LM_TO_INDEX.keys()))

        coords = torch.tensor(mesh_aug.vertices[v_subset_indices.cpu().numpy()])
        normals = torch.tensor(mesh_aug.vertex_normals[v_subset_indices.cpu().numpy()])

        # add normals of the points
        input_pcloud = torch.cat((coords, normals), dim=1)

        return input_pcloud, v_subset_indices, labels_heatmaps, lms_list_aug, case_id, file_path, matrix

    @staticmethod
    def collate_fn(batch):
        """
        """

        return {
            'input_pcloud': torch.stack([item[0] for item in batch]),
            'v_subset_indices': torch.stack([item[1] for item in batch]),
            'labels_heatmaps': torch.stack([item[2] for item in batch]),
            'lms_list': [item[3] for item in batch],
            'case_ids': [item[4] for item in batch],
            'file_paths': [item[5] for item in batch],
            'transform_matrices': [item[6] for item in batch]
        }

    def get_batch(self):
        """
        """

        try:
            return next(self.iterator)
        except StopIteration as e:
            self.iterator = iter(self.loader)
            return next(self.iterator)


if __name__ == '__main__':

    batch_size = 4
    num_workers = 1
    # dataset = TLDataset('D:/Data/3DTeethLand/3DTeethLand2024.json')
    dataset = TLDataset('D:/Data/3DTeethLand/3DTeethLand2024.json', tmp_data_path='D:/Data/3DTeethLand/PreprocessedMeshes/')
    dataset.init_sample_iterator(batch_size=batch_size, num_workers=num_workers)

    from tqdm import tqdm
    print("Measuring get_batch() speed with workers: ", num_workers, " and batch size: ", batch_size)
    for i in tqdm(range(100)):
        batch = dataset.get_batch()


