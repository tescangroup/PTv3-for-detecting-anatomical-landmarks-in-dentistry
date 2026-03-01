# Team 3DIMLand: Solution to MICCAI2024 3DTeethLand Challenge

## Introduction

This repository contains the official implementation of Team 3DIMLand's submission for the MICCAI 2024 3DTeethLand Challenge (https://www.synapse.org/Synapse:syn57400900/wiki/627259). 
## Usage (Ubuntu & NVIDIA GPU)

### Requirements
This project uses PDM and installs PyTorch CUDA 11.8 wheels from the official PyTorch wheel index, plus extra CUDA extensions (PyG ops, spconv, flash-attn) via a post_install script.
Make sure your GPU driver is working, and you have Python installed.
### Installation
Install `pdm`:

```bash
pip install pdm
```
Clone this repository:

```bash
git clone https://github.com/tescangroup/PTv3-for-detecting-anatomical-landmarks-in-dentistry.git
cd PTv3-for-detecting-anatomical-landmarks-in-dentistry
```
Install the dependencies and the package itself:
```bash
pdm use -f python3.12
pdm install
```
This will resolve and install dependencies into PDM’s environment. 

Install CUDA extension dependencies:

```bash
pdm run post_install
```
This installs:

PyTorch Geometric optional CUDA extension wheels (pyg_lib, torch_scatter, torch_sparse, torch_cluster, torch_spline_conv), 
spconv-cu118, flash-attn (attempts wheel install first; may build if no matching wheel), PyGeM from GitHub.
### Running the Code

**First, prepare the data**

1. Download the mesh files (STL format) from the challenge website:
   https://www.synapse.org/Synapse:syn57400900/wiki/627934

2. Place all STL files into the `STLData` subfolder, which should be located within the directory specified by `default_root_dir` in the Hydra config (e.g., `data/STLData`).
   - The `STLData` folder should directly contain the mesh files (no subfolders).

3. For annotations, download the single JSON file with landmark annotations from: https://drive.google.com/file/d/1YTbt6gbtO0_e9EgFTf-j2DiLKF1V1Kle/view?usp=sharing
   - Place this file as specified by `dataset_json_path` in the config (e.g., `data/3DTeethLand2024.json`).

4. Segmentation masks are not required and are not used in this pipeline, so you do not need to download them.

**Example directory structure:**
```
data/
  STLData/
    mesh1.stl
    mesh2.stl
    ...
  3DTeethLand2024.json
  preprocessed_data/
```

You can also download the preprocessed data from https://drive.google.com/file/d/1ZLdn5kjBxOYoXBL4jai7gJUEepzBhGeF/view?usp=sharing and place it in the `preprocessed_data` folder.
Otherwise, the code will preprocess the STL files on the fly during training and inference, and save the preprocessed data in this folder for future runs.
The preprocessing includes applying morphing of the meshes and generating geodesic maps, as well
as filtering out meshes without any landmarks. The final number of used meshes from the initial dataset is 220 (out of 1800).

To train the model, simply run:

```bash
python train.py
```

The script will automatically use the paths and training configurations specified in the Hydra config files (`config/paths/paths.yaml` and `config/training/training.yaml`).

You can modify any hyperparameters or paths by editing the corresponding YAML files in the `config` directory.

Experiment logging is performed using ClearML by default. If you prefer to use another logging platform (e.g., TensorBoard, WandB), you can change the `logger` implementation in `train.py` and pass your custom logger to the PyTorch Lightning `Trainer`.

## Acknowledgements

## References

If you found this code useful, please cite our work (paper of our method and the challenge paper):

```
@InProceedings{10.1007/978-3-031-88977-6_20,
    title={Leveraging Point Transformers for Detecting Anatomical Landmarks in Digital Dentistry},
    author={Kub{\'i}k, Tibor
    and Kodym, Old{\v{r}}ich
    and {\v{S}}illing, Petr
    and Tr{\'a}vn{\'i}{\v{c}}kov{\'a}, Kate{\v{r}}ina
    and Moj{\v{z}}i{\v{s}}, Tom{\'a}{\v{s}}
    and Matula, Jan},
    year={2025},
    booktitle={Supervised and Semi-supervised Multi-structure Segmentation and Landmark Detection in Dental Data},
    isbn={978-3-031-88977-6}
}
```

```
@misc{benhamadou2025detectingdentallandmarksintraoral,
    title={Detecting Dental Landmarks from Intraoral 3D Scans: the 3DTeethLand challenge}, 
    author={Achraf Ben-Hamadou and Nour Neifar and Ahmed Rekik and Oussama Smaoui and Firas Bouzguenda and Sergi Pujades and Niels van Nistelrooij and Shankeeth Vinayahalingam and Kaibo Shi and Hairong Jin and Youyi Zheng and Tibor Kubík and Oldřich Kodym and Petr Šilling and Kateřina Trávníčková and Tomáš Mojžiš and Jan Matula and Jeffry Hartanto and Xiaoying Zhu and Kim-Ngan Nguyen and Tudor Dascalu and Huikai Wu and and Weijie Liu and Shaojie Zhuang and Guangshun Wei and Yuanfeng Zhou},
    year={2025},
    eprint={2512.08323},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2512.08323}, 
}
```
