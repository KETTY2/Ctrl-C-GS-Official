<div align="center">

#Ctrlc gaussian splatting

## Installation

This repository is built on top of [***EasyVolcap***](https://github.com/zju3dv/EasyVolcap), and [***EnvGS***](https://github.com/zju3dv/EnvGS) 

```shell
# Create a new conda environment
conda env create -f environment.yml
conda activate envgs

# Install PyTorch
# Be sure you have CUDA installed, CUDA 11.8 is recommended for the best performance
# NOTE: you need to make sure the CUDA version used to compile the torch is the same as the version you installed
# NOTE: for avoiding any mismatch when installing other dependencies like Pytorch3D
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118  # change the CUDA version according to your own CUDA version

# Register EasyVolcp for imports
pip install -e . --no-build-isolation --no-deps
```

```shell

# Install the 2D Gaussian Tracer
pip install -v submodules/diff-surfel-tracing --no-build-isolation 

# Install the modified 2D Gaussian rasterizers
pip install submodules/diff-surfel-rasterizations/diff-surfel-rasterization-wet submodules/diff-surfel-rasterizations/diff-surfel-rasterization-wet-ch05 --no-build-isolation
```

## Datasets

In this section, we provide instructions on downloading the full dataset for *Ref-NeRF* sedan. You can download our pre-processed ***EasyVolcap*** format datasets in this [Google Drive link](https://drive.google.com/drive/folders/1ogZF8171GatQokbECf1yCabBwm3IvDSm?usp=sharing). 
After downloading, the extracted files should be placed at data/datasets/refnerf/ref_real/sedan/

***Ctrl-C*** follows the typical dataset setup of ***EasyVolcap***, where we group similar sequences into sub-directories of a particular dataset. Inside those sequences, the directory structure should generally remain the same. For example, after downloading and preparing the `sedan` sequence of the *Ref-Real* dataset, the directory structure should look like this:

```shell
# data/datasets/ref_real/sedan:
images # raw images, cameras inside: images/00, images/01 ...
sparse # SfM sparse reconstruction result copied from the original COLMAP format dataset
extri.yml # extrinsic camera parameters, not required if the optimized folder is present
intri.yml # intrinsic camera parameters, not required if the optimized folder is present
# optional, if no normals are provided, set `dataloader_cfg.dataset_cfg.use_normals=False`
normals # prepared monocular normal maps, cameras inside: normals/00, normals/01 ...
```

We have provided the dataset configurations for the *Ref-NeRF*  in the [`configs/datasets/ref_real`](configs/datasets/ref_real) and , and their corresponding training configurations in the [`configs/exps/envgs/ref_real`](configs/exps/envgs/ref_real)


## Usage

### Rendering

You can download our pre-trained models from this [Google Drive link](https://drive.google.com/drive/folders/1p3bohsSSVf1mP3K26Sy47nm1Fl_YvE7I?usp=sharing). After downloading, place them into `data/trained_model` (e.g., `data/trained_model/envgs/envgs/envgs_audi/latest.pt`).

After placing the models and datasets in their respective places, you can run ***EasyVolcap*** with their corresponding experiment configs located in [`configs/exps/envgs`](configs/exps/envgs) to perform rendering operations with ***EnvGS***.

For example, to render the `audi` scene of the *EnvGS* dataset, you can run:

```shell
evc-train -c configs/exps/envgs/ref_real/envgs_sedan.yaml exp_name=envgs/ref_real/envgs_sedan # sedan
evc-test -c configs/exps/envgs/ref_real/envgs_sedan_ctrlc.yaml 
```


Please pay attention to the console logs and keep an eye out for the loss and metrics. All records and training time evalutions will be saved to `data/record` and `data/result` respectively. So, launch your tensorboard or other viewing tools for training inspection.

For preparation and training on custom datasets, you can follow the instructions in the [Custom Datasets Section](#custom-datasets), and then run the training and rendering commands specified above.

<details> <summary> Some useful parameters you can explore with, good luck with them </summary>

+ `runner_cfg.resume`: whether to restart the training from where you stopped the last time.
+ `runner_cfg.epochs`: number of epochs to train, a epoch consist 500 iterations by default.
+ `model_cfg.supervisor_cfg.perc_loss_weight=0.01`: the default LPIPS loss weight is set to 0.01, you can try setting it to 0.1, which may produce better results for some scenes, or you can disable it by setting it to 0.0 for faster training, and comparable results.
+ `model_cfg.sampler_cfg.init_specular=0.001`: you can try to set it to a larger value like 0.01 or 0.1 for scenes or objects with strong reflection for better results.
+ `model_cfg.sampler_cfg.env_max_gs=2000000`: set the maximum number of environment Gaussian.
+ `model_cfg.sampler_cfg.normal_prop_until_iter=18000`: maximum iteration that normal propagation is performed.
+ `model_cfg.sampler_cfg.color_sabotage_until_iter=18000`: maximum iteration that color sabotage is performed.
+ `model_cfg.sampler_cfg.densify_until_iter=21000`: maximum densification iteration for the base Gaussian.
+ `model_cfg.sampler_cfg.env_densify_until_iter=21000`: maximum desification iteration for the environment Gaussian.

</details>


## Custom Datasets

### Dataset Preparation

In the following, we'll be walking throught the process of training *EnvGS* on a custom multi-view dataset.

Let's call the dataset `envgs` and call the scene `audi` for notation. Note that you can change out the `envgs` and `audi` parts for other names for your custom dataset. Other namings like *envgs* should remain the same.

Let's assume a typical input contains calibrated camera parameters compatible with [***EasyVolcap***](https://github.com/zju3dv/EasyVolcap), where the folder & directory structure looks like this:

```shell
data/envgs/audi
│── extri.yml
│── intri.yml
├── images
│   ├── 00
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   ...
│   │   ...
│   └── 01
│   ...
└── normal (optional)
    ├── 00
    │   ├── 000000.jpg
    │   ├── 000001.jpg
    │   ...
    │   ...
    └── 01
    ...
```

We assume the [***EasyVolcap***](https://github.com/zju3dv/EasyVolcap) format dataset has already been prepared. If not, you could follow the instructions below:

```shell
# Define the dataset name and scene name
dataset=envgs
scene=audi
colmap_root=data/datasets/original/$dataset
easyvolcap_root=data/datasets/$dataset

# 1. Run ffmpeg: if you start with a video at `data/datasets/original/envgs/audi/video.mp4`
# 1.1 Make sure the images directory exists
mkdir -p $colmap_root/$scene/images
# 1.2 Set the frame extraction step, e.g., 1 for 1 frame per second, 5 for 5 frames per second, usually a total number of around 200 frames is enough for training
step=2
# 1.3 Run ffmpeg for frame extraction
ffmpeg -i $colmap_root/$scene/video.mp4 -q:v 1 -start_number 0 -r $step $colmap_root/$scene/images/%06d.jpg -loglevel quiet

# 2. Run COLMAP: once you have images stored in `data/datasets/original/envgs/audi/images/*.jpg`
python scripts/colmap/run_colmap.py --data_root $colmap_root/$scene --images images

# 3. COLMAP to EasyVolcap: convert the colmap format dataset `data/datasets/original/envgs/audi` to EasyVolcap format `data/datasets/envgs/audi`
# `--colmap colmap/colmap_sparse/0` is the default COLMAP sparse output directory if you are using the `run_colmap.py` script in the previous step, you can change it to your own COLMAP sparse output directory
python scripts/preprocess/colmap_to_easyvolcap.py --data_root $colmap_root --output $easyvolcap_root --scenes $scene --colmap colmap/colmap_sparse/0

# 4. Run StableNormal: prepare the monocular normal maps for supervision
python submodules/StableNormal/run.py --data_root $easyvolcap_root --scenes $scene

# 5. Metadata: prepare the scene-specific dataset configs parameters for EnvGS
# `--eval` is used for standard evaluation, namely use [0, None, 8] as the testing view sample
python scripts/preprocess/tools/compute_metadata.py --data_root $easyvolcap_root --scenes $scene --eval
```

### Configurations

Given the dataset, you're now prepared to create your corresponding configuration file for *EnvGS*.
The first file corresponds to the dataset itself, where data loading paths and input ratios or view numbers are defined. Let's put it in [`configs/datasets/envgs/audi.yaml`](configs/datasets/envgs/audi.yaml). You can look at the actual file to get a grasp of what info this file should contain. At the minimum, you should specify the data loading root for the dataset. If you feel unfamiliar with the configuration system, feel free to check out the specific [documentation](docs/design/config.md) for that part. The content of the `audi.yaml` (and its parent `envgs.yaml`) file should look something like this:

```yaml
# Content of configs/datasets/envgs/envgs.yaml
dataloader_cfg:
    dataset_cfg: &dataset_cfg
        ratio: 0.25

val_dataloader_cfg:
    dataset_cfg:
        <<: *dataset_cfg

model_cfg:
    sampler_cfg:
        bounds: [[-20.0, -20.0, -20.0], [20.0, 20.0, 20.0]]
        env_bounds: [[-20.0, -20.0, -20.0], [20.0, 20.0, 20.0]]

# Content of configs/datasets/envgs/audi.yaml
configs: configs/datasets/envgs/envgs.yaml

dataloader_cfg:
    dataset_cfg: &dataset_cfg
        ratio: 0.25
        data_root: data/datasets/envgs/audi
        view_sample: [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23,
                      25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45,
                      46, 47, 49, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 63, 65, 66, 67,
                      68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 81, 82, 83, 84, 85, 86, 87, 89,
                      90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102, 103, 105, 106, 107, 108,
                      109, 110, 111, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 125,
                      126, 127, 129, 130, 131, 132, 133, 134, 135, 137, 138, 139, 140, 141, 142,
                      143, 145, 146, 147, 148, 149, 150, 151, 153, 154, 155, 156, 157, 158, 159,
                      161, 162, 163, 164, 165, 166, 167, 169, 170, 171, 172, 173, 174, 175, 177,
                      178, 179, 180, 181, 182, 183, 185, 186, 187, 188, 189, 190, 191, 193, 194,
                      195, 196, 197, 198, 199, 201]

val_dataloader_cfg:
    dataset_cfg:
        <<: *dataset_cfg
        view_sample: [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136,
                      144, 152, 160, 168, 176, 184, 192, 200]

model_cfg:
    sampler_cfg:
        preload_gs: data/datasets/envgs/audi/sparse/0/points3D.ply
        spatial_scale: 6.437488746643067
        # Environment Gaussian
        env_preload_gs: data/datasets/envgs/audi/envs/points3D.ply
        env_bounds: [[-24.43369652781677, -9.675989911182787, -21.932267889066896],
                     [24.759688617142107, 1.977861847608774, 54.04323229716381]]
```

Here you'll see I created a general description file `configs/datasets/envgs/envgs.yaml` for the whole *EnvGS* dataset, which is a good practice if your multi-view dataset contains multiple different scenes but they are under roughly the same setting (view count, lighting condition, image ratio you want to train with, etc.). You'll also note I explicitly specified how many views and frames this dataset has. The number you put in here should not exceed the actual amount. If you're feeling lazy you can also just write `[0, null, 1]` for `view_sample` and `frame_sample`, however doing this means a trained model will still require access to the original dataset to perform some loading and rendering.

#### EnvGS Required Configurations

NOTE: There are some specific dataset-related parameters required by *EnvGS*, you can get all these parameters by running [`scripts/preprocess/tools/compute_metadata.py`](scripts/preprocess/tools/compute_metadata.py).

+ `model_cfg.sampler_cfg.env_bounds=[[..., ..., ...], [..., ..., ...]]`: this the calculated 3d bounding box of the COLMAP sparse points., used for the environment Gaussian initialization.
+ `val_dataloader_cfg.dataset_cfg.view_sample=[...,]`: following default evaluation setting of previous works like [Ref-NeRF](https://dorverbin.github.io/refnerf/) and [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), the test views are selected every 8th view.
+ `dataloader_cfg.dataset_cfg.view_sample=[...,]`: the training views are the remaining views.

Until now, such data preparation is generalizable across all multi-view datasets supported by ***EasyVolcap***, you should always create the corresponding dataset configurations for your custom ones as this helps in reproducibility.

Our next step is to create the corresponding *EnvGS* configuration for running experiments on the `audi` scene. You can create a [`configs/exps/envgs/envgs/audi.yaml`](configs/exps/envgs/envgs/envgs_audi.yaml) to hold such information, you can look at the actual file to get a grasp of what info this file should contain:

```yaml
configs:
    - configs/base.yaml # default arguments for the whole codebase
    - configs/models/envgs.yaml # network model configuration
    - configs/datasets/envgs/audi.yaml # dataset usage configuration
    # - configs/specs/optimized.yaml # specific usage configuration, maybe optimize the camera parameters?

# prettier-ignore
exp_name: {{fileBasenameNoExtension}}
```

You'll notice I placed the configurations in an order of `base`, `model`, `dataset`, and then `specs`. This is typically the best practice as you get more and more specific about the experiment you want to perform here.


## TODOs

- [x] TODO: Release 2D Gaussian ray tracer.
- [x] TODO: Release the training and evaluation code of EnvGS.


## Acknowledgments

This work is implemented using our PyTorch framework, [EasyVolcap](https://github.com/zju3dv/EasyVolcap), feel free to explore it.


We would also like to acknowledge the following inspiring prior work:
- [EasyVolcap: Accelerating Neural Volumetric Video Research](https://github.com/zju3dv/EasyVolcap)
- [NeRF-Casting: Improved View-Dependent Appearance with Consistent Reflections](https://dorverbin.github.io/nerf-casting/)
- [3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes](https://gaussiantracer.github.io/)
- [Ref-NeRF: Structured View-Dependent Appearance for Neural Radiance Fields](https://dorverbin.github.io/refnerf/)
- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://github.com/graphdeco-inria/gaussian-splatting)
- [2DGS: 2D Gaussian Splatting for Geometrically Accurate Radiance Fields](https://surfsplatting.github.io/)


## Citation

If you find this code useful for your research, please cite us using the following BibTeX entry.

```
@article{xie2024envgs,
  title={EnvGS: Modeling View-Dependent Appearance with Environment Gaussian},
  author={Xie, Tao and Chen, Xi and Xu, Zhen and Xie, Yiman and Jin, Yudong and Shen, Yujun and Peng, Sida and Bao, Hujun and Zhou, Xiaowei},
  journal={arXiv preprint arXiv:2412.15215},
  year={2024}
}

@article{xu2023easyvolcap,
  title={EasyVolcap: Accelerating Neural Volumetric Video Research},
  author={Xu, Zhen and Xie, Tao and Peng, Sida and Lin, Haotong and Shuai, Qing and Yu, Zhiyuan and He, Guangzhao and Sun, Jiaming and Bao, Hujun and Zhou, Xiaowei},
  booktitle={SIGGRAPH Asia 2023 Technical Communications},
  year={2023}
}
```
