<div align="center">

#Ctrlc gaussian splatting

## Installation

This repository is built on top of [***EasyVolcap***](https://github.com/zju3dv/EasyVolcap), and [***EnvGS***](https://github.com/zju3dv/EnvGS) 

```shell
# Create a new conda environment(this may take an hour for I have to set environment based on ubuntu 18, I installed modules such as conda-forge av)
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
cd submodules
git clone https://github.com/xbillowy/diff-surfel-tracing.git --recursive
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

You can download our pre-trained models from this [Google Drive link](https://drive.google.com/drive/folders/1p3bohsSSVf1mP3K26Sy47nm1Fl_YvE7I?usp=sharing). After downloading, place them into `data/trained_model` (e.g., `data/trained_model/envgs_sedan_ctrlc/latest.pt`).

After placing the models and datasets in their respective places, you can run ***EasyVolcap*** with their corresponding experiment configs located in [`configs/exps/envgs_ctrl`](configs/exps/envgs_ctrl) to perform rendering operations with ***Ctrl-C gaussian splatting***.

For example, to render the `sedan` scene of the *Refreal* dataset, you must first train light features and then clone and render:

```shell
training:
evc-train -c configs/exps/envgs/ref_real/envgs_sedan.yaml exp_name=envgs/ref_real/envgs_sedan # sedan

cloning and rendering:

evc-test -c configs/exps/envgs/ref_real/envgs_sedan_ctrlc.yaml 
```
You can change (a,b,c) in configs/exps/envgs/ref_real/envgs_sedan_ctrlc.yaml to manipulate the location the car will be rendered. The picture from the paper used a: -0.5 b: -0.6  c: 1.5 and frame0000_camera0008.png.

Please pay attention to the console logs and keep an eye out for the loss and metrics. All records and training time evalutions will be saved to `data/record` and `data/result` respectively. So, launch your tensorboard or other viewing tools for training inspection.

<details> <summary> Some useful parameters you can explore with, good luck with them </summary>

+ `runner_cfg.resume`: whether to restart the training from where you stopped the last time.
+ `runner_cfg.epochs`: number of epochs to train, a epoch consist 500 iterations by default.
+ `model_cfg.supervisor_cfg.perc_loss_weight=0.01`: the default LPIPS loss weight is set to 0.01, you can try setting it to 0.1, which may produce better results for some scenes, or you can disable it by setting it to 0.0 for faster training, and comparable results.
+ `model_cfg.sampler_cfg.init_specular=0.001`: to make 0 or 1 it is well trained with initial value 0.001
+ `model_cfg.sampler_cfg.env_max_gs=2000000`: set the maximum number of environment Gaussian.
+ `model_cfg.sampler_cfg.normal_prop_until_iter=18000`: maximum iteration that normal propagation is performed.
+ `model_cfg.sampler_cfg.color_sabotage_until_iter=18000`: maximum iteration that color sabotage is performed.
+ `model_cfg.sampler_cfg.densify_until_iter=21000`: maximum densification iteration for the base Gaussian.
+ `model_cfg.sampler_cfg.env_densify_until_iter=21000`: maximum desification iteration for the environment Gaussian.

</details>


