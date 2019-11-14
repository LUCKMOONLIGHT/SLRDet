# MMDetection For Remote Sensing

**News**: This project is base on mmdetection to reimplement RRPN and use the model Faster R-CNN OBB

## Introduction

The master branch works with **PyTorch 1.1** or higher.

mmdetection is an open source object detection toolbox based on PyTorch. It is
a part of the open-mmlab project developed by [Multimedia Laboratory, CUHK](http://mmlab.ie.cuhk.edu.hk/).

![demo image](demo/P1858.png)


## Benchmark and model zoo

Supported methods and backbones are shown in the below table.
Results and models are available in the [Model zoo](docs/MODEL_ZOO.md).

|                    | ResNet   | ResNeXt  | SENet    | VGG      | HRNet |
|--------------------|:--------:|:--------:|:--------:|:--------:|:-----:|
| RPN                | ✓        | ✓        | ☐        | ✗        | ✓     |
| Fast R-CNN         | ✓        | ✓        | ☐        | ✗        | ✓     |
| Faster R-CNN       | ✓        | ✓        | ☐        | ✗        | ✓     |
| Mask R-CNN         | ✓        | ✓        | ☐        | ✗        | ✓     |
| Cascade R-CNN      | ✓        | ✓        | ☐        | ✗        | ✓     |
| Cascade Mask R-CNN | ✓        | ✓        | ☐        | ✗        | ✓     |
| SSD                | ✗        | ✗        | ✗        | ✓        | ✗     |
| RetinaNet          | ✓        | ✓        | ☐        | ✗        | ✓     |
| GHM                | ✓        | ✓        | ☐        | ✗        | ✓     |
| Mask Scoring R-CNN | ✓        | ✓        | ☐        | ✗        | ✓     |
| FCOS               | ✓        | ✓        | ☐        | ✗        | ✓     |
| Double-Head R-CNN  | ✓        | ✓        | ☐        | ✗        | ✓     |
| Grid R-CNN (Plus)  | ✓        | ✓        | ☐        | ✗        | ✓     |
| Hybrid Task Cascade| ✓        | ✓        | ☐        | ✗        | ✓     |
| Libra R-CNN        | ✓        | ✓        | ☐        | ✗        | ✓     |
| Guided Anchoring   | ✓        | ✓        | ☐        | ✗        | ✓     |

Other features
- [x] DCNv2
- [x] Group Normalization
- [x] Weight Standardization
- [x] OHEM
- [x] Soft-NMS
- [x] Generalized Attention
- [x] GCNet
- [x] Mixed Precision (FP16) Training


## Installation

1. Please refer to [INSTALL.md](docs/INSTALL.md) for installation and dataset preparation.
2. Before install, you should make sure the configuration is correct

```shell
vim ~/.condarc
channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
show_channel_urls: true
vim ~/.bashrc
export GCCPATH=/mnt/lustre/share/gcc/gcc-5.3.0
export PATH=$GCCPATH/bin:$PATH
export CC=$GCCPATH/bin/gcc
export CXX=$GCCPATH/bin/g++
export LD_LIBRARY_PATH=$GCCPATH/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/mnt/lustre/share/gcc/gmp-4.3.2/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/mnt/lustre/share/gcc/mpc-0.8.1/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/mnt/lustre/share/gcc/mpfr-2.4.2/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/mnt/lustre/share/cuda-9.0
export PATH=$CUDA_HOME/bin:$PATH
export PATH=/mnt/lustre/share/cuda-9.0/lib64/libcudnn.so.7.0.4::$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
export LIBRARY_PATH=$LIBRARY_PATH:$CUDA_HOME/lib64
```

3. You can install directly from the script below

```shell
export INSTALL_DIR=$PWD
conda create -n open-mmlab python=3.7 -y
source activate open-mmlab
conda install pytorch torchvision==0.2.2 cuda90 cudatoolkit=9.0 -y
conda install cython -y
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
cd $INSTALL_DIR
git clone git@gitlab.bj.sensetime.com:yanhongchang/mmdetection.git
cd mmdetection
git checkout rotated
python setup.py build develop
python setup_rotated.py build develop
unset INSTALL_DIR
rm -rf /mnt/lustre/yanhongchang/.conda/envs/open-mmlab/lib/python3.7/site-packages/torchvision-0.4.1-py3.7-linux-x86_64.egg/
```

## Get Started

Please see [GETTING_STARTED.md](docs/GETTING_STARTED.md) for the basic usage of MMDetection.

