# CEM3DMG
## Overview
**This project is the implementation and data of CEM3DMG, which is a framework for generating three-dimensional microstructures of cement from 2D images.**
**If your publication utilises our project, please cite our paper**
```
@article{ZHAO2025107726,
title = {3D microstructural generation from 2D images of cement paste using generative adversarial networks},
journal = {Cement and Concrete Research},
volume = {187},
pages = {107726},
year = {2025},
issn = {0008-8846},
doi = {https://doi.org/10.1016/j.cemconres.2024.107726},
url = {https://www.sciencedirect.com/science/article/pii/S0008884624003077},
author = {Xin Zhao and Lin Wang and Qinfei Li and Heng Chen and Shuangrong Liu and Pengkun Hou and Jiayuan Ye and Yan Pei and Xu Wu and Jianfeng Yuan and Haozhong Gao and Bo Yang},
}
```
## Configuration requirements:
* Linux or Windows
* NVIDIA GPU (Recommended A100 or A800) 
* PyTorch 1.11
* Cudatoolkit 11.3.1
* Numpy 1.21.5
* Torchvision 0.12.0
* Pillow 9.0.1

## Files description
### code
* **networkdemo.py** network structure of the models in CEM3DMG
* **functiondemo.py** implementation of different functions in CEM3DMG
* **Train.py** the design of the training process and the hyperparameters settings of CEM3DMG
* **generate3D.py** Using the trained model to generate 3D microstructure images
### image
* training data
### data
* real 2D images
* generated 3D microstructure images
### model
* saved models
### trained
* saved image results during training
### output
* saved generated 3D images
### checkpoint
* saved checkpoints during training

## Usage Example
To run the code you need to get the pytorch VGG19-Model. This project already contains the VGG-19 model. You only need to unzip vgg_conv.zip in the code folder to get the VGG-19 pre-trained model. Note: The pytorch VGG19-Model from the paper:"Very Deep Convolutional Networks for Large-Scale Image Recognition"
### Training CEM3DMG
* python ./code/Train.py --img='Example.bmp' --direction_w=48  --tscale=200 --cha=8 --iter_train=50000
* python ./code/Train.py --img='SampleB.bmp' --direction_w=32  --tscale=304 --cha=32 --iter_train=100000
### Using CEM3DMG
* python ./code/generate3D.py

**Suggestions for some hyperparameters: training iterations (iter_train) and the image cropping size (tscale) are critical in the training process of CEM3DMG**
* **iter_train**: The training iteration is usually set to 50,000-120,000.
* **tscale**: The image crop size needs to be determined based on the given 2D image and the crop size should be representative.

