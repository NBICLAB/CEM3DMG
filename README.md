# CEM3DMG

## Overview
**This project is the implementation and data of CEM3DMG, which is a framework for generating three-dimensional microstructures of cement from 2D images.**

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
### To run the code you need to get the pytorch VGG19-Model from the bethge lab. This project already contains the VGG-19 model. You only need to unzip vgg_conv.zip in the code folder to get the VGG-19 pre-trained model.
### Training CEM3DMG
* python ./code/Train.py --img=Example.bmp --direction_w=48  --tscale=200 --cha=8 --iter_train=50000
* python ./code/Train.py --img=SampleB.bmp --direction_w=32  --tscale=304 --cha=32 --iter_train=100000
### Using CEM3DMG
* python ./code/generate3D.py

**Suggestions for some hyperparameters: training iterations (iter_train) and the image cropping size (tscale) are critical in the training process of CEM3DMG**
* **iter_train**: The training iteration is usually set to 50,000-120,000.
* **tscale**: The image crop size needs to be determined based on the given 2D image and the crop size should be representative.

