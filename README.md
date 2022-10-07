# Semantic camera calibration toolkit

This repository contains the code used for the ITSC 2022 paper [Extrinsic Camera Calibration with Semantic Segmentation](https://arxiv.org/abs/2208.03949).

# Installation

A linux distribution with a working Python installation is required.  
To create the Python environment, creating a [Python venv](https://docs.python.org/3/library/venv.html) is advised.  
To install the required dependencies:  
`sudo apt install python3.7-dev ffmpeg`, replace 3.7 with the Python version installed on your system.  
`pip3 install -r requirements.txt`  
To checkout the patched Cylinder3D version for automatic point cloud labeling:  
`git submodule update --init --recursive`  
In order to use the CARLA data generator, the CARLA server and Python API needs to installed, see [this link](https://carla.readthedocs.io/en/latest/start_quickstart/) for instructions.  
In order to generate data from the KITTI dataset, the KITTI dataset needs to be symlinked to **datasets/KITTI**.

# Use

To evaluate the algorithm on CARLA data, use `evaluate_carla.sh`. To generate data in CARLA, uncomment the second line in `evaluate_carla.sh`.
