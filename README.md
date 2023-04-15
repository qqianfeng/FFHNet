# FFHNet : Generating Multi-Fingered Robotic Grasps for Unknown Objects in Real-time
FFHNet (ICRA 2022 [Paper](https://ieeexplore.ieee.org/document/9811666)) is an ML model which can generate a wide variety of high-quality multi-fingered grasps for unseen objects from a single view. 

Generating and evaluating grasps with FFHNet takes only 30ms on a commodity GPU. To the best of our knowledge, FFHNet is the first ML-based real-time system for multi-fingered grasping with the ability to perform grasp inference at 30 frames per second (FPS). 

For training, we synthetically generate 180k grasp samples for 129 objects. We are able to achieve 91% grasping success for unknown objects in simulation and we demonstrate the model's capabilities of synthesizing high-quality grasps also for real unseen objects.

![](docs/images/pipeline.png)
## Installation

python >= 3.8
bps_torch installed from repo:
`https://github.com/otaheri/bps_torch`

```
pip install torch==1.7.1 torchvision==0.8.2 tensorboard
pip install pyyaml h5py pandas transforms3d
pip install opencv-python open3d pyrender
```

install bps_torch package

In case of conflict between pytorch3d package and torch 1.7.1, try to ungrade torch to 1.6.0

## To run the train script

## To run the evaluation script

```
python eval.py
```
| Data distribution from FFHGenerator  | Filter grasps with 0.5 thresh | Filter grasps with 0.75 thresh
| --------------------------------------- | --------------------------------------- |--------------------------------------- |
| ![](docs/images/ffhgen.png)       | ![](docs/images/filter.png) | ![](docs/images/filter2.png) | 

| Filter grasps with 0.9 thresh  | Best grasp |
| --------------------------------------- | --------------------------------------- |
| ![](docs/images/filter_last.png)       | ![](docs/images/best_grasp.png) |  | 
