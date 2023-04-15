# Five-finger Hand Net (FFHNet)

### Variational Grasp Generation and Evaluation for the DLR-HIT Hand II

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

## Insights into grasp data

You can now visualize individual objects and the grasps generated for them.

## To run the train script

## To run the evaluation script

```
python eval.py
```
