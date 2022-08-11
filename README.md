# Five-finger Hand Net (FFHNet)

### Variational Grasp Generation and Evaluation for the DLR-HIT Hand II

## Grasp Data Preparation

This section details the step to bring the grasping data into the right format.

1. Collect all ground truth data into one directory with folders for each recording.

```bash
    Data
      ├── 2021-03-02
      |         ├── grasp_data (folder with images of grasp)
      |         ├── grasp_data.h5 (file with all the poses)
      ...
      ├── 2021-03-10
```

2. Execute the script `grasp_pipeline/src/grasp_pipeline/grasp_data_processing/ffhnet_adapter_grasp_data.py` \
This will go through all the folders under `Data/` and combine all `ffhnet_data.h5` files into one combined `ffhnet-grasp.h5`.

3. Execute the script `grasp_pipeline/src/grasp_pipeline/grasp_data_processing/ffhnet_adapter_pcs_and_tfs`
This will go through all the objects in `ffhnet-grasp.h5` and spawn each object in `n_pcds_per_object` random positions and orientations, record a segmented point cloud observation as well as the transformation between the mesh frame of the object and the object centroid frame. All transforms get stored in `pcd_transforms.h5`.
3.1 The script also creates `metadata.csv` which contains the columns \
object_name | collision | negative | positive | train | test | val \
An `X` in train/test/val indicates that this object belongs to the training, test or val set.
A `XXX` indicates that the object should be excluded, because the data acquisition was invalid.

4. Execute the script `hithand_grasp/scripts/train_test_val_split.py` which given the `metadata.csv` file splits the data in the three folders `train`, `test`, `val`. Under each of the three folders lies a folder `point_clouds` with N `.pcd` files of objects from different angles.

5. Execute the script `bps_torch/convert_pcds_to_grabnet.py` which will first compute a BPS (basis point set) and then computes the bps representation for each object storing them in a folder `bps` under the respective object name.

## Insights into grasp data

You can now visualize individual objects and the grasps generated for them.

## To run the train script

Activate conda base environment, here you need to modify the bashrc to activate the conda environment.
