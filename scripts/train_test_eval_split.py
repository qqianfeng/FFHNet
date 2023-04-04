""" This script uses the information about which objects belong to the train/test/val
datasets in the metadata.csv file to split all the obejct pointclouds into their respective folders.
"""
import os
import pandas as pd
import shutil


def main(metadata_csv_path, src_pc_path, dst_base_path):
    """
    Args:
        metadata_csv_path (str): generated csv file path
        src_pc_path (_type_): source point cloud data folder path
        dst_base_path (_type_): destination to store the data
    """
    df = pd.read_csv(metadata_csv_path)
    for split in ['train', 'test', 'eval']:
        # Get the object names for this split
        obj_names = list(df[df[split] == 'X'].iloc[:, 0].real)
        # Create dst folder
        dst_split_pc_folder = os.path.join(dst_base_path, split, 'point_clouds')
        if not os.path.exists(dst_split_pc_folder):
            os.makedirs(dst_split_pc_folder)
        # copy each object from src to dst
        for obj in obj_names:
            src_obj_folder = os.path.join(src_pc_path, obj)
            dst_obj_folder = os.path.join(dst_split_pc_folder, obj)
            #os.mkdir(dst_obj_folder)
            # copy entire src folder with files to dst
            shutil.copytree(src_obj_folder, dst_obj_folder)
