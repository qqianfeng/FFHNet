import numpy as np
import h5py
import os
import pandas as pd
import sys
import torch
from torch.utils import data

from FFHNet.utils.grasp_data_handler import GraspDataHandlerVae
from FFHNet.utils import utils, visualization


class FFHGeneratorDataSet(data.Dataset):
    def __init__(self, cfg, dtype=torch.float32):
        super(FFHGeneratorDataSet, self).__init__()
        self.cfg = cfg
        self.dtype = dtype
        self.num_bps_per_obj = cfg.num_bps_per_object

        self.ds_path = os.path.join(cfg.data_dir, cfg.ds_name)
        self.objs_names = self.get_objs_names(self.ds_path)
        self.objs_folder = os.path.join(self.ds_path, 'bps')
        self.grasp_data_path = os.path.join(cfg.data_dir, cfg.grasp_data_file_name)
        self.gazebo_obj_path = cfg.gazebo_obj_path

        self.grasp_data_handler = GraspDataHandlerVae(self.grasp_data_path)
        df = pd.read_csv(os.path.join(cfg.data_dir, 'metadata.csv'))
        df_name_pos = df[df[cfg.ds_name] == 'X'].loc[:, ['Unnamed: 0', 'positive']]
        self.num_success_per_object = dict(
            zip(df_name_pos.iloc[:, 0], df_name_pos.iloc[:, 1].astype('int64')))
        self.bps_paths, self.grasp_idxs = self.get_all_bps_paths_and_grasp_idxs(
            self.objs_folder, self.num_success_per_object)

        self.is_debug = False
        if self.is_debug:
            print("The size in KB is: ", sys.getsizeof(self.bps_paths) / 1000)

    def get_objs_names(self, path):
        objs_folder = os.path.join(path, 'pcd')
        return [obj for obj in os.listdir(objs_folder) if '.' not in obj]

    def verify_num_bps_per_object(self):
        """ Counts the number of bps for one object. Assumes all objects have identical num of bps.
        """
        for obj in self.objs_names:
            obj_bps_folder = os.path.join(self.objs_folder, obj)
            bps_names = [bps for bps in os.listdir(obj_bps_folder) if 'bps' in bps]
            if len(bps_names) != self.cfg.num_bps_per_object:
                return False
        return True

    def get_all_bps_paths_and_grasp_idxs(self, objs_folder, success_per_obj_dict):
        """ Creates a long list where each of the N bps per object get repeated as many times
        as there are positive grasps for this object. It also returns a list of indexes with the same length as the bps list
        indicating the grasp index. This way each bps is uniquely belonging to each valid grasp ONCE

        Args:
            objs_folder (str, path): The path to the folder where the bps lie.
            num_success_per_object (dict): A dict with keys being all the object names in the current dataset and values the successful grasps for each object.

        Returns:
            paths (list): List of bps file paths. Each bps occuring as often as there are positive grasps for each object.
            grasp_idxs (list): List of ranges from 0 to n_success_grasps per object and bps.
        """
        paths = []
        grasp_idxs = []
        for obj, n_success in success_per_obj_dict.items():
            obj_path = os.path.join(objs_folder, obj)
            for f_name in os.listdir(obj_path):
                f_path = os.path.join(obj_path, f_name)
                if 'bps' in os.path.split(f_name)[1]:
                    paths += n_success * [f_path]
                    grasp_idxs += range(0, n_success)

        assert len(paths) == len(grasp_idxs)
        return paths, grasp_idxs

    def read_pcd_transform(self, bps_path):
        # pcd save path from bps save path
        base_path, bps_name = os.path.split(bps_path)
        pcd_name = bps_name.replace('bps', 'pcd')
        pcd_name = pcd_name.replace('.npy', '.pcd')
        path = os.path.join(base_path, pcd_name)

        # Extract object name from path
        head, pcd_file_name = os.path.split(path)
        pcd_name = pcd_file_name.split('.')[0]
        obj = os.path.split(head)[1]

        # Read the corresponding transform in
        path = os.path.join(os.path.split(self.ds_path)[0], 'pcd_transforms.h5')
        with h5py.File(path, 'r') as hdf:
            pos_quat_list = hdf[obj][pcd_name + '_mesh_to_centroid'][()]

        # Transform the transform to numpy 4*4 array
        hom_matrix = utils.hom_matrix_from_pos_quat_list(pos_quat_list)
        return hom_matrix

    def __getitem__(self, idx):
        """ Batch contains: N random different object bps, each one successful grasp

        Dataset size = total_num_successful_grasps * N_bps_per_object, e.g. 15.000 * 50 = 750k

        Should fetch one bps for an object + a single grasp for that object.
        Returns a dict with palm_position, palm_orientation, finger_configuration and bps encoding of the object.
        """
        bps_path = self.bps_paths[idx]
        # Load the bps encoding
        base_path, bps_name = os.path.split(bps_path)
        obj_name = '_'.join(bps_name.split('_bps')[:-1])
        bps_obj = np.load(bps_path)

        # Read the corresponding transform between mesh_frame and object_centroid
        centr_T_mesh = self.read_pcd_transform(bps_path)

        # Read in a grasp for a given object (in mesh frame)
        palm_pose, joint_conf = self.grasp_data_handler.get_single_successful_grasp(obj_name,
                                                                                    random=True)
        palm_pose_hom = utils.hom_matrix_from_pos_quat_list(palm_pose)

        # Transform grasp from mesh frame to object centroid
        palm_pose_centr = np.matmul(centr_T_mesh, palm_pose_hom)

        # Before reducing joint conf
        if self.is_debug:
            j = joint_conf
            diffs = np.abs([j[3] - j[2], j[7] - j[6], j[11] - j[10], j[15] - j[14]])
            print(diffs[diffs > 0.09])

        # Turn the full 20 DoF into 15 DoF as every 4th joint is coupled with the third
        joint_conf = utils.reduce_joint_conf(joint_conf)

        # Extract rotmat and transl
        palm_rot_matrix = palm_pose_centr[:3, :3]
        palm_transl = palm_pose_centr[:3, 3]

        # Test restored grasp
        if self.is_debug:
            print(joint_conf)
            print(palm_transl)
            visualization.show_dataloader_grasp(bps_path, obj_name, centr_T_mesh, palm_pose_hom,
                                                palm_pose_centr, self.gazebo_obj_path)

            # Visualize full hand config
            visualization.show_grasp_and_object(bps_path, palm_pose_centr, joint_conf)

        # Build output dict
        data_out = {'rot_matrix': palm_rot_matrix,\
                    'transl': palm_transl,\
                    'joint_conf': joint_conf,\
                    'bps_object': bps_obj}

        # If we want to evaluate, also return the pcd path to load from for vis
        #if self.cfg.ds_name == 'eval':
        data_out['pcd_path'] = bps_path.replace('bps', 'pcd').replace('npy', 'pcd')
        data_out['obj_name'] = obj_name

        return data_out

    def __len__(self):
        #len of dataset is number of bps per object x num_success_grasps
        return len(self.bps_paths)


if __name__ == '__main__':
    from FFHNet.config.train_config import TrainConfig
    cfg = TrainConfig().parse()
    gds = FFHGeneratorDataSet(cfg)

    while True:
        i = np.random.randint(0, gds.__len__())
        gds.__getitem__(i)