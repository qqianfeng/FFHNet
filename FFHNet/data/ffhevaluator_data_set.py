import numpy as np
import h5py
import os
import pandas as pd
import torch
from torch.utils import data

from FFHNet.utils.grasp_data_handler import GraspDataHandlerVae
from FFHNet.utils import utils, visualization


class FFHEvaluatorDataSet(data.Dataset):
    def __init__(self, cfg, dtype=torch.float32):
        super(FFHEvaluatorDataSet, self).__init__()
        self.cfg = cfg
        self.dtype = dtype

        self.pos_ratio = 0.3
        self.neg_ratio = 0.3
        self.hard_negative_ratio = 0.4

        self.ds_path = os.path.join(cfg.data_dir, cfg.ds_name)
        self.objs_names = self.get_objs_names(self.ds_path)
        self.objs_folder = os.path.join(self.ds_path, 'bps')
        grasp_data_path = os.path.join(cfg.data_dir, cfg.grasp_data_file_name)
        self.gazebo_obj_path = cfg.gazebo_obj_path
        self.grasp_data_handler = GraspDataHandlerVae(grasp_data_path)

        # Info about dataset from csv
        df = pd.read_csv(os.path.join(cfg.data_dir, 'metadata.csv'))
        df_name_pos = df[df[cfg.ds_name] == 'X'].loc[:, ['Unnamed: 0', 'positive']]
        self.num_success_per_object = dict(
            zip(df_name_pos.iloc[:, 0], df_name_pos.iloc[:, 1].astype('int64')))

        # Build paths list and corr. labels
        self.bps_paths, self.labels = self.get_all_bps_paths_and_labels(
            self.objs_folder, self.num_success_per_object)

        self.debug = False

    def get_objs_names(self, ds_path):
        objs_folder = os.path.join(ds_path, 'bps')
        return [obj for obj in os.listdir(objs_folder) if '.' not in obj]

    def get_all_bps_paths_and_labels(self, objs_folder, success_per_obj_dict):
        """ Creates a long list of paths to bps object files. Each bps for a respective object gets repeated
        as many times as there are success grasps. Based on this it is repeated for negative and collision grasps. E.g. if we have pos_ratio=0.3,
        neg_ratio=0.3, hard_negative_ratio=0.4 and for one object we have 100 pos grasps, then each of the objects bps files get repeated, 100 times for pos grasps
        100 times for neg grasps and 133 times for collision grasps.

        Args:
            obj_folder (string): The path to the folder where the BPS lie
            success_per_obj_dict (dict): A dict with one key for each object and val being the amount of success grasps per object

        Returns:
            bps_paths (list of strings): Long list of BPS paths.
            labels (list of strings): Indicating whether the grasp was successful, unsuccessful or in collision
        """
        paths = []
        labels = []
        for obj, n_success in success_per_obj_dict.items():
            n_total = n_success // self.pos_ratio
            obj_path = os.path.join(objs_folder, obj)
            for f_name in os.listdir(obj_path):
                f_path = os.path.join(obj_path, f_name)
                if 'bps' in os.path.split(f_name)[1]:
                    # Paths
                    paths += n_success * [f_path]
                    paths += int(self.neg_ratio * n_total) * [f_path]
                    paths += int(self.hard_negative_ratio * n_total) * [f_path]
                    # Labels
                    labels += n_success * ['positive']
                    labels += int(self.neg_ratio * n_total) * ['negative']
                    labels += int(self.hard_negative_ratio * n_total) * ['hard_negative']

        assert len(paths) == len(labels)
        return paths, labels

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
        bps_path = self.bps_paths[idx]
        label = self.labels[idx]
        success = 1 if label == 'positive' else 0

        # Load the bps encoding
        base_path, bps_name = os.path.split(bps_path)
        obj_name = '_'.join(bps_name.split('_bps')[:-1])
        bps_obj = np.load(bps_path)

        # Read the corresponding transform between mesh_frame and object_centroid
        centr_T_mesh = self.read_pcd_transform(bps_path)

        # Read in a grasp for a given object (in mesh frame). If the label is hard-negative, read a positive one and shift it
        outcome = label if label != 'hard_negative' else 'positive'
        palm_pose, joint_conf = self.grasp_data_handler.get_single_grasp_of_outcome(
            obj_name, outcome=outcome, random=True)
        palm_pose_hom = utils.hom_matrix_from_pos_quat_list(palm_pose)

        # Transform grasp from mesh frame to object centroid. If label is hard_negative perturb the positive grasp sufficiently
        # by adding +- 3cm and 0.6 rad in each dimension
        palm_pose_centr = np.matmul(centr_T_mesh, palm_pose_hom)
        if label == 'hard_negative':
            palm_pose_centr = utils.hard_negative_from_positive(palm_pose_centr)

        # Turn the full 20 DoF into 15 DoF as every 4th joint is coupled with the third
        joint_conf = utils.reduce_joint_conf(joint_conf)

        # Extract rotmat and transl
        palm_rot_matrix = palm_pose_centr[:3, :3]
        palm_transl = palm_pose_centr[:3, 3]

        # Test visualize grasp
        if self.debug:
            print(palm_transl)
            print(joint_conf)
            print("label", label)
            visualization.show_dataloader_grasp(bps_path, obj_name, centr_T_mesh, palm_pose_hom,
                                                palm_pose_centr, self.gazebo_obj_path)
            # Visualize full hand config
            visualization.show_grasp_and_object(bps_path, palm_pose_centr, joint_conf)

        # Build output dict
        data_out = {'rot_matrix': palm_rot_matrix,\
                    'transl': palm_transl,\
                    'joint_conf': joint_conf,\
                    'bps_object': bps_obj,\
                    'label': success}

        # If we want to evaluate, also return the pcd path to load from for vis
        if self.cfg.ds_name == 'eval':
            data_out['pcd_path'] = bps_path.replace('bps', 'pcd').replace('npy', 'pcd')
            data_out['obj_name'] = obj_name

        return data_out

    def __len__(self):
        return len(self.bps_paths)


if __name__ == '__main__':
    from FFHNet.config.train_config import TrainConfig
    cfg = TrainConfig().parse()
    gds = FFHEvaluatorDataSet(cfg)

    while True:
        i = np.random.randint(0, gds.__len__())
        gds.__getitem__(i)