import copy
import h5py
import numpy as np
import open3d as o3d
import os
import sys
import torch

if sys.version[0] == '3':
    import bps_torch.bps as b_torch
elif sys.version[0] == '2':
    import tf.transformations as tft

from FFHNet.utils.definitions import HAND_CFG


def class_labels_from_logits(logits, threshold):
    logits[logits >= threshold] = 1.
    logits[logits < threshold] = 0.

    return logits


# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  #batch*3

    return out


def control_points_from_transl_rot_matrix(transl, rot_matrix, device):
    """ Applys the given 3D translation and batch*3*3 rotation matrix to point cloud of hithand base
    and outputs the transformed point cloud.
    """
    batch_size = transl.shape[0]

    # Load control points and create tensor
    control_points_np = np.load('../..hithand_palm/hithand_palm_control_points.npy')  # N_points*3
    n_points = control_points_np.shape[0]
    control_points = torch.Tensor(control_points_np, device=device, dtype=torch.float32)
    control_points = control_points.expand(batch_size, control_points.shape[0],
                                           control_points.shape[1])  # batch_size * N_points * 3
    assert control_points.shape == (batch_size, n_points, 3)

    # Create tensor transformation matrix
    T = hom_matrix_batch_from_transl_rot_matrix(transl, rot_matrix)  # batch_size * 4 * 4
    T = T.to(device=device, dtype=torch.float32)
    assert T.shape == (batch_size, 4, 4)

    # Apply translformations to control points
    control_points_tfd = torch.matmul(T, control_points)
    assert control_points_tfd.shape == (batch_size, n_points, 3)

    return control_points_tfd


def data_dict_to_dtype(data, dtype):
    for k, v in data.items():
        if not isinstance(v, list):
            data[k] = v.to(dtype)
    return data


def dict_to_tensor(in_dict, device, dtype):
    out_dict = {}
    for k, v in in_dict.items():
        out_dict[k] = torch.tensor(v, dtype=dtype, device=device)
    return out_dict


def display_inlier_outlier(pcd, ind):
    pcd_in = pcd.select_down_sample(ind)
    pcd_out = pcd.select_down_sample(ind, invert=True)
    pcd_in.paint_uniform_color([0.5, 0.5, 0.5])
    pcd_out.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([pcd_in, pcd_out])


def encode_pcd_with_bps(pcd, save_path, bps_path='/home/vm/data/ffhnet-data/basis_point_set.npy'):
    bps_np = np.load(bps_path)
    bps = b_torch.bps_torch(custom_basis=bps_np)
    pcd_np = np.asarray(pcd.points)
    enc_dict = bps.encode(pcd_np)
    enc_np = enc_dict['dists'].cpu().detach().numpy()
    np.save(save_path, enc_np)


def filter_pcd_workspace_boundaries(pcd, x_min, x_max, y_min, y_max, z_min, z_max):
    points, colors = np.asarray(pcd.points), np.asarray(pcd.colors)
    # filter x direction
    mask = np.logical_and(points[:, 0] > x_min, points[:, 0] < x_max)
    points, colors = points[mask], colors[mask]

    # filter y direction
    mask = np.logical_and(points[:, 1] > y_min, points[:, 1] < y_max)
    points, colors = points[mask], colors[mask]

    # filter z direction
    mask = np.logical_and(points[:, 2] > z_min, points[:, 2] < z_max)
    points, colors = points[mask], colors[mask]

    # new pcd
    del pcd
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def full_joint_conf_from_partial_joint_conf(partial_joint_conf):
    """Takes in the 15 dimensional joint conf output from VAE and repeats the 3*N-th dimension to turn dim 15 into dim 20.

    Args:
        partial_joint_conf (np.array): Output from vae with dim(partial_joint_conf.position) = 15

    Returns:
        full_joint_conf (np.array): Full joint state with dim(full_joint_conf.position) = 20
    """
    full_joint_pos = 20 * [0]
    ix_full_joint_pos = 0
    for i, val in enumerate(partial_joint_conf):
        if (i + 1) % 3 == 0:
            full_joint_pos[ix_full_joint_pos] = val
            full_joint_pos[ix_full_joint_pos + 1] = val
            ix_full_joint_pos += 2
        else:
            full_joint_pos[ix_full_joint_pos] = val
            ix_full_joint_pos += 1

    full_joint_conf = full_joint_pos
    return full_joint_conf


def get_hand_cfg_map(cfg_arr):
    cfg_map = HAND_CFG
    keys = sorted(HAND_CFG.keys())
    for idx, k in enumerate(keys):
        cfg_map[k] = cfg_arr[idx]
    return cfg_map


def grasp_numpy_from_data_dict(data):
    return {
        'rot_matrix': data['rot_matrix'].cpu().data.numpy(),
        'transl': data['transl'].cpu().data.numpy(),
        'joint_conf': data['joint_conf'].cpu().data.numpy()
    }


def hom_matrix_from_pos_euler_list(pos_euler_list):
    pos = pos_euler_list[:3]
    r, p, y = pos_euler_list[3:]
    T = tft.euler_matrix(r, p, y)
    T[:3, 3] = pos
    return T


def hard_negative_from_positive(palm_pos_hom):
    """Transforms a positive grasp into a negative one by perturbing it sufficiently.

    Args:
        palm_pos_hom (4x4 array): The positive grasp as hom transform.
        
    Returns:
        palm_hneg_hom (4x4 array):
    """
    dist_vec = np.array([0.03, 0.03, 0.03, 0.6, 0.6, 0.6])  #disturb by 0.03 cm and by 0.6 rad

    ori = np.array(tft.euler_from_matrix(palm_pos_hom))
    pos_ori = np.concatenate([palm_pos_hom[:3, 3], ori])

    rand_sign = np.random.random(6)
    rand_sign[rand_sign < 0.5] = -1
    rand_sign[rand_sign >= 0.5] = 1

    pos_ori_d = pos_ori + rand_sign * dist_vec

    palm_hneg_hom = tft.euler_matrix(pos_ori_d[3], pos_ori_d[4], pos_ori_d[5])
    palm_hneg_hom[:3, 3] = pos_ori_d[:3]

    return palm_hneg_hom


def hom_matrix_from_pos_quat_list(rot_quat_list):
    p = rot_quat_list[:3]
    q = rot_quat_list[3:]
    T = tft.quaternion_matrix(q)
    T[:3, 3] = p
    return T


def hom_matrix_from_transl_rot_matrix(transl, rot_matrix):
    """Transform rot_matrix and transl vector into 4x4 homogenous transform.

    Args:
        transl (array): Translation array 3x1
        rot_matrix (array): Rotation matrix 3x3

    Returns:
        hom_matrix (array): 4x4 homogenous transform.
    """
    hom_matrix = np.eye(4)
    hom_matrix[:3, :3] = rot_matrix
    hom_matrix[:3, 3] = transl
    return hom_matrix


def hom_matrix_batch_from_transl_rot_matrix(transl, rot_matrix, tensors=True):
    """ 
    args:    
        transl [batch_size*3]
        rot_matrix [batch_size*3*3]

    returns:
        T [batch_size*4*4]
    """
    batch_size = transl.shape[0]
    if tensors:
        assert torch.is_tensor(transl) and torch.is_tensor(rot_matrix)
        T = torch.eye(4)
        T = T.expand(batch_size, 4, 4).contiguous()

        T[:, :3, :3] = rot_matrix
        T[:, :3, 3] = transl
    else:
        assert isinstance(transl, np.ndarray) and isinstance(rot_matrix, np.ndarray)
        T = np.eye(4)
        T = np.tile(T, (batch_size, 1, 1))
        T[:, :3, :3] = rot_matrix
        T[:, :3, 3] = transl
    # Transformt the tensors trans [batch_size*3], rot_matri* [batch_size*3*3] into homogenous transforms [batch_size*4*4]
    print("Verify this function")
    return T


def load_rendered_pcd(path, k_down=1):
    if 'bps' in path:
        path = path.replace('bps', 'pcd')
        path = path.replace('.npy', '.pcd')
    rendered_pc = o3d.io.read_point_cloud(path)
    rendered_pc.paint_uniform_color(np.array([1, 0, 0]))

    # Display less points of the rendered point cloud
    rendered_pc = rendered_pc.uniform_down_sample(k_down)
    return rendered_pc


# batch*n
def normalize_vector(v, return_mag=False):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    if (return_mag == True):
        return v, v_mag[:, 0]
    else:
        return v


def print_grasp_file():
    with h5py.File('/home/vm/data/exp_data/2021-03-16/grasp_data.h5', 'r') as f:
        grasps_gp = f['recording_sessions']['recording_session_0003']['grasp_trials'][
            'bigbird_dove_go_fresh_burst']['grasps']['no_collision']
        for k in grasps_gp:
            grasp = grasps_gp[k]
            print('Desired', grasp['desired_preshape_joint_state'][()])
            print('True', grasp['true_preshape_joint_state'][()])
            print('Lifted', grasp['lifted_joint_state'][()])
            print('Closed', grasp['closed_joint_state'][()])

    with h5py.File('/home/vm/data/ffhnet-data/ffhnet-grasp.h5', 'r') as f:
        obj_gp = f[f.keys()[12]]
        pos_gp = obj_gp['positive']
        print(len(pos_gp.keys()))
        for k in pos_gp.keys():
            grasp = pos_gp[k]
            print('Desired', grasp['desired_preshape_joint_state'][()])
            print('True', grasp['true_preshape_joint_state'][()])


def reduce_joint_conf(jc_full):
    """Turn the 20 DoF input joint array into 15 DoF by either dropping each 3rd or 4th joint value, depending on which is smaller.

    Args:
        jc_full (np array): 20 dimensional array of hand joint values

    Returns:
        jc_red (np array): 15 dimensional array of reduced hand joint values
    """
    idx = 0
    jc_red = np.zeros((15, ))
    for i, _ in enumerate(jc_red):
        if (i + 1) % 3 == 0:
            if jc_full[idx + 1] > jc_full[idx]:
                jc_red[i] = jc_full[idx + 1]
            else:
                jc_red[i] = jc_full[idx]
            idx += 2
        else:
            jc_red[i] = jc_full[idx]
            idx += 1
    return jc_red


def rot_matrix_from_ortho6d(ortho6d, dtype=torch.float32):
    x_raw = ortho6d[:, 0:3]  #batch*3
    y_raw = ortho6d[:, 3:6]  #batch*3

    x = normalize_vector(x_raw)  #batch*3
    z = cross_product(x, y_raw)  #batch*3
    z = normalize_vector(z)  #batch*3
    y = cross_product(z, x)  #batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  #batch*3*3
    matrix.to(dtype)
    return matrix


def filter_pcd_manually():
    path = '/home/vm/data/real_objects/object/mustard_bottle_02_need_filt.pcd'
    pcd = o3d.io.read_point_cloud(path)

    orig = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    o3d.visualization.draw_geometries([pcd, orig])

    pcd = filter_pcd_workspace_boundaries(pcd, -0.032, 1, -1, 1, -1, 1)
    o3d.visualization.draw_geometries([pcd, orig])
    _, inliers = pcd.remove_radius_outlier(150, 0.02)
    display_inlier_outlier(pcd, inliers)
    pcd = pcd.select_down_sample(inliers)
    o3d.visualization.draw_geometries([pcd, orig])

    o3d.io.write_point_cloud(path.replace('_need_filt', ''), pcd)


if __name__ == '__main__':
    palm_hard_neg = hard_negative_from_positive(np.eye(4))
    # base_path = '/home/vm/data/real_objects'
    # path_objs = os.path.join(base_path, 'object')
    # path_bps = os.path.join(base_path, 'bps')
    # for f_name in os.listdir(path_objs):
    #     pcd_path = os.path.join(path_objs, f_name)
    #     pcd = o3d.io.read_point_cloud(pcd_path)

    #     f_name = f_name.replace('.pcd', '.npy')
    #     bps_save_path = os.path.join(path_bps, f_name)
    #     encode_pcd_with_bps(pcd, bps_save_path)
