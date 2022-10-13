#!/usr/bin/env python2

import tf.transformations as tft
import transforms3d as tf
import numpy as np

# def quat_wxyz2xyzw(q):
#     sawp = q[0]
#     q = np.delete(q, 0)
#     q = np.append(q, sawp)
#     return q

# quat = np.array([0.06644604, -0.14909851, 0.98653622, -0.01004246])

# T = tft.quaternion_matrix(quat)

# quat_new = quat_xyzw2wxyz(quat)
# T_backup = tf.quaternions.quat2mat(quat_new)
# # print(T_backup)
# # print(T[:3, :3])
# # # Two functions not the same????
# assert np.allclose(T_backup, T[:3, :3])
# r = 1
# p = 1
# y = 2
# T_backup = tf.euler.euler2mat(r, p, y)
# T = tft.euler_matrix(r, p, y)
# print(T_backup)
# print(T)
# assert np.allclose(T_backup, T[:3, :3])

palm_pos_hom = np.eye(4, 4)
ori = np.array(tft.euler_from_matrix(palm_pos_hom))
ori_2 = np.array(tf.euler.mat2euler(palm_pos_hom[:3, :3]))
assert np.allclose(ori, ori_2)
