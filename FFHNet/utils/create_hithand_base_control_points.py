import open3d as o3d
from utils import hom_matrix_from_pos_euler_list
import numpy as np
import os


def custom_draw_geometry(pcd, orig):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(orig)
    vis.get_render_option().show_coordinate_frame = True
    vis.run()
    vis.destroy_window()


# read the mesh
curr_file_path = os.path.dirname(os.path.abspath(__file__))
base = os.path.split(os.path.split(curr_file_path)[0])[0]

path = os.path.join(base, 'hithand_palm', 'hit-hand-2-palm-right-merged_faces_300.stl')
mesh = o3d.io.read_triangle_mesh(path)

pcd = mesh.sample_points_uniformly(number_of_points=200)

orig = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

# palm link relative to baselink
# <origin xyz="0.02 0 0.06" rpy="-0.15 -1.3 0"/>
palm_T_base = hom_matrix_from_pos_euler_list([0.02, 0, 0.06, -0.15, -1.3, 0])

p1 = np.asarray(pcd.points)
print(p1)

pcd.transform(np.linalg.inv(palm_T_base))

p2 = np.asarray(pcd.points)
print(p2)
o3d.visualization.draw_geometries([pcd, orig])

p2_hom = np.ones((p2.shape[0], 4))
p2_hom[:, :3] = p2

#custom_draw_geometry(pcd, orig)
save_path = os.path.join(os.path.split(path)[0], 'hithand_palm_control_points.npy')
np.save(save_path, p2_hom)
