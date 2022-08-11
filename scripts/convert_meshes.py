import os
import pymeshlab


def convert_mesh_and_save(path, new_format='stl'):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(path)
    path_no_ending = '.'.join(path.split('.')[:-1])
    to_path = path_no_ending + '.' + new_format
    if os.path.exists(to_path):
        print('The mesh already exists')
        return
    else:
        ms.save_current_mesh(to_path)


def convert_all_hithand_meshes(base_path, from_format='dae', to_format='stl'):
    for dir_name in os.listdir(base_path):
        dir_path = os.path.join(base_path, dir_name)
        for file_name in os.listdir(dir_path):
            if file_name.split('.')[-1] == from_format:
                file_path = os.path.join(dir_path, file_name)
                convert_mesh_and_save(file_path, to_format)


if __name__ == '__main__':
    print('Hi')
