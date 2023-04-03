import argparse
import datetime
import os
import sys
import yaml

# If you want to continue training, set continue_train=True, start_epoch=desired_epoch and load_path=/path/to/pretrained
path = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.split(os.path.split(path)[0])[0]
while path[-2:] != 'vm':
    path = os.path.split(path)[0]
ROOT_PATH = path


class BaseConfig(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.cfg = None
        self.file_dir = os.path.dirname(os.path.abspath(__file__))

    def initialize(self):
        self.parser.add_argument('--batch_size',
                                 type=int,
                                 default=5012,
                                 help='Training batch size.')
        self.parser.add_argument('--continue_train',
                                 type=bool,
                                 default=False,
                                 help='Whether to continue train from prev ckpt.')
        self.parser.add_argument('--data_dir',
                                 type=str,
                                 default=os.path.join(ROOT_PATH, 'data/ffhnet-data'),
                                 help='Path to root directory of the dataset.')
        self.parser.add_argument('--gpu_ids',
                                 type=list,
                                 default=[0],
                                 help='gpu ids: e.g. [0]  [0,1,2] , [0,2]. use [-1] for CPU')
        self.parser.add_argument('--grasp_data_file_name',
                                 type=str,
                                 default='grasp_data_all.h5',
                                 help='The name of the grasp data file')
        self.parser.add_argument('--latentD',
                                 type=int,
                                 default=5,
                                 help='Dimensionality of the latent space.')
        self.parser.add_argument(
            '--load_path',
            type=str,
            default=os.path.join(BASE_PATH, 'FFHNet/checkpoints/2021-04-29T15_06_25'),
            help='If you want to load a pretrained model, from where should it be loaded?')
        self.parser.add_argument('--num_bps_per_object',
                                 type=int,
                                 default=5,
                                 help='# of bps per object.')
        self.parser.add_argument('--num_threads',
                                 type=int,
                                 default=10,
                                 help='# of threads for data loading.')
        self.parser.add_argument('--save_freq',
                                 type=int,
                                 default=5,
                                 help='# epochs between saving the model.')
        self.parser.add_argument(
            '--scale',
            type=int,
            default=1,
            help='Scaling factor to scale the entire model dimensions (n_neurons).')
        self.parser.add_argument(
            '--start_epoch',
            type=int,
            default=0,
            help='The epoch from whose end to start training and load the model.')
        self.parser.add_argument('--to_tensorboard',
                                 type=bool,
                                 default=True,
                                 help='Whether to log results to tensorboard or not.')
        self.parser.add_argument('--vis_grasp_refinement',
                                 default=False,
                                 type=bool,
                                 help='Whether to visualize the grasp refinement procedure')

    def parse(self):
        self.initialize()
        # If this gets called from ros launch file, need to hand over correct argv
        if len(sys.argv) > 1:
            if sys.argv[1] == '__name:=infer_grasp_poses':
                cfg = self.parser.parse_args(sys.argv[4:])
            else:
                raise Exception('Calling from this file is not implemented. Change base_config.py')
        else:
            cfg = self.parser.parse_args()

        ## Only do this if in training mode
        # Add a name dependent on settings
        if cfg.is_train:
            name = 'ffhnet' + '_lr_' + str(cfg.lr) + '_bs_' + str(
                cfg.batch_size) + '_scale_' + str(cfg.scale) + '_latentd_' + str(cfg.latentD)
            cfg.name = name

            # create and set checkpoints dir if training from scratch. Otherwise set load directory as ckpts directory.
            if cfg.continue_train:
                cfg.save_dir = cfg.load_path
            else:
                cfg.start_epoch = 0  # if we start training from scratch set start_epoch to 0 always
                ckpts_dir = os.path.join(BASE_PATH, 'checkpoints')
                cfg.ckpts_dir = ckpts_dir
                if not os.path.exists(ckpts_dir):
                    os.mkdir(ckpts_dir)

                # Create folder with datetime.time as name und ckpts dir
                now = datetime.datetime.now().replace(microsecond=0).isoformat().replace(':', '_')
                cfg.save_dir = os.path.join(ckpts_dir, now)
                os.mkdir(cfg.save_dir)

                # Create eval dir
                cfg.eval_dir = os.path.join(cfg.save_dir, 'eval')
                if not os.path.exists(cfg.eval_dir):
                    os.mkdir(cfg.eval_dir)

                # Save the config
                yaml_path = os.path.join(cfg.save_dir, 'config.yaml')
                with open(yaml_path, 'w') as yaml_file:
                    yaml.dump(vars(cfg), yaml_file)

        return cfg
