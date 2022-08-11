from __future__ import division
import numpy as np
import os
import time
import torch
from FFHNet.utils.train_tools import EarlyStopping
from FFHNet.utils import utils

import FFHNet.models.losses as losses
import FFHNet.models.networks as networks


class FFHNet(object):
    """ Wrapper which houses the network blocks, the losses and training logic.
    """
    def __init__(self, cfg):
        self.dtype = torch.float32
        torch.set_default_dtype(self.dtype)
        self.cfg = cfg
        self.is_train = cfg.is_train

        if torch.cuda.is_available:
            self.device = torch.device('cuda:{}'.format(cfg.gpu_ids[0]))
            torch.cuda.empty_cache()
        else:
            self.device = torch.device('cpu')

        # model, optimizer, scheduler_ffhgenerator, losses
        self.FFHGenerator, self.FFHEvaluator = networks.build_ffhnet(cfg, is_train=self.is_train)

        if self.is_train:
            self.bce_weight = 10.
            self.kl_coef = cfg.kl_coef
            self.transl_coef = 100.
            self.rot_coef = 1.
            self.conf_coef = 10.
            self.train_ffhgenerator = cfg.train_ffhgenerator
            self.train_ffhevaluator = cfg.train_ffhevaluator
            self.optim_ffhgenerator = torch.optim.Adam(self.FFHGenerator.parameters(),
                                                       lr=cfg.lr,
                                                       betas=(cfg.beta1, 0.999),
                                                       weight_decay=cfg.weight_decay)
            self.optim_ffhevaluator = torch.optim.Adam(self.FFHEvaluator.parameters(),
                                                       lr=cfg.lr,
                                                       betas=(cfg.beta1, 0.999),
                                                       weight_decay=cfg.weight_decay)
            self.scheduler_ffhgenerator = networks.get_scheduler(self.optim_ffhgenerator, cfg)
            self.scheduler_ffhevaluator = networks.get_scheduler(self.optim_ffhevaluator, cfg)
            self.estop_ffhgenerator = EarlyStopping()
            self.estop_ffhevaluator = EarlyStopping()

        self.kl_loss, self.rec_pose_loss = networks.define_losses('transl_rot_6D_l2')
        self.L2_loss = torch.nn.MSELoss(reduction='mean')
        self.BCE_loss = torch.nn.BCELoss(reduction='mean')

        self.compute_eva_accuracy = losses.accuracy_evaluator

        # Wrap models if we use multi-gpu
        if len(cfg.gpu_ids) > 1:
            self.FFHGenerator = torch.nn.DataParallel(self.FFHGenerator, device_ids=cfg.gpu_ids)
            self.FFHEvaluator = torch.nn.DataParallel(self.FFHEvaluator, device_ids=cfg.gpu_ids)
        # count params
        ffhgenerator_vars = [var[1] for var in self.FFHGenerator.named_parameters()]
        ffhgenerator_n_params = sum(p.numel() for p in ffhgenerator_vars if p.requires_grad)
        print("The ffhgenerator has {:2.2f} parms".format(ffhgenerator_n_params))
        ffhevaluator_vars = [var[1] for var in self.FFHEvaluator.named_parameters()]
        ffhevaluator_n_params = sum(p.numel() for p in ffhevaluator_vars if p.requires_grad)
        print("The ffhevaluator has {:2.2f} parms".format(ffhevaluator_n_params))

        self.file_path = os.path.dirname(os.path.abspath(__file__))
        self.logit_thresh = 0.5

    def compute_loss_ffhevaluator(self, pred_success_p):
        """Computes the binary cross entropy loss between predicted success-label and true success"""
        bce_loss_val = self.bce_weight * self.BCE_loss(pred_success_p, self.FFHEvaluator.gt_label)
        loss_dict = {'total_loss_eva': bce_loss_val, 'bce_loss': bce_loss_val}
        return bce_loss_val, loss_dict

    def compute_loss_ffhgenerator(self, data_rec):
        """ The model should output a 6D representation of a rotation, which then gets mapped back to 
        """
        # KL loss
        kl_loss_val = self.kl_loss(data_rec["mu"], data_rec["logvar"])

        # Pose loss, translation rotation
        gt_transl_rot_matrix = {
            'transl': self.FFHGenerator.transl,
            'rot_matrix': self.FFHGenerator.rot_matrix
        }
        transl_loss_val, rot_loss_val = self.rec_pose_loss(data_rec, gt_transl_rot_matrix,
                                                           self.L2_loss, self.device)

        # Loss on joint angles
        conf_loss_val = self.L2_loss(data_rec['joint_conf'], self.FFHGenerator.joint_conf)

        # Put all losses in one dict and weigh them individually
        loss_dict = {
            'kl_loss': self.kl_coef * kl_loss_val,
            'transl_loss': self.transl_coef * transl_loss_val,
            'rot_loss': self.rot_coef * rot_loss_val,
            'conf_loss': self.conf_coef * conf_loss_val
        }
        total_loss = self.kl_coef * kl_loss_val + self.transl_coef * transl_loss_val + self.rot_coef * rot_loss_val + self.conf_coef * conf_loss_val
        loss_dict["total_loss_gen"] = total_loss

        return total_loss, loss_dict

    def eval_ffhevaluator_accuracy(self, data):
        logits = self.FFHEvaluator(data)  # network outputs logits

        # Turn the output logits into class labels. logits > thresh = 1, < thresh = 0
        pred_label = utils.class_labels_from_logits(logits, self.logit_thresh)

        # Compute the accuracy for positive and negative class
        pos_acc, neg_acc = self.compute_eva_accuracy(pred_label, self.FFHEvaluator.gt_label)

        # Turn the raw predictions into np arrays and return for confusion matrix
        pred_label_np = pred_label.detach().cpu().numpy()
        gt_label_np = self.FFHEvaluator.gt_label.detach().cpu().numpy()

        return pos_acc, neg_acc, pred_label_np, gt_label_np

    def eval_ffhevaluator_loss(self, data):
        self.FFHEvaluator.eval()

        with torch.no_grad():
            out = self.FFHEvaluator(data)
            _, loss_dict_ffhevaluator = self.compute_loss_ffhevaluator(out)

        return loss_dict_ffhevaluator

    def eval_ffhgenerator_loss(self, data):
        self.FFHGenerator.eval()

        with torch.no_grad():
            data_rec = self.FFHGenerator(data)
            _, loss_dict_ffhgenerator = self.compute_loss_ffhgenerator(data_rec)

        return loss_dict_ffhgenerator

    def evaluate_grasps(self, bps, grasps, thresh=0.5, return_arr=True):
        """Receives n grasps together with bps encodings of queried object and evaluates the probability of success.

        Args:
            bps (np array): [description]
            grasps (dict): Dict holding the grasp information 
            thresh (float, optional): Reject grasps with lower success p than this. Defaults to 0.5.

        Returns:
            p_success [tensor or arr, n_samples*1]: Success probability of each grasp
        """
        n_samples = grasps['rot_matrix'].shape[0]
        if len(bps.shape) > 1:
            bps = bps.squeeze()
        bps = np.tile(bps, (n_samples, 1))

        grasps['bps_object'] = bps
        grasps_t = utils.dict_to_tensor(grasps, device=self.device, dtype=self.dtype)

        p_success = self.FFHEvaluator(grasps_t).squeeze()

        if return_arr:
            p_success = p_success.cpu().detach().numpy()

        return p_success

    def filter_grasps(self, bps, grasps, thresh=0.5, return_arr=True):
        """ Takes in grasps generated by the FFHGenerator for a bps encoding of an object and removes every grasp#
        with predicted success probability less than thresh 

        Args:
            bps (np array): Bps encoding of the object. n*4096
            grasps (dict): Dict holding the grasp information. keys: transl n*3, rot_matrix n*3*3, joint_conf n*15
            thresh (float, optional): Reject grasps with lower success p than this. Defaults to 0.5.
        """
        start = time.time()
        n_samples = grasps['rot_matrix'].shape[0]
        if len(bps.shape) > 1:
            bps = bps.squeeze()
        bps = np.tile(bps, (n_samples, 1))

        grasps['bps_object'] = bps
        grasps_t = utils.dict_to_tensor(grasps, device=self.device, dtype=self.dtype)

        p_success = self.FFHEvaluator(grasps_t).squeeze()

        filt_grasps = {}
        for k, v in grasps_t.items():
            filt_grasps[k] = v[p_success > thresh]
            if return_arr:
                filt_grasps[k] = filt_grasps[k].cpu().detach().numpy()
        #print("Filtering took: %.4f" % (time.time() - start))
        return filt_grasps

    def generate_grasps(self, bps, n_samples, return_arr=True):
        """Samples n grasps either from combining given bps encoding with z or sampling from random normal distribution.

        Args:
            bps (np arr) 1*4096: BPS encoding of the segmented object point cloud.
            n_samples (int): How many samples sould be generated. 
            return_arr (bool): Whether to return results as np arr or tensor.

        Returns:
            rot_matrix (tensor or array) n_samples*3*3: palm rotation matrix
            transl (tensor or array) n_samples*3: 3D palm translation
            joint_conf (tensor or array) n_samples*15: 15 dim finger configuration
        """
        # turn np arr to tensor and repeat n_samples times
        if len(bps.shape) > 1:
            bps = bps.squeeze()
        bps = np.tile(bps, (n_samples, 1))
        bps_tensor = torch.tensor(bps, dtype=self.dtype, device=self.device)

        return self.FFHGenerator.generate_poses(bps_tensor, return_arr=return_arr)

    def improve_grasps_gradient_based(self, data, last_success):
        """Apply small gradient steps to improve an initial grasp.

        Args:
            data (dict): Keys being bps_object, rot_matrix, transl, joint_conf describing sensor observation and grasp pose.
            last_success (None): Only exists to have the same interface as sampling-based refinement.

        Returns:
            p_success (tensor): success probability of grasps in data.
        """
        p_success = self.FFHEvaluator(data)
        p_success.squeeze().backward(torch.ones(p_success.shape[0]).to(self.device))

        diff = np.abs(p_success.cpu().data.numpy().squeeze() -
                      data['label'].cpu().data.numpy().squeeze())

        # Adjust the alpha so that it won't update more than 1 cm. Gradient is only valid in small neighborhood.
        norm_transl = torch.norm(data['transl'].grad, p=2, dim=-1).to(self.device)
        alpha = torch.min(0.01 / norm_transl, torch.tensor(1.0, device=self.device))

        # Take a small step on each variable
        data['transl'].data += data['transl'].grad * alpha[:, None]
        data['rot_matrix'].data += data['rot_matrix'].grad * alpha[:, None, None]
        data['joint_conf'].data += data['joint_conf'].grad * alpha[:, None]

        return p_success.squeeze(), None

    def improve_grasps_sampling_based(self, pcs, grasp_eulers, grasp_trans, last_success=None):
        with torch.no_grad():
            if last_success is None:
                grasp_pcs = utils.control_points_from_rot_and_trans(grasp_eulers, grasp_trans,
                                                                    self.device)
                last_success = self.grasp_evaluator.eval_ffhevaluator_accuracy(pcs, grasp_pcs)

            delta_t = 2 * (torch.rand(grasp_trans.shape).to(self.device) - 0.5)
            delta_t *= 0.02
            delta_euler_angles = (torch.rand(grasp_eulers.shape).to(self.device) - 0.5) * 2
            perturbed_translation = grasp_trans + delta_t
            perturbed_euler_angles = grasp_eulers + delta_euler_angles
            grasp_pcs = utils.control_points_from_rot_and_trans(perturbed_euler_angles,
                                                                perturbed_translation, self.device)

            perturbed_success = self.grasp_evaluator.eval_ffhevaluator_accuracy(pcs, grasp_pcs)
            ratio = perturbed_success / torch.max(last_success,
                                                  torch.tensor(0.0001).to(self.device))

            mask = torch.rand(ratio.shape).to(self.device) <= ratio

            next_success = last_success
            ind = torch.where(mask)[0]
            next_success[ind] = perturbed_success[ind]
            grasp_trans[ind].data = perturbed_translation.data[ind]
            grasp_eulers[ind].data = perturbed_euler_angles.data[ind]
            return last_success.squeeze(), next_success

    def load_ffhnet(self, epoch):
        self.load_ffhevaluator(epoch)
        self.load_ffhgenerator(epoch)

    def load_ffhevaluator(self, epoch, load_path=None):
        """Load ffhevaluator from disk and set to eval or train mode.
        """
        if epoch == -1:
            path = os.path.split(os.path.split(self.file_path)[0])[0]
            dirs = sorted(os.path.listdir(os.path.join(path, 'checkpoints')))
            load_path = os.path.join(path, dirs[-1], str(epoch) + '_eva_net.pt')
        else:
            if load_path is None:
                load_path = self.cfg.load_path
            load_path = os.path.join(load_path, str(epoch) + '_eva_net.pt')

        ckpt = torch.load(load_path, map_location=self.device)
        self.FFHEvaluator.load_state_dict(ckpt['ffhevaluator_state_dict'])

        if self.cfg.is_train:
            self.optim_ffhevaluator.load_state_dict(ckpt['optim_ffhevaluator_state_dict'])
            self.scheduler_ffhevaluator.load_state_dict(ckpt['scheduler_ffhevaluator_state_dict'])
            self.cfg.start_epoch = ckpt['epoch']
            self.FFHEvaluator.train()
        else:
            self.FFHEvaluator.eval()

    def load_ffhgenerator(self, epoch, load_path=None):
        """Load ffhgenerator from disk and set to eval or train mode
        """
        if epoch == -1:
            path = os.path.split(os.path.split(self.file_path)[0])[0]
            dirs = sorted(os.path.listdir(os.path.join(path, 'checkpoints')))
            load_path = os.path.join(path, dirs[-1], str(epoch) + '_gen_net.pt')
        else:
            if load_path is None:
                load_path = self.cfg.load_path
            load_path = os.path.join(load_path, str(epoch) + '_gen_net.pt')

        ckpt = torch.load(load_path, map_location=self.device)
        self.FFHGenerator.load_state_dict(ckpt['ffhgenerator_state_dict'])

        if self.cfg.is_train:
            print("Load TRAIN mode")
            self.optim_ffhgenerator.load_state_dict(ckpt['optim_ffhgenerator_state_dict'])
            self.scheduler_ffhgenerator.load_state_dict(ckpt['scheduler_ffhgenerator_state_dict'])
            self.FFHGenerator.train()
        else:
            print("Network in EVAL mode")
            self.FFHGenerator.eval()

    def refine_grasps(self, data, refine_method, num_refine_steps=10, dtype=torch.float32):
        """ Refine sampled and ranked grasps.

        Args:
            data (dict): Keys being bps_object, rot_matrix, transl, joint_conf describing sensor observation and grasp pose.
            refine_method (str): Choose gradient or sampling based refinement method.
            num_refine_steps (int, optional): How many steps of gradient-based refinement to apply. Defaults to 10.

        Returns:
            refined: [description]
        """
        start = time.time()
        data = utils.data_dict_to_dtype(data, dtype)
        if refine_method == "gradient":
            refine_fn = self.improve_grasps_gradient_based

            # Wrap input in Variable class, this way gradients are computed
            data['rot_matrix'] = torch.autograd.Variable(data['rot_matrix'].to(self.device),
                                                         requires_grad=True)
            data['transl'] = torch.autograd.Variable(data['transl'].to(self.device),
                                                     requires_grad=True)
            data['joint_conf'] = torch.autograd.Variable(data['joint_conf'].to(self.device),
                                                         requires_grad=True)

        else:
            refine_fn = self.improve_grasps_sampling_based

        refined_success = []
        refined_data = []
        refined_data.append(utils.grasp_numpy_from_data_dict(data))
        last_success = None
        for i in range(num_refine_steps):
            p_success, last_success = refine_fn(data, last_success)
            refined_success.append(p_success.cpu().data.numpy())
            refined_data.append(utils.grasp_numpy_from_data_dict(data))

        # we need to run the success on the final refined grasps
        refined_success.append(self.FFHEvaluator(data).squeeze().cpu().data.numpy())

        #print('Refinement took: ' + str(time.time() - start))

        return refined_data, refined_success

    def save_ffhevaluator(self, net_name, epoch):
        """ Save ffhevaluator to disk

        Args:
            net_name (str): The name of the model.
            epoch (int): Current epoch.
        """
        save_path = os.path.join(self.cfg.save_dir, net_name + '_eva_net.pt')
        if len(self.cfg.gpu_ids) > 1:
            ffhevaluator_state_dict = self.FFHEvaluator.module.cpu().state_dict()
        else:
            ffhevaluator_state_dict = self.FFHEvaluator.cpu().state_dict()
        torch.save(
            {
                'epoch': epoch,
                'ffhevaluator_state_dict': ffhevaluator_state_dict,
                'optim_ffhevaluator_state_dict': self.optim_ffhevaluator.state_dict(),
                'scheduler_ffhevaluator_state_dict': self.scheduler_ffhevaluator.state_dict(),
            }, save_path)

        if torch.cuda.is_available():
            self.FFHEvaluator.cuda()

    def save_ffhgenerator(self, net_name, epoch):
        """ Save ffhgenerator to disk

        Args:
            net_name (str): The name of the model.
            epoch (int): Current epoch.
        """
        save_path = os.path.join(self.cfg.save_dir, net_name + '_gen_net.pt')
        if len(self.cfg.gpu_ids) > 1:
            ffhgenerator_state_dict = self.FFHGenerator.module.cpu().state_dict()
        else:
            ffhgenerator_state_dict = self.FFHGenerator.cpu().state_dict()
        torch.save(
            {
                'epoch': epoch,
                'ffhgenerator_state_dict': ffhgenerator_state_dict,
                'optim_ffhgenerator_state_dict': self.optim_ffhgenerator.state_dict(),
                'scheduler_ffhgenerator_state_dict': self.scheduler_ffhgenerator.state_dict(),
            }, save_path)

        if torch.cuda.is_available():
            self.FFHGenerator.cuda()

    def update_estop(self, eval_loss_dict):
        """"[summary]"

        Args:
            eval_loss_dict (dict): Dict with all the relevant losses
        """
        if self.train_ffhevaluator:
            if self.estop_ffhevaluator(eval_loss_dict['total_loss_eva']):
                self.train_ffhevaluator = False
        if self.train_ffhgenerator:
            if self.estop_ffhgenerator(eval_loss_dict['total_loss_gen']):
                self.train_ffhgenerator = False

    def update_learning_rate(self, eval_loss_dict):
        """update learning rate (called once every epoch)"""
        if self.train_ffhevaluator:
            self.scheduler_ffhevaluator.step(eval_loss_dict['total_loss_eva'])
            lr_eva = self.optim_ffhevaluator.param_groups[0]['lr']
            print('learning rate evaluator = %.7f' % lr_eva)

        if self.train_ffhgenerator:
            self.scheduler_ffhgenerator.step(eval_loss_dict['total_loss_gen'])
            lr_gen = self.optim_ffhgenerator.param_groups[0]['lr']
            print('learning rate generator = %.7f' % lr_gen)

    def update_parameters(self):
        """ This method will handle one complete update step for all models
        enclosed by FFHNet.
        """
        raise NotImplementedError

    def update_ffhevaluator(self, data):
        # Make sure net is in train mode
        self.FFHEvaluator.train()

        # Run forward pass of ffhevaluator and predict grasp success
        out = self.FFHEvaluator(data)

        # Compute loss based on reconstructed data
        total_loss_ffhevaluator, loss_dict_ffhevaluator = self.compute_loss_ffhevaluator(out)

        # Zero gradients, backprop new loss gradient, run one step
        self.optim_ffhevaluator.zero_grad()
        total_loss_ffhevaluator.backward()
        self.optim_ffhevaluator.step()

        # Return loss
        return loss_dict_ffhevaluator

    def update_ffhgenerator(self, data):
        """ Receives a dict with all the input data to the ffhgenerator, sets the model input and runs one complete update step.
        """
        # Make sure net is in train mode
        self.FFHGenerator.train()

        # Run forward pass of ffhgenerator and reconstruct the data
        data_rec = self.FFHGenerator(data)

        # Compute loss based on reconstructed data
        total_loss_ffhgenerator, loss_dict_ffhgenerator = self.compute_loss_ffhgenerator(data_rec)

        # Zero gradients, backprop new loss gradient, run one step
        self.optim_ffhgenerator.zero_grad()
        total_loss_ffhgenerator.backward()
        self.optim_ffhgenerator.step()

        # Return the loss
        return loss_dict_ffhgenerator


if __name__ == '__main__':
    # test sampling
    from FFHNet.config.eval_config import EvalConfig
    cfg = EvalConfig().parse()
    ffhnet = FFHNet(cfg)
    ffhnet.load_ffhgenerator(10, is_train=False)
    bps = np.load('pcd_enc.npy')
    ffhnet.generate_grasps(bps, 20, return_arr=True)