"""
model by dhiraj inspried from Charles
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCAutoEncoder(nn.Module):
    """ Autoencoder for Point Cloud 
    Input: 
    Output: 
    """

    def __init__(self, point_dim, num_points):
        super(PCAutoEncoder, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=point_dim,  # why con1d here? It should be max pooling??
                               out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv5 = nn.Conv1d(
            in_channels=128, out_channels=1024, kernel_size=1)

        self.fc1 = nn.Linear(in_features=1024, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=num_points*3)

        # batch norm
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):

        batch_size = x.shape[0]
        point_dim = x.shape[1]
        num_points = x.shape[2]

        # encoder
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn1(self.conv2(x)))
        x = F.relu(self.bn1(self.conv3(x)))
        x = F.relu(self.bn2(self.conv4(x)))
        x = F.relu(self.bn3(self.conv5(x)))

        # do max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        # get the global embedding
        global_feat = x

        # decoder
        x = F.relu(self.bn3(self.fc1(x)))
        x = F.relu(self.bn3(self.fc2(x)))
        reconstructed_points = self.fc3(x)

        # do reshaping
        reconstructed_points = reconstructed_points.reshape(
            batch_size, point_dim, num_points)

        return reconstructed_points, global_feat


class PointNetGenerator(nn.Module):

    def __init__(self,
                 cfg,
                 n_neurons=512,
                 in_bps=4096,
                 in_pose=9 + 3 + 15,
                 dtype=torch.float32,
                 **kwargs):

        super(PointNetGenerator, self).__init__()

        self.cfg = cfg

        self.latentD = cfg.latentD

        self.enc_bn1 = nn.BatchNorm1d(in_bps + in_pose)
        self.enc_rb1 = ResBlock(in_bps + in_pose, n_neurons)
        self.enc_rb2 = ResBlock(n_neurons + in_bps + in_pose, n_neurons)

        self.enc_mu = nn.Linear(n_neurons, self.latentD)
        self.enc_logvar = nn.Linear(n_neurons, self.latentD)
        self.do = nn.Dropout(p=.1, inplace=False)

        self.dec_bn1 = nn.BatchNorm1d(in_bps)
        self.dec_rb1 = ResBlock(self.latentD + in_bps, n_neurons)
        self.dec_rb2 = ResBlock(n_neurons + self.latentD + in_bps, n_neurons)

        self.dec_joint_conf = nn.Linear(n_neurons, 15)
        self.dec_rot = nn.Linear(n_neurons, 6)
        self.dec_transl = nn.Linear(n_neurons, 3)

        if self.cfg.is_train:
            print("FFHGenerator currently in TRAIN mode!")
            self.train()
        else:
            print("FFHGenerator currently in EVAL mode!")
            self.eval()

        self.dtype = dtype
        self.device = torch.device('cuda:{}'.format(
            cfg.gpu_ids[0])) if torch.cuda.is_available() else torch.device('cpu')

    def decode(self, Zin, bps_object):

        bs = Zin.shape[0]
        o_bps = self.dec_bn1(bps_object)

        X0 = torch.cat([Zin, o_bps], dim=1)
        X = self.dec_rb1(X0, True)
        X = self.dec_rb2(torch.cat([X0, X], dim=1), True)

        joint_conf = self.dec_joint_conf(X)
        rot_6D = self.dec_rot(X)
        transl = self.dec_transl(X)

        results = {"rot_6D": rot_6D, "transl": transl,
                   "joint_conf": joint_conf, "z": Zin}

        return results

    def encode(self, data):
        self.set_input(data)
        X = torch.cat([self.bps_object, self.rot_matrix,
                      self.transl, self.joint_conf], dim=1)

        X0 = self.enc_bn1(X)
        X = self.enc_rb1(X0, True)
        X = self.enc_rb2(torch.cat([X0, X], dim=1), True)

        return self.enc_mu(X), self.enc_logvar(X)

    def forward(self, data):
        # Encode data, get mean and logvar
        mu, logvar = self.encode(data)

        std = logvar.exp().pow(0.5)
        q_z = torch.distributions.normal.Normal(mu, std)
        z = q_z.rsample()

        data_recon = self.decode(z, self.bps_object)
        results = {'mu': mu, 'logvar': logvar}
        results.update(data_recon)

        return results

    def generate_poses(self, bps_object, return_arr=False, seed=None, sample_uniform=False):
        """[summary]

        Args:
            bps_object (tensor): BPS encoding of object point cloud.
            return_arr (bool): Returns np array if True
            seed (int, optional): np random seed. Defaults to None.

        Returns:
            results (dict): keys being 1.rot_matrix, 2.transl, 3.joint_conf
        """
        start = time.time()
        n_samples = bps_object.shape[0]
        self.eval()
        with torch.no_grad():
            if not sample_uniform:
                Zgen = torch.randn((n_samples, self.latentD),
                                   dtype=self.dtype, device=self.device)
            else:
                Zgen = 8 * torch.rand(
                    (n_samples, self.latentD), dtype=self.dtype, device=self.device) - 4

        results = self.decode(Zgen, bps_object)

        # Transforms rot_6D to rot_matrix
        results['rot_matrix'] = utils.rot_matrix_from_ortho6d(
            results.pop('rot_6D'))

        if return_arr:
            for k, v in results.items():
                results[k] = v.cpu().detach().numpy()
        print("Sampling took: %.3f" % (time.time() - start))
        return results

    def set_input(self, data):
        """ Bring input tensors to correct dtype and device. Set whether gradient is required depending on 
        we are in train or eval mode.
        """
        rot_matrix = data["rot_matrix"].to(
            dtype=self.dtype, device=self.device)
        transl = data["transl"].to(dtype=self.dtype, device=self.device)
        joint_conf = data["joint_conf"].to(
            dtype=self.dtype, device=self.device)
        bps_object = data["bps_object"].to(
            dtype=self.dtype, device=self.device).contiguous()

        self.rot_matrix = rot_matrix.requires_grad_(self.cfg.is_train)
        self.transl = transl.requires_grad_(self.cfg.is_train)
        self.joint_conf = joint_conf.requires_grad_(self.cfg.is_train)
        self.bps_object = bps_object.requires_grad_(self.cfg.is_train)

        self.rot_matrix = self.rot_matrix.view(self.bps_object.shape[0], -1)
