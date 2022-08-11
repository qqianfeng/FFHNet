import torch
import FFHNet.utils.utils as utils


def accuracy_evaluator(pred_label, gt_label):
    correct = torch.eq(pred_label, gt_label)

    # acc
    positive_acc = torch.sum(correct * gt_label) / torch.sum(gt_label)
    negative_acc = torch.sum(correct * (1. - gt_label)) / torch.sum(1. - gt_label)

    return positive_acc, negative_acc


def control_point_l1_loss(pred_transl_rot_6D,
                          gt_transl_rot_6D,
                          confidence=None,
                          confidence_weight=None,
                          device="cpu"):
    """ L1 between predicted hand control points and ground truth hand control points.
    Expects two dicts with the prediction translation and 6D rotation and true translation and 6D rotation.
    """
    # out: batch*3*3 in: batch*6
    pred_rot_matrix = utils.rot_matrix_from_ortho6d(pred_transl_rot_6D['rot_6D'])
    gt_rot_matrix = utils.rot_matrix_from_ortho6d(gt_transl_rot_6D['rot_6D'])

    # out: batch*n_points*3 in: batch*3, batch*3*3
    pred_control_points = utils.control_points_from_transl_rot_matrix(pred_transl_rot_6D['transl'],
                                                                      pred_rot_matrix,
                                                                      device=device)
    gt_control_points = utils.control_points_from_transl_rot_matrix(gt_transl_rot_6D['transl'],
                                                                    gt_rot_matrix,
                                                                    device=device)

    error = torch.sum(torch.abs(pred_control_points - gt_control_points), -1)
    error = torch.mean(error, -1)
    if confidence is not None:
        assert confidence_weight is not None
        error *= confidence
        confidence_term = torch.mean(
            torch.log(torch.max(confidence,
                                torch.tensor(1e-10).to(device)))) * confidence_weight

    if confidence is None:
        return torch.mean(error)
    else:
        return torch.mean(error), -confidence_term


def kl_divergence(mu, logvar, device="cpu"):
    """
      Computes the kl divergence for batch of mu and logvar.
    """
    return torch.mean(-.5 * torch.sum(1. + logvar - mu**2 - torch.exp(logvar), dim=-1))


def transl_rot_6D_l2_loss(pred_transl_rot_6D,
                          gt_transl_rot_matrix,
                          torch_l2_loss_fn,
                          device='cpu'):
    """ Takes in the 3D translation and 6D rotation prediction and computes l2 loss to ground truth 3D translation
    and 3x3 rotation matrix by translforming the 6D rotation to 3x3 rotation matrix.
    """
    pred_rot_matrix = utils.rot_matrix_from_ortho6d(pred_transl_rot_6D['rot_6D'])  #batch_size*3*3
    pred_rot_matrix = pred_rot_matrix.view(pred_rot_matrix.shape[0], -1)  #batch_size*9
    gt_rot_matrix = gt_transl_rot_matrix['rot_matrix']
    # l2 loss on rotation matrix
    l2_loss_rot = torch_l2_loss_fn(pred_rot_matrix, gt_rot_matrix)
    # l2 loss on translation
    l2_loss_transl = torch_l2_loss_fn(pred_transl_rot_6D['transl'], gt_transl_rot_matrix['transl'])

    return l2_loss_transl, l2_loss_rot
