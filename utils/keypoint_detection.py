# TODO: add documentation
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def calc_dists(preds, target, normalize):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[1], preds.shape[0]))
    for n in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                normed_preds = preds[n, c, :] / normalize[n]
                normed_targets = target[n, c, :] / normalize[n]
                dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
            else:
                dists[c, n] = -1
    return dists


def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
    else:
        return -1


def accuracy(output, target, hm_type='gaussian', thr=0.5):
    '''
    Calculate accuracy according to PCK,
    but uses ground truth heatmap rather than x,y locations
    First value to be returned is average accuracy across 'idxs',
    followed by individual accuracies
    '''
    idx = list(range(output.shape[1]))
    norm = 1.0
    if hm_type == 'gaussian':
        pred, _ = get_max_preds(output)
        target, _ = get_max_preds(target)
        h = output.shape[2]
        w = output.shape[3]
        norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
    dists = calc_dists(pred, target, norm)

    acc = np.zeros(len(idx))
    avg_acc = 0
    cnt = 0

    for i in range(len(idx)):
        acc[i] = dist_acc(dists[idx[i]], thr)
        if acc[i] >= 0:
            avg_acc = avg_acc + acc[i]
            cnt += 1

    avg_acc = avg_acc / cnt if cnt != 0 else 0

    return acc, avg_acc, cnt, pred


def accuracy_3d(pre, target):

    pre = pre * 1000
    target = target * 1000
    b, k, s = pre.shape
    avg_est_error = 0.0
    for i in range(b):
        dis = pre[i] - target[i]
        avg_est_error += dis.pow(2).sum(-1).sqrt().mean()

    avg_est_error /= b

    thresholds = 1 * np.array(range(20, 51, 3))
    pck_list = []
    for thr in thresholds:
        joint_threshold = np.ones(21) * thr
        num_joints_under_threshold = 0
        for i in range(b):
            under_threshold = np.zeros(21)
            dist = pre[i] - target[i]  # K x 3
            joint_est_error = dist.pow(2).sum(-1).sqrt()
            joint_est_error = np.asarray(joint_est_error.cpu())
            under_threshold[joint_est_error < joint_threshold] = 1
            num_joints_under_threshold += under_threshold.sum()
        pck_list.append(num_joints_under_threshold / (b * 21))

    threshold_list = [i * 1 for i in thresholds]
    AUC = np.trapz(pck_list, threshold_list)
    AUC = AUC / 30


    return avg_est_error, AUC

def accuracy_2d(pre, target):
    b, k, s = pre.shape
    avg_est_error = 0.0
    for i in range(b):
        dis = pre[i] - target[i]
        avg_est_error += dis.pow(2).sum(-1).sqrt().mean()

    avg_est_error /= b
    return avg_est_error


def find_keypoints_max(heatmaps):
  """
  heatmaps: C x H x W
  return: C x 3
  """
  # flatten the last axis
  heatmaps_flat = heatmaps.view(heatmaps.size(0), -1)

  # max loc
  max_val, max_ind = heatmaps_flat.max(1)
  max_ind = max_ind.float()

  max_v = torch.floor(torch.div(max_ind, heatmaps.size(1)))
  max_u = torch.fmod(max_ind, heatmaps.size(2))
  return torch.cat((max_u.view(-1,1), max_v.view(-1,1), max_val.view(-1,1)), 1)

def compute_uv_from_heatmaps(hm, resize_dim):
  """
  :param hm: B x K x H x W (Variable)
  :param resize_dim:
  :return: uv in resize_dim (Variable)
  """
  upsample = nn.Upsample(size=resize_dim, mode='bilinear')  # (B x K) x H x W
  a=hm.size
  resized_hm = upsample(hm).view(-1, resize_dim[0], resize_dim[1])
  b=resized_hm.shape

  uv_confidence = find_keypoints_max(resized_hm)  # (B x K) x 3
  uv_confidence = uv_confidence.view(-1, hm.size(1), 3)
  uv_confidence = uv_confidence[:, :, :2]
  return uv_confidence


def compute_uv_from_heatmaps2(hm, resize_dim):
  """
  :param hm: B x K x H x W (Variable)
  :param resize_dim:
  :return: uv in resize_dim (Variable)
  """
  upsample = nn.Upsample(size=resize_dim, mode='bilinear')  # (B x K) x H x W
  a=hm.size
  resized_hm = upsample(hm)
  b=resized_hm.shape

  batch_size = resized_hm.shape[0]
  num_joints = resized_hm.shape[1]
  width = resized_hm.shape[3]
  heatmaps_reshaped = resized_hm.reshape((batch_size, num_joints, -1))
  idx = torch.argmax(heatmaps_reshaped, 2)
  maxvals = torch.amax(heatmaps_reshaped, 2)

  maxvals = maxvals.reshape((batch_size, num_joints, 1))
  idx = idx.reshape((batch_size, num_joints, 1))

  #preds = torch.repeat(idx, (1, 1, 2))
  preds = idx.repeat(1, 1, 2)

  preds[:, :, 0] = (preds[:, :, 0]) % width
  preds[:, :, 1] = (preds[:, :, 1]) / width

 # pred_mask = torch.tile(torch.greater(maxvals, 0.0), (1, 1, 2))
  pred_mask0 = torch.greater(maxvals, 0.0)
  pred_mask = pred_mask0.repeat(1, 1, 2)
  #pred_mask = pred_mask.astype(np.float32)

  preds *= pred_mask
  return preds



def compute_uv_from_heatmaps3(heatmap: torch.Tensor) -> torch.Tensor:
    """
    :param heatmap: The input heatmap is of size B x N x H x W.
    :return: The index of the maximum 2d coordinates is of size B x N x 2.
    """
    heatmap = heatmap.mul(100)
    batch_size, num_channel, height, width = heatmap.size()
    device: str = heatmap.device

    softmax: torch.Tensor = F.softmax(
        heatmap.view(batch_size, num_channel, height * width), dim=2
    ).view(batch_size, num_channel, height, width)

    xx, yy = torch.meshgrid(list(map(torch.arange, [height, width])))

    approx_x = (
        softmax.mul(xx.float().to(device))
            .view(batch_size, num_channel, height * width)
            .sum(2)
            .unsqueeze(2)
    )
    approx_y = (
        softmax.mul(yy.float().to(device))
            .view(batch_size, num_channel, height * width)
            .sum(2)
            .unsqueeze(2)
    )

    output = [approx_y, approx_x]
    output = torch.cat(output, 2)
    return output*4

