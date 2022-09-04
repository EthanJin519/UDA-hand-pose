# ------------------------------------------------------------------------------
# Modified from https://github.com/thuml/Regressive-Domain-Adaptation-for-Unsupervised-Keypoint-Detection
# ------------------------------------------------------------------------------


from typing import Optional
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from utils.gl import WarmStartGradientLayer
from utils.keypoint_detection import get_max_preds
from utils.net_utils import Residual, make_head, make_head2, Residualx, make_head3


class PseudoLabelGenerator(nn.Module):
    """
    Generate ground truth heatmap and ground false heatmap from a prediction.

    Args:
        num_keypoints (int): Number of keypoints
        height (int): height of the heatmap. Default: 64
        width (int): width of the heatmap. Default: 64
        sigma (int): sigma parameter when generate the heatmap. Default: 2

    Inputs:
        - y: predicted heatmap

    Outputs:
        - ground_truth: heatmap conforming to Gaussian distribution
        - ground_false: ground false heatmap

    Shape:
        - y: :math:`(minibatch, K, H, W)` where K means the number of keypoints,
          H and W is the height and width of the heatmap respectively.
        - ground_truth: :math:`(minibatch, K, H, W)`
        - ground_false: :math:`(minibatch, K, H, W)`
    """
    def __init__(self, num_keypoints, height=64, width=64, sigma=2):
        super(PseudoLabelGenerator, self).__init__()
        self.height = height
        self.width = width
        self.sigma = sigma

        heatmaps = np.zeros((width, height, height, width), dtype=np.float32)

        tmp_size = sigma * 3
        for mu_x in range(width):
            for mu_y in range(height):
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

                # Generate gaussian
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], width) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], height) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], width)
                img_y = max(0, ul[1]), min(br[1], height)

                heatmaps[mu_x][mu_y][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        self.heatmaps = heatmaps
        self.false_matrix = 1. - np.eye(num_keypoints, dtype=np.float32)

    def forward(self, y):
        B, K, H, W = y.shape
        y = y.detach()
        preds, max_vals = get_max_preds(y.cpu().numpy())  # B x K x (x, y)
        preds = preds.reshape(-1, 2).astype(np.int)
        ground_truth = self.heatmaps[preds[:, 0], preds[:, 1], :, :].copy().reshape(B, K, H, W).copy()

        ground_false = ground_truth.reshape(B, K, -1).transpose((0, 2, 1))
        ground_false = ground_false.dot(self.false_matrix).clip(max=1., min=0.).transpose((0, 2, 1)).reshape(B, K, H, W).copy()
    #    plt.imshow(ground_false[0])
        return torch.from_numpy(ground_truth).to(y.device), torch.from_numpy(ground_false).to(y.device)


class RegressionDisparity(nn.Module):
    """
    Regression Disparity proposed by `Regressive Domain Adaptation for Unsupervised Keypoint Detection (CVPR 2021) <https://arxiv.org/abs/2103.06175>`_.

    Args:
        pseudo_label_generator (PseudoLabelGenerator): generate ground truth heatmap and ground false heatmap
          from a prediction.
        criterion (torch.nn.Module): the loss function to calculate distance between two predictions.

    Inputs:
        - y: output by the main head
        - y_adv: output by the adversarial head
        - weight (optional): instance weights
        - mode (str): whether minimize the disparity or maximize the disparity. Choices includes ``min``, ``max``.
          Default: ``min``.

    Shape:
        - y: :math:`(minibatch, K, H, W)` where K means the number of keypoints,
          H and W is the height and width of the heatmap respectively.
        - y_adv: :math:`(minibatch, K, H, W)`
        - weight: :math:`(minibatch, K)`.
        - Output: depends on the ``criterion``.

    Examples::

        >>> num_keypoints = 5
        >>> batch_size = 10
        >>> H = W = 64
        >>> pseudo_label_generator = PseudoLabelGenerator(num_keypoints)
        >>> from common.vision.models.keypoint_detection.loss import JointsKLLoss
        >>> loss = RegressionDisparity(pseudo_label_generator, JointsKLLoss())
        >>> # output from source domain and target domain
        >>> y_s, y_t = torch.randn(batch_size, num_keypoints, H, W), torch.randn(batch_size, num_keypoints, H, W)
        >>> # adversarial output from source domain and target domain
        >>> y_s_adv, y_t_adv = torch.randn(batch_size, num_keypoints, H, W), torch.randn(batch_size, num_keypoints, H, W)
        >>> # minimize regression disparity on source domain
        >>> output = loss(y_s, y_s_adv, mode='min')
        >>> # maximize regression disparity on target domain
        >>> output = loss(y_t, y_t_adv, mode='max')
    """
    def __init__(self, pseudo_label_generator: PseudoLabelGenerator, criterion: nn.Module):
        super(RegressionDisparity, self).__init__()
        self.criterion = criterion
        self.pseudo_label_generator = pseudo_label_generator

    def forward(self, y, y_adv, weight=None, mode='min'):
        assert mode in ['min', 'max']
        ground_truth, ground_false = self.pseudo_label_generator(y.detach())
        
        self.ground_truth = ground_truth
        self.ground_false = ground_false
        if mode == 'min':
            return self.criterion(y_adv, ground_truth, weight)
        else:
            return self.criterion(y_adv, ground_false, weight)

class RegressionDisparity2(nn.Module):
    """
    Regression Disparity proposed by `Regressive Domain Adaptation for Unsupervised Keypoint Detection (CVPR 2021) <https://arxiv.org/abs/2103.06175>`_.

    Args:
        pseudo_label_generator (PseudoLabelGenerator): generate ground truth heatmap and ground false heatmap
          from a prediction.
        criterion (torch.nn.Module): the loss function to calculate distance between two predictions.

    Inputs:
        - y: output by the main head
        - y_adv: output by the adversarial head
        - weight (optional): instance weights
        - mode (str): whether minimize the disparity or maximize the disparity. Choices includes ``min``, ``max``.
          Default: ``min``.

    Shape:
        - y: :math:`(minibatch, K, H, W)` where K means the number of keypoints,
          H and W is the height and width of the heatmap respectively.
        - y_adv: :math:`(minibatch, K, H, W)`
        - weight: :math:`(minibatch, K)`.
        - Output: depends on the ``criterion``.

    Examples::

        >>> num_keypoints = 5
        >>> batch_size = 10
        >>> H = W = 64
        >>> pseudo_label_generator = PseudoLabelGenerator(num_keypoints)
        >>> from common.vision.models.keypoint_detection.loss import JointsKLLoss
        >>> loss = RegressionDisparity(pseudo_label_generator, JointsKLLoss())
        >>> # output from source domain and target domain
        >>> y_s, y_t = torch.randn(batch_size, num_keypoints, H, W), torch.randn(batch_size, num_keypoints, H, W)
        >>> # adversarial output from source domain and target domain
        >>> y_s_adv, y_t_adv = torch.randn(batch_size, num_keypoints, H, W), torch.randn(batch_size, num_keypoints, H, W)
        >>> # minimize regression disparity on source domain
        >>> output = loss(y_s, y_s_adv, mode='min')
        >>> # maximize regression disparity on target domain
        >>> output = loss(y_t, y_t_adv, mode='max')
    """
    def __init__(self, pseudo_label_generator: PseudoLabelGenerator, criterion: nn.Module):
        super(RegressionDisparity2, self).__init__()
        self.criterion = criterion
        self.pseudo_label_generator = pseudo_label_generator
        self.reset()

    def reset(self):
        self.label_x = 0.
    def updata(self, label_x):
        self.label_x = label_x
    def forward(self, y, y_adv, label_1, label_2, weight=None, mode='min'):
        assert mode in ['min', 'max']
        label_x = self.label_x
        b, c, _, _ = y.shape
        ground_truth, ground_false0 = self.pseudo_label_generator(y.detach())
        ground_truth1, ground_false1 = self.pseudo_label_generator(label_1.detach())
        ground_truth2, ground_false2 = self.pseudo_label_generator(label_2.detach())
        label_p1 = torch.sum(ground_truth, dim=1)
        label_p2 = torch.sum(ground_truth1, dim=1)
        label_p3 = torch.sum(ground_truth2, dim=1)

        label_p = label_p2 + label_p1 + label_p3
        label_p = [label_p[k] / torch.max(label_p[k]) for k in range(b)]
        label_p = torch.stack(label_p)

        label_p = label_p.unsqueeze(1).repeat(1, 21, 1, 1)
        ground_false = (label_p - ground_truth*10).clip(max=1., min=0.)


        self.ground_truth = ground_truth
        self.ground_false = ground_false
        if mode == 'min':
            return self.criterion(y_adv, ground_truth, weight)
        else:
            return self.criterion(y_adv, ground_false, weight)


class RegressionDisparity3(nn.Module):
    """
    Regression Disparity proposed by `Regressive Domain Adaptation for Unsupervised Keypoint Detection (CVPR 2021) <https://arxiv.org/abs/2103.06175>`_.

    Args:
        pseudo_label_generator (PseudoLabelGenerator): generate ground truth heatmap and ground false heatmap
          from a prediction.
        criterion (torch.nn.Module): the loss function to calculate distance between two predictions.

    Inputs:
        - y: output by the main head
        - y_adv: output by the adversarial head
        - weight (optional): instance weights
        - mode (str): whether minimize the disparity or maximize the disparity. Choices includes ``min``, ``max``.
          Default: ``min``.

    Shape:
        - y: :math:`(minibatch, K, H, W)` where K means the number of keypoints,
          H and W is the height and width of the heatmap respectively.
        - y_adv: :math:`(minibatch, K, H, W)`
        - weight: :math:`(minibatch, K)`.
        - Output: depends on the ``criterion``.

    Examples::

        >>> num_keypoints = 5
        >>> batch_size = 10
        >>> H = W = 64
        >>> pseudo_label_generator = PseudoLabelGenerator(num_keypoints)
        >>> from common.vision.models.keypoint_detection.loss import JointsKLLoss
        >>> loss = RegressionDisparity(pseudo_label_generator, JointsKLLoss())
        >>> # output from source domain and target domain
        >>> y_s, y_t = torch.randn(batch_size, num_keypoints, H, W), torch.randn(batch_size, num_keypoints, H, W)
        >>> # adversarial output from source domain and target domain
        >>> y_s_adv, y_t_adv = torch.randn(batch_size, num_keypoints, H, W), torch.randn(batch_size, num_keypoints, H, W)
        >>> # minimize regression disparity on source domain
        >>> output = loss(y_s, y_s_adv, mode='min')
        >>> # maximize regression disparity on target domain
        >>> output = loss(y_t, y_t_adv, mode='max')
    """
    def __init__(self, pseudo_label_generator: PseudoLabelGenerator, criterion: nn.Module):
        super(RegressionDisparity3, self).__init__()
        self.criterion = criterion
        self.pseudo_label_generator = pseudo_label_generator
        self.reset()

    def reset(self):
        self.label_x = 0.
    def updata(self, label_x):
        self.label_x = label_x
    def forward(self, y, y_adv, label_1, label_2, weight=None, mode='min'):
        assert mode in ['min', 'max']
        label_x = self.label_x
        b, c, _, _ = y.shape
        ground_truth, ground_false0 = self.pseudo_label_generator(y.detach())
        ground_truth1, ground_false1 = self.pseudo_label_generator(label_1.detach())
        ground_truth2, ground_false2 = self.pseudo_label_generator(label_2.detach())
        label_p1 = torch.sum(ground_truth, dim=1).clip(max=1., min=0.)
        label_p2 = torch.sum(ground_truth1, dim=1).clip(max=1., min=0.)
        label_p3 = torch.sum(ground_truth2, dim=1).clip(max=1., min=0.)

        label_p = label_p1 + label_p2 + label_p3
        label_p = [label_p[k] / torch.max(label_p[k]) for k in range(b)]
        label_p = torch.stack(label_p)


        label_p = label_p.unsqueeze(1).repeat(1, 21, 1, 1)
        ground_false = (label_p - ground_truth*10).clip(max=1., min=0.)


        self.ground_truth = ground_truth
        self.ground_false = ground_false
        if mode == 'min':
            return self.criterion(y_adv, ground_truth, weight)
        else:
            return self.criterion(y_adv, ground_false, weight)

class RegressionDisparity4(nn.Module):
    """
    Regression Disparity proposed by `Regressive Domain Adaptation for Unsupervised Keypoint Detection (CVPR 2021) <https://arxiv.org/abs/2103.06175>`_.

    Args:
        pseudo_label_generator (PseudoLabelGenerator): generate ground truth heatmap and ground false heatmap
          from a prediction.
        criterion (torch.nn.Module): the loss function to calculate distance between two predictions.

    Inputs:
        - y: output by the main head
        - y_adv: output by the adversarial head
        - weight (optional): instance weights
        - mode (str): whether minimize the disparity or maximize the disparity. Choices includes ``min``, ``max``.
          Default: ``min``.

    Shape:
        - y: :math:`(minibatch, K, H, W)` where K means the number of keypoints,
          H and W is the height and width of the heatmap respectively.
        - y_adv: :math:`(minibatch, K, H, W)`
        - weight: :math:`(minibatch, K)`.
        - Output: depends on the ``criterion``.

    Examples::

        >>> num_keypoints = 5
        >>> batch_size = 10
        >>> H = W = 64
        >>> pseudo_label_generator = PseudoLabelGenerator(num_keypoints)
        >>> from common.vision.models.keypoint_detection.loss import JointsKLLoss
        >>> loss = RegressionDisparity(pseudo_label_generator, JointsKLLoss())
        >>> # output from source domain and target domain
        >>> y_s, y_t = torch.randn(batch_size, num_keypoints, H, W), torch.randn(batch_size, num_keypoints, H, W)
        >>> # adversarial output from source domain and target domain
        >>> y_s_adv, y_t_adv = torch.randn(batch_size, num_keypoints, H, W), torch.randn(batch_size, num_keypoints, H, W)
        >>> # minimize regression disparity on source domain
        >>> output = loss(y_s, y_s_adv, mode='min')
        >>> # maximize regression disparity on target domain
        >>> output = loss(y_t, y_t_adv, mode='max')
    """
    def __init__(self, pseudo_label_generator: PseudoLabelGenerator, criterion: nn.Module):
        super(RegressionDisparity4, self).__init__()
        self.criterion = criterion
        self.pseudo_label_generator = pseudo_label_generator

    def forward(self, y, y_adv, weight=None, mode='min'):
        assert mode in ['min', 'max']
        ground_truth, ground_false = self.pseudo_label_generator(y.detach())
        label_p = torch.sum(ground_truth, dim=1).clip(max=1., min=0.)
        label_p = label_p.unsqueeze(1).repeat(1, 21, 1, 1)
        ground_false = (label_p - ground_truth*10).clip(max=1., min=0.)
        self.ground_truth = ground_truth
        self.ground_false = ground_false
        if mode == 'min':
            return self.criterion(y_adv, ground_truth, weight)
        else:
            return self.criterion(y_adv, ground_false, weight)
            
            
class RegressionDisparity5(nn.Module):
    """
    Regression Disparity proposed by `Regressive Domain Adaptation for Unsupervised Keypoint Detection (CVPR 2021) <https://arxiv.org/abs/2103.06175>`_.

    Args:
        pseudo_label_generator (PseudoLabelGenerator): generate ground truth heatmap and ground false heatmap
          from a prediction.
        criterion (torch.nn.Module): the loss function to calculate distance between two predictions.

    Inputs:
        - y: output by the main head
        - y_adv: output by the adversarial head
        - weight (optional): instance weights
        - mode (str): whether minimize the disparity or maximize the disparity. Choices includes ``min``, ``max``.
          Default: ``min``.

    Shape:
        - y: :math:`(minibatch, K, H, W)` where K means the number of keypoints,
          H and W is the height and width of the heatmap respectively.
        - y_adv: :math:`(minibatch, K, H, W)`
        - weight: :math:`(minibatch, K)`.
        - Output: depends on the ``criterion``.

    Examples::

        >>> num_keypoints = 5
        >>> batch_size = 10
        >>> H = W = 64
        >>> pseudo_label_generator = PseudoLabelGenerator(num_keypoints)
        >>> from common.vision.models.keypoint_detection.loss import JointsKLLoss
        >>> loss = RegressionDisparity(pseudo_label_generator, JointsKLLoss())
        >>> # output from source domain and target domain
        >>> y_s, y_t = torch.randn(batch_size, num_keypoints, H, W), torch.randn(batch_size, num_keypoints, H, W)
        >>> # adversarial output from source domain and target domain
        >>> y_s_adv, y_t_adv = torch.randn(batch_size, num_keypoints, H, W), torch.randn(batch_size, num_keypoints, H, W)
        >>> # minimize regression disparity on source domain
        >>> output = loss(y_s, y_s_adv, mode='min')
        >>> # maximize regression disparity on target domain
        >>> output = loss(y_t, y_t_adv, mode='max')
    """
    def __init__(self, pseudo_label_generator: PseudoLabelGenerator, criterion: nn.Module):
        super(RegressionDisparity5, self).__init__()
        self.criterion = criterion
        self.pseudo_label_generator = pseudo_label_generator

    def forward(self, y, y_adv, label_1, label_2, weight=None, mode='min'):
        assert mode in ['min', 'max']
        b, c, _, _ = y.shape
        ground_truth, ground_false0 = self.pseudo_label_generator(y.detach())
        ground_truth1, ground_false1 = self.pseudo_label_generator(label_1.detach())
        ground_truth2, ground_false2 = self.pseudo_label_generator(label_2.detach())
        label_p1 = torch.sum(ground_truth, dim=1).clip(max=1., min=0.)
        label_p2 = torch.sum(ground_truth1, dim=1).clip(max=1., min=0.)
        label_p3 = torch.sum(ground_truth2, dim=1).clip(max=1., min=0.)
        
        labelx1 = (label_p2 - label_p1).clip(max=1., min=0.)
        labelx2 = (label_p3 - label_p1).clip(max=1., min=0.)

        label_p = (label_p1 + labelx1 + labelx2).clip(max=1., min=0.)

        label_p = label_p.unsqueeze(1).repeat(1, 21, 1, 1)
        ground_false = (label_p - ground_truth*10).clip(max=1., min=0.)


        self.ground_truth = ground_truth
        self.ground_false = ground_false
        if mode == 'min':
            return self.criterion(y_adv, ground_truth, weight)
        else:
            return self.criterion(y_adv, ground_false, weight)

class RegressionDisparity6(nn.Module):
    """
    Regression Disparity proposed by `Regressive Domain Adaptation for Unsupervised Keypoint Detection (CVPR 2021) <https://arxiv.org/abs/2103.06175>`_.

    Args:
        pseudo_label_generator (PseudoLabelGenerator): generate ground truth heatmap and ground false heatmap
          from a prediction.
        criterion (torch.nn.Module): the loss function to calculate distance between two predictions.

    Inputs:
        - y: output by the main head
        - y_adv: output by the adversarial head
        - weight (optional): instance weights
        - mode (str): whether minimize the disparity or maximize the disparity. Choices includes ``min``, ``max``.
          Default: ``min``.

    Shape:
        - y: :math:`(minibatch, K, H, W)` where K means the number of keypoints,
          H and W is the height and width of the heatmap respectively.
        - y_adv: :math:`(minibatch, K, H, W)`
        - weight: :math:`(minibatch, K)`.
        - Output: depends on the ``criterion``.

    Examples::

        >>> num_keypoints = 5
        >>> batch_size = 10
        >>> H = W = 64
        >>> pseudo_label_generator = PseudoLabelGenerator(num_keypoints)
        >>> from common.vision.models.keypoint_detection.loss import JointsKLLoss
        >>> loss = RegressionDisparity(pseudo_label_generator, JointsKLLoss())
        >>> # output from source domain and target domain
        >>> y_s, y_t = torch.randn(batch_size, num_keypoints, H, W), torch.randn(batch_size, num_keypoints, H, W)
        >>> # adversarial output from source domain and target domain
        >>> y_s_adv, y_t_adv = torch.randn(batch_size, num_keypoints, H, W), torch.randn(batch_size, num_keypoints, H, W)
        >>> # minimize regression disparity on source domain
        >>> output = loss(y_s, y_s_adv, mode='min')
        >>> # maximize regression disparity on target domain
        >>> output = loss(y_t, y_t_adv, mode='max')
    """
    def __init__(self, pseudo_label_generator: PseudoLabelGenerator, criterion: nn.Module):
        super(RegressionDisparity6, self).__init__()
        self.criterion = criterion
        self.pseudo_label_generator = pseudo_label_generator

    def forward(self, y, y_adv, label_1, weight=None, mode='min'):
        assert mode in ['min', 'max']
        b, c, _, _ = y.shape
        ground_truth, ground_false0 = self.pseudo_label_generator(y.detach())
        ground_truth1, ground_false1 = self.pseudo_label_generator(label_1.detach())
        label_p1 = torch.sum(ground_truth, dim=1).clip(max=1., min=0.)
        label_p2 = torch.sum(ground_truth1, dim=1).clip(max=1., min=0.)
        
        labelx1 = (label_p2 - label_p1).clip(max=1., min=0.)

        label_p = (label_p1 + labelx1).clip(max=1., min=0.)

        label_p = label_p.unsqueeze(1).repeat(1, 21, 1, 1)
        ground_false = (label_p - ground_truth*10).clip(max=1., min=0.)


        self.ground_truth = ground_truth
        self.ground_false = ground_false
        if mode == 'min':
            return self.criterion(y_adv, ground_truth, weight)
        else:
            return self.criterion(y_adv, ground_false, weight)

class RegressionDisparity7(nn.Module):
    """
    Regression Disparity proposed by `Regressive Domain Adaptation for Unsupervised Keypoint Detection (CVPR 2021) <https://arxiv.org/abs/2103.06175>`_.

    Args:
        pseudo_label_generator (PseudoLabelGenerator): generate ground truth heatmap and ground false heatmap
          from a prediction.
        criterion (torch.nn.Module): the loss function to calculate distance between two predictions.

    Inputs:
        - y: output by the main head
        - y_adv: output by the adversarial head
        - weight (optional): instance weights
        - mode (str): whether minimize the disparity or maximize the disparity. Choices includes ``min``, ``max``.
          Default: ``min``.

    Shape:
        - y: :math:`(minibatch, K, H, W)` where K means the number of keypoints,
          H and W is the height and width of the heatmap respectively.
        - y_adv: :math:`(minibatch, K, H, W)`
        - weight: :math:`(minibatch, K)`.
        - Output: depends on the ``criterion``.

    Examples::

        >>> num_keypoints = 5
        >>> batch_size = 10
        >>> H = W = 64
        >>> pseudo_label_generator = PseudoLabelGenerator(num_keypoints)
        >>> from common.vision.models.keypoint_detection.loss import JointsKLLoss
        >>> loss = RegressionDisparity(pseudo_label_generator, JointsKLLoss())
        >>> # output from source domain and target domain
        >>> y_s, y_t = torch.randn(batch_size, num_keypoints, H, W), torch.randn(batch_size, num_keypoints, H, W)
        >>> # adversarial output from source domain and target domain
        >>> y_s_adv, y_t_adv = torch.randn(batch_size, num_keypoints, H, W), torch.randn(batch_size, num_keypoints, H, W)
        >>> # minimize regression disparity on source domain
        >>> output = loss(y_s, y_s_adv, mode='min')
        >>> # maximize regression disparity on target domain
        >>> output = loss(y_t, y_t_adv, mode='max')
    """
    def __init__(self, pseudo_label_generator: PseudoLabelGenerator, criterion: nn.Module):
        super(RegressionDisparity7, self).__init__()
        self.criterion = criterion
        self.pseudo_label_generator = pseudo_label_generator
        self.reset()

    def reset(self):
        self.label_x = 0.
    def updata(self, label_x):
        self.label_x = label_x
    def forward(self, y, y_adv, label_1, weight=None, mode='min'):
        assert mode in ['min', 'max']
        label_x = self.label_x
        b, c, _, _ = y.shape
        ground_truth, ground_false0 = self.pseudo_label_generator(y.detach())
        ground_truth1, ground_false1 = self.pseudo_label_generator(label_1.detach())

        label_p1 = torch.sum(ground_truth, dim=1).clip(max=1., min=0.)
        label_p2 = torch.sum(ground_truth1, dim=1).clip(max=1., min=0.)


        label_p = label_p2 + label_p1
        label_p = [label_p[k] / torch.max(label_p[k]) for k in range(b)]
        label_p = torch.stack(label_p)

        label_p = label_p.unsqueeze(1).repeat(1, 21, 1, 1)
        ground_false = (label_p - ground_truth*10).clip(max=1., min=0.)


        self.ground_truth = ground_truth
        self.ground_false = ground_false
        if mode == 'min':
            return self.criterion(y_adv, ground_truth, weight)
        else:
            return self.criterion(y_adv, ground_false, weight)


class RegressionDisparity8(nn.Module):
    """
    Regression Disparity proposed by `Regressive Domain Adaptation for Unsupervised Keypoint Detection (CVPR 2021) <https://arxiv.org/abs/2103.06175>`_.

    Args:
        pseudo_label_generator (PseudoLabelGenerator): generate ground truth heatmap and ground false heatmap
          from a prediction.
        criterion (torch.nn.Module): the loss function to calculate distance between two predictions.

    Inputs:
        - y: output by the main head
        - y_adv: output by the adversarial head
        - weight (optional): instance weights
        - mode (str): whether minimize the disparity or maximize the disparity. Choices includes ``min``, ``max``.
          Default: ``min``.

    Shape:
        - y: :math:`(minibatch, K, H, W)` where K means the number of keypoints,
          H and W is the height and width of the heatmap respectively.
        - y_adv: :math:`(minibatch, K, H, W)`
        - weight: :math:`(minibatch, K)`.
        - Output: depends on the ``criterion``.

    Examples::

        >>> num_keypoints = 5
        >>> batch_size = 10
        >>> H = W = 64
        >>> pseudo_label_generator = PseudoLabelGenerator(num_keypoints)
        >>> from common.vision.models.keypoint_detection.loss import JointsKLLoss
        >>> loss = RegressionDisparity(pseudo_label_generator, JointsKLLoss())
        >>> # output from source domain and target domain
        >>> y_s, y_t = torch.randn(batch_size, num_keypoints, H, W), torch.randn(batch_size, num_keypoints, H, W)
        >>> # adversarial output from source domain and target domain
        >>> y_s_adv, y_t_adv = torch.randn(batch_size, num_keypoints, H, W), torch.randn(batch_size, num_keypoints, H, W)
        >>> # minimize regression disparity on source domain
        >>> output = loss(y_s, y_s_adv, mode='min')
        >>> # maximize regression disparity on target domain
        >>> output = loss(y_t, y_t_adv, mode='max')
    """
    def __init__(self, pseudo_label_generator: PseudoLabelGenerator, criterion: nn.Module):
        super(RegressionDisparity8, self).__init__()
        self.criterion = criterion
        self.pseudo_label_generator = pseudo_label_generator

    def forward(self, y, y_adv, label_1, label_2, weight=None, mode='min'):
        assert mode in ['min', 'max']
        b, c, _, _ = y.shape
        ground_truth, ground_false0 = self.pseudo_label_generator(y.detach())
        ground_truth1, ground_false1 = self.pseudo_label_generator(label_1.detach())
        ground_truth2, ground_false2 = self.pseudo_label_generator(label_2.detach())
        label_p1 = torch.sum(ground_truth, dim=1).clip(max=1., min=0.)
        label_p2 = torch.sum(ground_truth1, dim=1).clip(max=1., min=0.)
        label_p3 = torch.sum(ground_truth2, dim=1).clip(max=1., min=0.)
        
        labelx1 = torch.sum((ground_truth1 - ground_truth).clip(max=1., min=0.), dim=1).clip(max=1., min=0.)

        labelx2 = torch.sum((ground_truth2 - ground_truth).clip(max=1., min=0.), dim=1).clip(max=1., min=0.)


        label_p = (label_p1 + labelx1 + labelx2).clip(max=1., min=0.)

        label_p = label_p.unsqueeze(1).repeat(1, 21, 1, 1)
        ground_false = (label_p - ground_truth*10).clip(max=1., min=0.)


        self.ground_truth = ground_truth
        self.ground_false = ground_false
        if mode == 'min':
            return self.criterion(y_adv, ground_truth, weight)
        else:
            return self.criterion(y_adv, ground_false, weight)



class PoseResNet(nn.Module):
    """
    Pose ResNet for RegDA has one backbone, one upsampling, while two regression heads.

    Args:
        backbone (torch.nn.Module): Backbone to extract 2-d features from data
        upsampling (torch.nn.Module): Layer to upsample image feature to heatmap size
        feature_dim (int): The dimension of the features from upsampling layer.
        num_keypoints (int): Number of keypoints
        gl (WarmStartGradientLayer):
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: True
        num_head_layers (int): Number of head layers. Default: 2

    Inputs:
        - x (tensor): input data

    Outputs:
        - outputs: logits outputs by the main regressor
        - outputs_adv: logits outputs by the adversarial regressor

    Shapes:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, K, H, W)`, where K means the number of keypoints.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

            >>> # x is inputs, model is an PoseResNet
            >>> outputs, outputs_adv = model(x)
            >>> model.step()
    """
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints,
                 gl: Optional[WarmStartGradientLayer] = None, finetune: Optional[bool] = True, num_head_layers=2):
        super(PoseResNet, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv2 = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.finetune = finetune
        self.gl_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) if gl is None else gl

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints):
        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
            ])
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return layers

    def forward(self, x):
        x = self.backbone(x)
        f = self.upsampling(x)
        f_adv = self.gl_layer(f)
        y = self.head(f)
        y_adv = self.head_adv(f_adv)
        y_adv2 = self.head_adv2(f)

        if self.training:
            return y, y_adv, y_adv2
        else:
            return y_adv2

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
            {'params': self.head_adv.parameters(), 'lr': lr},
            {'params': self.head_adv2.parameters(), 'lr': lr},
        ]

    def step(self):
        """Call step() each iteration during training.
        Will increase :math:`\lambda` in GL layer.
        """
        self.gl_layer.step()

class PoseResNet2(nn.Module):
    """
    Pose ResNet for RegDA has one backbone, one upsampling, while two regression heads.

    Args:
        backbone (torch.nn.Module): Backbone to extract 2-d features from data
        upsampling (torch.nn.Module): Layer to upsample image feature to heatmap size
        feature_dim (int): The dimension of the features from upsampling layer.
        num_keypoints (int): Number of keypoints
        gl (WarmStartGradientLayer):
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: True
        num_head_layers (int): Number of head layers. Default: 2

    Inputs:
        - x (tensor): input data

    Outputs:
        - outputs: logits outputs by the main regressor
        - outputs_adv: logits outputs by the adversarial regressor

    Shapes:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, K, H, W)`, where K means the number of keypoints.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

            >>> # x is inputs, model is an PoseResNet
            >>> outputs, outputs_adv = model(x)
            >>> model.step()
    """
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints,
                 gl: Optional[WarmStartGradientLayer] = None, finetune: Optional[bool] = True, num_head_layers=2):
        super(PoseResNet2, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv2 = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.finetune = finetune
        self.gl_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) if gl is None else gl

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints):
        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
            ])
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return layers

    def forward(self, x):
        x = self.backbone(x)
        f = self.upsampling(x)
        f_adv = self.gl_layer(f)
        y = self.head(f)
        y_adv = self.head_adv(f_adv)
        y_adv2 = self.head_adv2(f)
        
        return y, y_adv, y_adv2
    
     

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
            {'params': self.head_adv.parameters(), 'lr': lr},
            {'params': self.head_adv2.parameters(), 'lr': lr},
        ]

    def step(self):
        """Call step() each iteration during training.
        Will increase :math:`\lambda` in GL layer.
        """
        self.gl_layer.step()
        
        
class PoseResNet3(nn.Module):
    """
    Pose ResNet for RegDA has one backbone, one upsampling, while two regression heads.

    Args:
        backbone (torch.nn.Module): Backbone to extract 2-d features from data
        upsampling (torch.nn.Module): Layer to upsample image feature to heatmap size
        feature_dim (int): The dimension of the features from upsampling layer.
        num_keypoints (int): Number of keypoints
        gl (WarmStartGradientLayer):
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: True
        num_head_layers (int): Number of head layers. Default: 2

    Inputs:
        - x (tensor): input data

    Outputs:
        - outputs: logits outputs by the main regressor
        - outputs_adv: logits outputs by the adversarial regressor

    Shapes:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, K, H, W)`, where K means the number of keypoints.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

            >>> # x is inputs, model is an PoseResNet
            >>> outputs, outputs_adv = model(x)
            >>> model.step()
    """
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints,
                 gl: Optional[WarmStartGradientLayer] = None, finetune: Optional[bool] = True, num_head_layers=2):
        super(PoseResNet3, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv2 = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.finetune = finetune
        self.gl_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) if gl is None else gl

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints):
        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
            ])
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return layers

    def forward(self, x):
        x = self.backbone(x)
        f = self.upsampling(x)
        f_adv = self.gl_layer(f)
        y = self.head(f)
        y_adv = self.head_adv(f_adv)
        y_adv2 = self.head_adv2(f)

        if self.training:
            return y, y_adv, y_adv2, f
        else:
            return y_adv2

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
            {'params': self.head_adv.parameters(), 'lr': lr},
            {'params': self.head_adv2.parameters(), 'lr': lr},
        ]

    def step(self):
        """Call step() each iteration during training.
        Will increase :math:`\lambda` in GL layer.
        """
        self.gl_layer.step()





class PoseResNet001(nn.Module):
    """
    Pose ResNet for RegDA has one backbone, one upsampling, while two regression heads.

    Args:
        backbone (torch.nn.Module): Backbone to extract 2-d features from data
        upsampling (torch.nn.Module): Layer to upsample image feature to heatmap size
        feature_dim (int): The dimension of the features from upsampling layer.
        num_keypoints (int): Number of keypoints
        gl (WarmStartGradientLayer):
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: True
        num_head_layers (int): Number of head layers. Default: 2

    Inputs:
        - x (tensor): input data

    Outputs:
        - outputs: logits outputs by the main regressor
        - outputs_adv: logits outputs by the adversarial regressor

    Shapes:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, K, H, W)`, where K means the number of keypoints.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

            >>> # x is inputs, model is an PoseResNet
            >>> outputs, outputs_adv = model(x)
            >>> model.step()
    """
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints,
                 gl: Optional[WarmStartGradientLayer] = None, finetune: Optional[bool] = True, num_head_layers=3):
        super(PoseResNet001, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv2 = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.finetune = finetune
        self.gl_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) if gl is None else gl

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints):
        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
            ])
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return layers

    def forward(self, x):
        x = self.backbone(x)
        f = self.upsampling(x)
        f_adv = self.gl_layer(f)
        y = self.head(f)
        y_adv = self.head_adv(f_adv)
        y_adv2 = self.head_adv2(f)
        
        return y, y_adv, y_adv2
    
     

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
            {'params': self.head_adv.parameters(), 'lr': lr},
            {'params': self.head_adv2.parameters(), 'lr': lr},
        ]

    def step(self):
        """Call step() each iteration during training.
        Will increase :math:`\lambda` in GL layer.
        """
        self.gl_layer.step()
        
        
class PoseResNet002(nn.Module):
    """
    Pose ResNet for RegDA has one backbone, one upsampling, while two regression heads.

    Args:
        backbone (torch.nn.Module): Backbone to extract 2-d features from data
        upsampling (torch.nn.Module): Layer to upsample image feature to heatmap size
        feature_dim (int): The dimension of the features from upsampling layer.
        num_keypoints (int): Number of keypoints
        gl (WarmStartGradientLayer):
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: True
        num_head_layers (int): Number of head layers. Default: 2

    Inputs:
        - x (tensor): input data

    Outputs:
        - outputs: logits outputs by the main regressor
        - outputs_adv: logits outputs by the adversarial regressor

    Shapes:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, K, H, W)`, where K means the number of keypoints.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

            >>> # x is inputs, model is an PoseResNet
            >>> outputs, outputs_adv = model(x)
            >>> model.step()
    """
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints,
                 gl: Optional[WarmStartGradientLayer] = None, finetune: Optional[bool] = True, num_head_layers=3):
        super(PoseResNet002, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv2 = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.finetune = finetune
        self.gl_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) if gl is None else gl

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints):
        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
            ])
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return layers

    def forward(self, x):
        x = self.backbone(x)
        f = self.upsampling(x)
        f_adv = self.gl_layer(f)
        y = self.head(f)
        y_adv = self.head_adv(f_adv)
        y_adv2 = self.head_adv2(f)

        if self.training:
            return y, y_adv, y_adv2, f
        else:
            return y_adv2

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
            {'params': self.head_adv.parameters(), 'lr': lr},
            {'params': self.head_adv2.parameters(), 'lr': lr},
        ]

    def step(self):
        """Call step() each iteration during training.
        Will increase :math:`\lambda` in GL layer.
        """
        self.gl_layer.step()





class PoseResNet003(nn.Module):
    """
    Pose ResNet for RegDA has one backbone, one upsampling, while two regression heads.

    Args:
        backbone (torch.nn.Module): Backbone to extract 2-d features from data
        upsampling (torch.nn.Module): Layer to upsample image feature to heatmap size
        feature_dim (int): The dimension of the features from upsampling layer.
        num_keypoints (int): Number of keypoints
        gl (WarmStartGradientLayer):
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: True
        num_head_layers (int): Number of head layers. Default: 2

    Inputs:
        - x (tensor): input data

    Outputs:
        - outputs: logits outputs by the main regressor
        - outputs_adv: logits outputs by the adversarial regressor

    Shapes:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, K, H, W)`, where K means the number of keypoints.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

            >>> # x is inputs, model is an PoseResNet
            >>> outputs, outputs_adv = model(x)
            >>> model.step()
    """
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints,
                 gl: Optional[WarmStartGradientLayer] = None, finetune: Optional[bool] = True, num_head_layers=2):
        super(PoseResNet003, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv2 = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.finetune = finetune
        self.gl_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) if gl is None else gl

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints):
        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
            ])
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return layers

    def forward(self, x):
        x = self.backbone(x)
        f = self.upsampling(x)
        f_adv = self.gl_layer(f)
        y = self.head(f)
        y_adv = self.head_adv(f_adv)
        y_adv2 = self.head_adv2(f)
        
        return y, y_adv, y_adv2
    
     

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
            {'params': self.head_adv.parameters(), 'lr': lr},
            {'params': self.head_adv2.parameters(), 'lr': lr},
        ]

    def step(self):
        """Call step() each iteration during training.
        Will increase :math:`\lambda` in GL layer.
        """
        self.gl_layer.step()
        
        
class PoseResNet004(nn.Module):
    """
    Pose ResNet for RegDA has one backbone, one upsampling, while two regression heads.

    Args:
        backbone (torch.nn.Module): Backbone to extract 2-d features from data
        upsampling (torch.nn.Module): Layer to upsample image feature to heatmap size
        feature_dim (int): The dimension of the features from upsampling layer.
        num_keypoints (int): Number of keypoints
        gl (WarmStartGradientLayer):
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: True
        num_head_layers (int): Number of head layers. Default: 2

    Inputs:
        - x (tensor): input data

    Outputs:
        - outputs: logits outputs by the main regressor
        - outputs_adv: logits outputs by the adversarial regressor

    Shapes:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, K, H, W)`, where K means the number of keypoints.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

            >>> # x is inputs, model is an PoseResNet
            >>> outputs, outputs_adv = model(x)
            >>> model.step()
    """
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints,
                 gl: Optional[WarmStartGradientLayer] = None, finetune: Optional[bool] = True, num_head_layers=2):
        super(PoseResNet004, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv2 = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.finetune = finetune
        self.gl_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) if gl is None else gl

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints):
        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
            ])
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return layers

    def forward(self, x):
        x = self.backbone(x)
        f = self.upsampling(x)
        f_adv = self.gl_layer(f)
        y = self.head(f)
        y_adv = self.head_adv(f_adv)
        y_adv2 = self.head_adv2(f)

        if self.training:
            return y, y_adv, y_adv2, f
        else:
            return y_adv2

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
            {'params': self.head_adv.parameters(), 'lr': lr},
            {'params': self.head_adv2.parameters(), 'lr': lr},
        ]

    def step(self):
        """Call step() each iteration during training.
        Will increase :math:`\lambda` in GL layer.
        """
        self.gl_layer.step()



class PoseResNet30(nn.Module):
    """
    Pose ResNet for RegDA has one backbone, one upsampling, while two regression heads.

    Args:
        backbone (torch.nn.Module): Backbone to extract 2-d features from data
        upsampling (torch.nn.Module): Layer to upsample image feature to heatmap size
        feature_dim (int): The dimension of the features from upsampling layer.
        num_keypoints (int): Number of keypoints
        gl (WarmStartGradientLayer):
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: True
        num_head_layers (int): Number of head layers. Default: 2

    Inputs:
        - x (tensor): input data

    Outputs:
        - outputs: logits outputs by the main regressor
        - outputs_adv: logits outputs by the adversarial regressor

    Shapes:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, K, H, W)`, where K means the number of keypoints.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

            >>> # x is inputs, model is an PoseResNet
            >>> outputs, outputs_adv = model(x)
            >>> model.step()
    """
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints,
                 gl: Optional[WarmStartGradientLayer] = None, finetune: Optional[bool] = True, num_head_layers=2):
        super(PoseResNet30, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv2 = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.finetune = finetune
        self.gl_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) if gl is None else gl

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints):
        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
            ])
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return layers

    def forward(self, x):
        x = self.backbone(x)
        f = self.upsampling(x)
        f_adv = self.gl_layer(f)
        y = self.head(f)
        y_adv = self.head_adv(f_adv)
        y_adv2 = self.head_adv2(f)

        if self.training:
            return y, y_adv, y_adv2, f
        else:
            return y, y_adv, y_adv2, f

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
            {'params': self.head_adv.parameters(), 'lr': lr},
            {'params': self.head_adv2.parameters(), 'lr': lr},
        ]

    def step(self):
        """Call step() each iteration during training.
        Will increase :math:`\lambda` in GL layer.
        """
        self.gl_layer.step()


class PoseResNet6(nn.Module):
    """
    Pose ResNet for RegDA has one backbone, one upsampling, while two regression heads.

    Args:
        backbone (torch.nn.Module): Backbone to extract 2-d features from data
        upsampling (torch.nn.Module): Layer to upsample image feature to heatmap size
        feature_dim (int): The dimension of the features from upsampling layer.
        num_keypoints (int): Number of keypoints
        gl (WarmStartGradientLayer):
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: True
        num_head_layers (int): Number of head layers. Default: 2

    Inputs:
        - x (tensor): input data

    Outputs:
        - outputs: logits outputs by the main regressor
        - outputs_adv: logits outputs by the adversarial regressor

    Shapes:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, K, H, W)`, where K means the number of keypoints.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

            >>> # x is inputs, model is an PoseResNet
            >>> outputs, outputs_adv = model(x)
            >>> model.step()
    """
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints,
                 gl: Optional[WarmStartGradientLayer] = None, finetune: Optional[bool] = True, num_head_layers=2):
        super(PoseResNet6, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv2 = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.finetune = finetune
        self.enc = self.encoder(256)
        self.gl_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) if gl is None else gl

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints):
        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
            ])
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return layers
    
    def encoder(self, num_feat_chan):
        encode = []
        for i in range(4):
            for j in range(2):
                encode.append(Residual(num_feat_chan, num_feat_chan))
            encode.append(nn.MaxPool2d(kernel_size=2, stride=2))

        enco = nn.Sequential(*encode)
        return enco

    def forward(self, x):
        x = self.backbone(x)
        f = self.upsampling(x)
        fea = self.enc(f)
        fea2 = fea.view(fea.size(0), -1)
        f_adv = self.gl_layer(f)
        y = self.head(f)
        y_adv = self.head_adv(f_adv)
        y_adv2 = self.head_adv2(f)

        if self.training:
            return y, y_adv, y_adv2, fea2
        else:
            return y

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
            {'params': self.head_adv.parameters(), 'lr': lr},
            {'params': self.head_adv2.parameters(), 'lr': lr},
            {'params': self.enc.parameters(), 'lr': 0.01 * lr},
        ]

    def step(self):
        """Call step() each iteration during training.
        Will increase :math:`\lambda` in GL layer.
        """
        self.gl_layer.step()

class PoseResNet7(nn.Module):
    """
    Pose ResNet for RegDA has one backbone, one upsampling, while two regression heads.

    Args:
        backbone (torch.nn.Module): Backbone to extract 2-d features from data
        upsampling (torch.nn.Module): Layer to upsample image feature to heatmap size
        feature_dim (int): The dimension of the features from upsampling layer.
        num_keypoints (int): Number of keypoints
        gl (WarmStartGradientLayer):
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: True
        num_head_layers (int): Number of head layers. Default: 2

    Inputs:
        - x (tensor): input data

    Outputs:
        - outputs: logits outputs by the main regressor
        - outputs_adv: logits outputs by the adversarial regressor

    Shapes:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, K, H, W)`, where K means the number of keypoints.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

            >>> # x is inputs, model is an PoseResNet
            >>> outputs, outputs_adv = model(x)
            >>> model.step()
    """
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints,
                 gl: Optional[WarmStartGradientLayer] = None, finetune: Optional[bool] = True, num_head_layers=2):
        super(PoseResNet7, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv2 = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.finetune = finetune
        self.enc = self.encoder(256)
        self.gl_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) if gl is None else gl

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints):
        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
            ])
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return layers
    def encoder(self, num_feat_chan):
        encode = []
        for i in range(4):
            for j in range(2):
                encode.append(Residual(num_feat_chan, num_feat_chan))
            encode.append(nn.MaxPool2d(kernel_size=2, stride=2))

        enco = nn.Sequential(*encode)
        return enco

    def forward(self, x):
        x = self.backbone(x)
        f = self.upsampling(x)
        fea = self.enc(f)
        fea2 = fea.view(fea.size(0), -1)
        f_adv = self.gl_layer(f)
        y = self.head(f)
        y_adv = self.head_adv(f_adv)
        y_adv2 = self.head_adv2(f)

        if self.training:
            return y, y_adv, y_adv2, fea2
        else:
            return y

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
            {'params': self.head_adv.parameters(), 'lr': lr},
            {'params': self.head_adv2.parameters(), 'lr': lr},
            {'params': self.enc.parameters(), 'lr': 0.01*lr},
        ]

    def step(self):
        """Call step() each iteration during training.
        Will increase :math:`\lambda` in GL layer.
        """
        self.gl_layer.step()


class PoseResNet8(nn.Module):
    """
    Pose ResNet for RegDA has one backbone, one upsampling, while two regression heads.

    Args:
        backbone (torch.nn.Module): Backbone to extract 2-d features from data
        upsampling (torch.nn.Module): Layer to upsample image feature to heatmap size
        feature_dim (int): The dimension of the features from upsampling layer.
        num_keypoints (int): Number of keypoints
        gl (WarmStartGradientLayer):
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: True
        num_head_layers (int): Number of head layers. Default: 2

    Inputs:
        - x (tensor): input data

    Outputs:
        - outputs: logits outputs by the main regressor
        - outputs_adv: logits outputs by the adversarial regressor

    Shapes:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, K, H, W)`, where K means the number of keypoints.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

            >>> # x is inputs, model is an PoseResNet
            >>> outputs, outputs_adv = model(x)
            >>> model.step()
    """
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints,
                 gl: Optional[WarmStartGradientLayer] = None, finetune: Optional[bool] = True, num_head_layers=2):
        super(PoseResNet8, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv2 = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.finetune = finetune
        self.enc = self.encoder(256)
        self.gl_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) if gl is None else gl

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints):
        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
            ])
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return layers
    def encoder(self, num_feat_chan):
        encode = []
        for i in range(4):
            for j in range(2):
                encode.append(Residual(num_feat_chan, num_feat_chan))
            encode.append(nn.MaxPool2d(kernel_size=2, stride=2))

        enco = nn.Sequential(*encode)
        return enco

    def forward(self, x):
        x = self.backbone(x)
        f = self.upsampling(x)
        fea = self.enc(f)
        fea2 = fea.view(x.size(0), -1)
        f_adv = self.gl_layer(f)
        y = self.head(f)
        y_adv = self.head_adv(f_adv)
        y_adv2 = self.head_adv2(f)

    
        return y, y_adv, y_adv2, fea2

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
            {'params': self.head_adv.parameters(), 'lr': lr},
            {'params': self.head_adv2.parameters(), 'lr': lr},
            {'params': self.enc.parameters(), 'lr': 0.01*lr},
        ]

    def step(self):
        """Call step() each iteration during training.
        Will increase :math:`\lambda` in GL layer.
        """
        self.gl_layer.step()




class PoseResNet01(nn.Module):
    """
    Pose ResNet for RegDA has one backbone, one upsampling, while two regression heads.

    Args:
        backbone (torch.nn.Module): Backbone to extract 2-d features from data
        upsampling (torch.nn.Module): Layer to upsample image feature to heatmap size
        feature_dim (int): The dimension of the features from upsampling layer.
        num_keypoints (int): Number of keypoints
        gl (WarmStartGradientLayer):
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: True
        num_head_layers (int): Number of head layers. Default: 2

    Inputs:
        - x (tensor): input data

    Outputs:
        - outputs: logits outputs by the main regressor
        - outputs_adv: logits outputs by the adversarial regressor

    Shapes:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, K, H, W)`, where K means the number of keypoints.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

            >>> # x is inputs, model is an PoseResNet
            >>> outputs, outputs_adv = model(x)
            >>> model.step()
    """
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints,
                 gl: Optional[WarmStartGradientLayer] = None, finetune: Optional[bool] = True, num_head_layers=2):
        super(PoseResNet01, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv2 = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.finetune = finetune
        self.enc = self.encoder(256)
        self.gl_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) if gl is None else gl

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints):
        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
            ])
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return layers
    def encoder(self, num_feat_chan):
        encode = []
        for i in range(4):
            for j in range(2):
                encode.append(Residualx(num_feat_chan, num_feat_chan))
            encode.append(nn.MaxPool2d(kernel_size=2, stride=2))

        enco = nn.Sequential(*encode)
        return enco

    def encoder2(self, num_feat_chan):
        encode = []
        for i in range(4):
            for j in range(2):
                encode.extend([
                    nn.Conv2d(num_feat_chan, num_feat_chan // 2, bias=True, kernel_size=1),
                    nn.BatchNorm2d(num_feat_chan//2),
                    nn.ReLU(),
                    nn.Conv2d(num_feat_chan // 2, num_feat_chan // 2, bias=True, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(num_feat_chan // 2),
                    nn.ReLU(),
                    nn.Conv2d(num_feat_chan // 2, num_feat_chan, bias=True, kernel_size=1)
                    ])
            encode.append(nn.MaxPool2d(kernel_size=2, stride=2))

        enco = nn.Sequential(*encode)
        for m in enco.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return enco

    def forward(self, x):
        x = self.backbone(x)
        f = self.upsampling(x)
        fea = self.enc(f)
        fea2 = fea.view(fea.size(0), -1)
        f_adv = self.gl_layer(f)
        y = self.head(f)
        y_adv = self.head_adv(f_adv)
        y_adv2 = self.head_adv2(f)

        if self.training:
            return y, y_adv, y_adv2, fea2
        else:
            return y

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
            {'params': self.head_adv.parameters(), 'lr': lr},
            {'params': self.head_adv2.parameters(), 'lr': lr},
            {'params': self.enc.parameters(), 'lr': lr},
        ]

    def step(self):
        """Call step() each iteration during training.
        Will increase :math:`\lambda` in GL layer.
        """
        self.gl_layer.step()



class PoseResNet02(nn.Module):
    """
    Pose ResNet for RegDA has one backbone, one upsampling, while two regression heads.

    Args:
        backbone (torch.nn.Module): Backbone to extract 2-d features from data
        upsampling (torch.nn.Module): Layer to upsample image feature to heatmap size
        feature_dim (int): The dimension of the features from upsampling layer.
        num_keypoints (int): Number of keypoints
        gl (WarmStartGradientLayer):
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: True
        num_head_layers (int): Number of head layers. Default: 2

    Inputs:
        - x (tensor): input data

    Outputs:
        - outputs: logits outputs by the main regressor
        - outputs_adv: logits outputs by the adversarial regressor

    Shapes:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, K, H, W)`, where K means the number of keypoints.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

            >>> # x is inputs, model is an PoseResNet
            >>> outputs, outputs_adv = model(x)
            >>> model.step()
    """
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints,
                 gl: Optional[WarmStartGradientLayer] = None, finetune: Optional[bool] = True, num_head_layers=2):
        super(PoseResNet02, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv2 = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.finetune = finetune
        self.enc = self.encoder(256)
        self.gl_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) if gl is None else gl

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints):
        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
            ])
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return layers
    def encoder(self, num_feat_chan):
        encode = []
        for i in range(4):
            for j in range(2):
                encode.append(Residualx(num_feat_chan, num_feat_chan))
            encode.append(nn.MaxPool2d(kernel_size=2, stride=2))

        enco = nn.Sequential(*encode)
        return enco

    def encoder2(self, num_feat_chan):
        encode = []
        for i in range(4):
            for j in range(2):
                encode.extend([
                    nn.Conv2d(num_feat_chan, num_feat_chan // 2, bias=True, kernel_size=1),
                    nn.BatchNorm2d(num_feat_chan//2),
                    nn.ReLU(),
                    nn.Conv2d(num_feat_chan // 2, num_feat_chan // 2, bias=True, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(num_feat_chan // 2),
                    nn.ReLU(),
                    nn.Conv2d(num_feat_chan // 2, num_feat_chan, bias=True, kernel_size=1)
                    ])
            encode.append(nn.MaxPool2d(kernel_size=2, stride=2))

        enco = nn.Sequential(*encode)
        for m in enco.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return enco

    def forward(self, x):
        x = self.backbone(x)
        f = self.upsampling(x)
        fea = self.enc(f)
        fea2 = fea.view(fea.size(0), -1)
        f_adv = self.gl_layer(f)
        y = self.head(f)
        y_adv = self.head_adv(f_adv)
        y_adv2 = self.head_adv2(f)


        return y, y_adv, y_adv2, fea2

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
            {'params': self.head_adv.parameters(), 'lr': lr},
            {'params': self.head_adv2.parameters(), 'lr': lr},
            {'params': self.enc.parameters(), 'lr': lr},
        ]

    def step(self):
        """Call step() each iteration during training.
        Will increase :math:`\lambda` in GL layer.
        """
        self.gl_layer.step()


class PoseResNet03(nn.Module):
    """
    Pose ResNet for RegDA has one backbone, one upsampling, while two regression heads.

    Args:
        backbone (torch.nn.Module): Backbone to extract 2-d features from data
        upsampling (torch.nn.Module): Layer to upsample image feature to heatmap size
        feature_dim (int): The dimension of the features from upsampling layer.
        num_keypoints (int): Number of keypoints
        gl (WarmStartGradientLayer):
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: True
        num_head_layers (int): Number of head layers. Default: 2

    Inputs:
        - x (tensor): input data

    Outputs:
        - outputs: logits outputs by the main regressor
        - outputs_adv: logits outputs by the adversarial regressor

    Shapes:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, K, H, W)`, where K means the number of keypoints.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

            >>> # x is inputs, model is an PoseResNet
            >>> outputs, outputs_adv = model(x)
            >>> model.step()
    """
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints,
                 gl: Optional[WarmStartGradientLayer] = None, finetune: Optional[bool] = True, num_head_layers=2):
        super(PoseResNet03, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv2 = make_head(num_head_layers, feature_dim, num_keypoints)
        self.finetune = finetune
        self.gl_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) if gl is None else gl

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints):
        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
            ])
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return layers

    def forward(self, x):
        x = self.backbone(x)
        f = self.upsampling(x)
        f_adv = self.gl_layer(f)
        y = self.head(f)
        y_adv = self.head_adv(f_adv)
        fea2, y_adv2 = self.head_adv2(f)

        if self.training:
            return y, y_adv, y_adv2, fea2
        else:
            return y

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
            {'params': self.head_adv.parameters(), 'lr': lr},
            {'params': self.head_adv2.parameters(), 'lr': lr},
        ]

    def step(self):
        """Call step() each iteration during training.
        Will increase :math:`\lambda` in GL layer.
        """
        self.gl_layer.step()



class PoseResNet04(nn.Module):
    """
    Pose ResNet for RegDA has one backbone, one upsampling, while two regression heads.

    Args:
        backbone (torch.nn.Module): Backbone to extract 2-d features from data
        upsampling (torch.nn.Module): Layer to upsample image feature to heatmap size
        feature_dim (int): The dimension of the features from upsampling layer.
        num_keypoints (int): Number of keypoints
        gl (WarmStartGradientLayer):
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: True
        num_head_layers (int): Number of head layers. Default: 2

    Inputs:
        - x (tensor): input data

    Outputs:
        - outputs: logits outputs by the main regressor
        - outputs_adv: logits outputs by the adversarial regressor

    Shapes:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, K, H, W)`, where K means the number of keypoints.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

            >>> # x is inputs, model is an PoseResNet
            >>> outputs, outputs_adv = model(x)
            >>> model.step()
    """
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints,
                 gl: Optional[WarmStartGradientLayer] = None, finetune: Optional[bool] = True, num_head_layers=2):
        super(PoseResNet04, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv2 = make_head(num_head_layers, feature_dim, num_keypoints)
        self.finetune = finetune
        self.gl_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) if gl is None else gl

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints):
        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
            ])
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return layers

    def forward(self, x):
        x = self.backbone(x)
        f = self.upsampling(x)
        f_adv = self.gl_layer(f)
        y = self.head(f)
        y_adv = self.head_adv(f_adv)
        fea2, y_adv2 = self.head_adv2(f)

        return y, y_adv, y_adv2, fea2

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
            {'params': self.head_adv.parameters(), 'lr': lr},
            {'params': self.head_adv2.parameters(), 'lr': lr},
        ]

    def step(self):
        """Call step() each iteration during training.
        Will increase :math:`\lambda` in GL layer.
        """
        self.gl_layer.step()
        
class PoseResNet05(nn.Module):
    """
    Pose ResNet for RegDA has one backbone, one upsampling, while two regression heads.

    Args:
        backbone (torch.nn.Module): Backbone to extract 2-d features from data
        upsampling (torch.nn.Module): Layer to upsample image feature to heatmap size
        feature_dim (int): The dimension of the features from upsampling layer.
        num_keypoints (int): Number of keypoints
        gl (WarmStartGradientLayer):
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: True
        num_head_layers (int): Number of head layers. Default: 2

    Inputs:
        - x (tensor): input data

    Outputs:
        - outputs: logits outputs by the main regressor
        - outputs_adv: logits outputs by the adversarial regressor

    Shapes:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, K, H, W)`, where K means the number of keypoints.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

            >>> # x is inputs, model is an PoseResNet
            >>> outputs, outputs_adv = model(x)
            >>> model.step()
    """
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints,
                 gl: Optional[WarmStartGradientLayer] = None, finetune: Optional[bool] = True, num_head_layers=2):
        super(PoseResNet05, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv2 = make_head2(num_head_layers, feature_dim, num_keypoints)
        self.finetune = finetune
        self.gl_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) if gl is None else gl

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints):
        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
            ])
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return layers

    def forward(self, x):
        x = self.backbone(x)
        f = self.upsampling(x)
        f_adv = self.gl_layer(f)
        y = self.head(f)
        y_adv = self.head_adv(f_adv)
        fea2, y_adv2 = self.head_adv2(f)

        if self.training:
            return y, y_adv, y_adv2, fea2
        else:
            return y_adv2

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
            {'params': self.head_adv.parameters(), 'lr': lr},
            {'params': self.head_adv2.parameters(), 'lr': lr},
        ]

    def step(self):
        """Call step() each iteration during training.
        Will increase :math:`\lambda` in GL layer.
        """
        self.gl_layer.step()
        
class PoseResNet06(nn.Module):
    """
    Pose ResNet for RegDA has one backbone, one upsampling, while two regression heads.

    Args:
        backbone (torch.nn.Module): Backbone to extract 2-d features from data
        upsampling (torch.nn.Module): Layer to upsample image feature to heatmap size
        feature_dim (int): The dimension of the features from upsampling layer.
        num_keypoints (int): Number of keypoints
        gl (WarmStartGradientLayer):
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: True
        num_head_layers (int): Number of head layers. Default: 2

    Inputs:
        - x (tensor): input data

    Outputs:
        - outputs: logits outputs by the main regressor
        - outputs_adv: logits outputs by the adversarial regressor

    Shapes:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, K, H, W)`, where K means the number of keypoints.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

            >>> # x is inputs, model is an PoseResNet
            >>> outputs, outputs_adv = model(x)
            >>> model.step()
    """
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints,
                 gl: Optional[WarmStartGradientLayer] = None, finetune: Optional[bool] = True, num_head_layers=2):
        super(PoseResNet06, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv2 = make_head2(num_head_layers, feature_dim, num_keypoints)
        self.finetune = finetune
        self.gl_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) if gl is None else gl

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints):
        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
            ])
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return layers

    def forward(self, x):
        x = self.backbone(x)
        f = self.upsampling(x)
        f_adv = self.gl_layer(f)
        y = self.head(f)
        y_adv = self.head_adv(f_adv)
        fea2, y_adv2 = self.head_adv2(f)


        return y, y_adv, y_adv2, fea2


    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
            {'params': self.head_adv.parameters(), 'lr': lr},
            {'params': self.head_adv2.parameters(), 'lr': lr},
        ]

    def step(self):
        """Call step() each iteration during training.
        Will increase :math:`\lambda` in GL layer.
        """
        self.gl_layer.step()
        
        
class PoseResNet07(nn.Module):
    """
    Pose ResNet for RegDA has one backbone, one upsampling, while two regression heads.

    Args:
        backbone (torch.nn.Module): Backbone to extract 2-d features from data
        upsampling (torch.nn.Module): Layer to upsample image feature to heatmap size
        feature_dim (int): The dimension of the features from upsampling layer.
        num_keypoints (int): Number of keypoints
        gl (WarmStartGradientLayer):
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: True
        num_head_layers (int): Number of head layers. Default: 2

    Inputs:
        - x (tensor): input data

    Outputs:
        - outputs: logits outputs by the main regressor
        - outputs_adv: logits outputs by the adversarial regressor

    Shapes:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, K, H, W)`, where K means the number of keypoints.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

            >>> # x is inputs, model is an PoseResNet
            >>> outputs, outputs_adv = model(x)
            >>> model.step()
    """
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints,
                 gl: Optional[WarmStartGradientLayer] = None, finetune: Optional[bool] = True, num_head_layers=2):
        super(PoseResNet07, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv2 = make_head(num_head_layers, feature_dim, num_keypoints)
        self.finetune = finetune
        self.gl_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) if gl is None else gl

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints):
        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
            ])
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return layers

    def forward(self, x):
        x = self.backbone(x)
        f = self.upsampling(x)
        f_adv = self.gl_layer(f)
        y = self.head(f)
        y_adv = self.head_adv(f_adv)
        fea2, y_adv2 = self.head_adv2(f)

        if self.training:
            return y, y_adv, y_adv2, fea2
        else:
            return y_adv2

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
            {'params': self.head_adv.parameters(), 'lr': lr},
            {'params': self.head_adv2.parameters(), 'lr': lr},
        ]

    def step(self):
        """Call step() each iteration during training.
        Will increase :math:`\lambda` in GL layer.
        """
        self.gl_layer.step()
        
class PoseResNet08(nn.Module):
    """
    Pose ResNet for RegDA has one backbone, one upsampling, while two regression heads.

    Args:
        backbone (torch.nn.Module): Backbone to extract 2-d features from data
        upsampling (torch.nn.Module): Layer to upsample image feature to heatmap size
        feature_dim (int): The dimension of the features from upsampling layer.
        num_keypoints (int): Number of keypoints
        gl (WarmStartGradientLayer):
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: True
        num_head_layers (int): Number of head layers. Default: 2

    Inputs:
        - x (tensor): input data

    Outputs:
        - outputs: logits outputs by the main regressor
        - outputs_adv: logits outputs by the adversarial regressor

    Shapes:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, K, H, W)`, where K means the number of keypoints.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

            >>> # x is inputs, model is an PoseResNet
            >>> outputs, outputs_adv = model(x)
            >>> model.step()
    """
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints,
                 gl: Optional[WarmStartGradientLayer] = None, finetune: Optional[bool] = True, num_head_layers=2):
        super(PoseResNet08, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv2 = make_head(num_head_layers, feature_dim, num_keypoints)
        self.finetune = finetune
        self.gl_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) if gl is None else gl

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints):
        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
            ])
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return layers

    def forward(self, x):
        x = self.backbone(x)
        f = self.upsampling(x)
        f_adv = self.gl_layer(f)
        y = self.head(f)
        y_adv = self.head_adv(f_adv)
        fea2, y_adv2 = self.head_adv2(f)


        return y, y_adv, y_adv2, fea2


    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
            {'params': self.head_adv.parameters(), 'lr': lr},
            {'params': self.head_adv2.parameters(), 'lr': lr},
        ]

    def step(self):
        """Call step() each iteration during training.
        Will increase :math:`\lambda` in GL layer.
        """
        self.gl_layer.step()
        
        
class PoseResNet09(nn.Module):
    """
    Pose ResNet for RegDA has one backbone, one upsampling, while two regression heads.

    Args:
        backbone (torch.nn.Module): Backbone to extract 2-d features from data
        upsampling (torch.nn.Module): Layer to upsample image feature to heatmap size
        feature_dim (int): The dimension of the features from upsampling layer.
        num_keypoints (int): Number of keypoints
        gl (WarmStartGradientLayer):
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: True
        num_head_layers (int): Number of head layers. Default: 2

    Inputs:
        - x (tensor): input data

    Outputs:
        - outputs: logits outputs by the main regressor
        - outputs_adv: logits outputs by the adversarial regressor

    Shapes:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, K, H, W)`, where K means the number of keypoints.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

            >>> # x is inputs, model is an PoseResNet
            >>> outputs, outputs_adv = model(x)
            >>> model.step()
    """
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints,
                 gl: Optional[WarmStartGradientLayer] = None, finetune: Optional[bool] = True, num_head_layers=2):
        super(PoseResNet09, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv2 = make_head3(num_head_layers, feature_dim, num_keypoints)
        self.finetune = finetune
        self.gl_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) if gl is None else gl

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints):
        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
            ])
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return layers

    def forward(self, x):
        x = self.backbone(x)
        f = self.upsampling(x)
        fea1, fea2, y_adv2 = self.head_adv2(f)
        f_adv = self.gl_layer(fea2)
        y = self.head(fea2)
        y_adv = self.head_adv(f_adv)
        

        if self.training:
            return y, y_adv, y_adv2, fea1
        else:
            return y_adv2

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
            {'params': self.head_adv.parameters(), 'lr': lr},
            {'params': self.head_adv2.parameters(), 'lr': lr},
        ]

    def step(self):
        """Call step() each iteration during training.
        Will increase :math:`\lambda` in GL layer.
        """
        self.gl_layer.step()
        
class PoseResNet10(nn.Module):
    """
    Pose ResNet for RegDA has one backbone, one upsampling, while two regression heads.

    Args:
        backbone (torch.nn.Module): Backbone to extract 2-d features from data
        upsampling (torch.nn.Module): Layer to upsample image feature to heatmap size
        feature_dim (int): The dimension of the features from upsampling layer.
        num_keypoints (int): Number of keypoints
        gl (WarmStartGradientLayer):
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: True
        num_head_layers (int): Number of head layers. Default: 2

    Inputs:
        - x (tensor): input data

    Outputs:
        - outputs: logits outputs by the main regressor
        - outputs_adv: logits outputs by the adversarial regressor

    Shapes:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, K, H, W)`, where K means the number of keypoints.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

            >>> # x is inputs, model is an PoseResNet
            >>> outputs, outputs_adv = model(x)
            >>> model.step()
    """
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints,
                 gl: Optional[WarmStartGradientLayer] = None, finetune: Optional[bool] = True, num_head_layers=2):
        super(PoseResNet10, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv2 = make_head3(num_head_layers, feature_dim, num_keypoints)
        self.finetune = finetune
        self.gl_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) if gl is None else gl

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints):
        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
            ])
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return layers

    def forward(self, x):
        x = self.backbone(x)
        f = self.upsampling(x)
        fea1, fea2, y_adv2 = self.head_adv2(f)
        f_adv = self.gl_layer(fea2)
        y = self.head(fea2)
        y_adv = self.head_adv(f_adv)
        


        return y, y_adv, y_adv2, fea1


    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
            {'params': self.head_adv.parameters(), 'lr': lr},
            {'params': self.head_adv2.parameters(), 'lr': lr},
        ]

    def step(self):
        """Call step() each iteration during training.
        Will increase :math:`\lambda` in GL layer.
        """
        self.gl_layer.step()
        
class PoseResNet11(nn.Module):
    """
    Pose ResNet for RegDA has one backbone, one upsampling, while two regression heads.

    Args:
        backbone (torch.nn.Module): Backbone to extract 2-d features from data
        upsampling (torch.nn.Module): Layer to upsample image feature to heatmap size
        feature_dim (int): The dimension of the features from upsampling layer.
        num_keypoints (int): Number of keypoints
        gl (WarmStartGradientLayer):
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: True
        num_head_layers (int): Number of head layers. Default: 2

    Inputs:
        - x (tensor): input data

    Outputs:
        - outputs: logits outputs by the main regressor
        - outputs_adv: logits outputs by the adversarial regressor

    Shapes:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, K, H, W)`, where K means the number of keypoints.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

            >>> # x is inputs, model is an PoseResNet
            >>> outputs, outputs_adv = model(x)
            >>> model.step()
    """
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints,
                 gl: Optional[WarmStartGradientLayer] = None, finetune: Optional[bool] = True, num_head_layers=2):
        super(PoseResNet11, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv2 = make_head3(num_head_layers, feature_dim, num_keypoints)
        self.finetune = finetune
        self.gl_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) if gl is None else gl

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints):
        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
            ])
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return layers

    def forward(self, x):
        x = self.backbone(x)
        f = self.upsampling(x)
        fea1, fea2, y_adv2 = self.head_adv2(f)
        f_adv = self.gl_layer(fea2)
        y = self.head(fea2)
        y_adv = self.head_adv(f_adv)
        

        if self.training:
            return y, y_adv, y_adv2, fea2
        else:
            return y_adv2

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
            {'params': self.head_adv.parameters(), 'lr': lr},
            {'params': self.head_adv2.parameters(), 'lr': lr},
        ]

    def step(self):
        """Call step() each iteration during training.
        Will increase :math:`\lambda` in GL layer.
        """
        self.gl_layer.step()
        
class PoseResNet12(nn.Module):
    """
    Pose ResNet for RegDA has one backbone, one upsampling, while two regression heads.

    Args:
        backbone (torch.nn.Module): Backbone to extract 2-d features from data
        upsampling (torch.nn.Module): Layer to upsample image feature to heatmap size
        feature_dim (int): The dimension of the features from upsampling layer.
        num_keypoints (int): Number of keypoints
        gl (WarmStartGradientLayer):
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: True
        num_head_layers (int): Number of head layers. Default: 2

    Inputs:
        - x (tensor): input data

    Outputs:
        - outputs: logits outputs by the main regressor
        - outputs_adv: logits outputs by the adversarial regressor

    Shapes:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, K, H, W)`, where K means the number of keypoints.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

            >>> # x is inputs, model is an PoseResNet
            >>> outputs, outputs_adv = model(x)
            >>> model.step()
    """
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints,
                 gl: Optional[WarmStartGradientLayer] = None, finetune: Optional[bool] = True, num_head_layers=2):
        super(PoseResNet12, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        self.head = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv = self._make_head(num_head_layers, feature_dim, num_keypoints)
        self.head_adv2 = make_head3(num_head_layers, feature_dim, num_keypoints)
        self.finetune = finetune
        self.gl_layer = WarmStartGradientLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False) if gl is None else gl

    @staticmethod
    def _make_head(num_layers, channel_dim, num_keypoints):
        layers = []
        for i in range(num_layers-1):
            layers.extend([
                nn.Conv2d(channel_dim, channel_dim, 3, 1, 1),
                nn.BatchNorm2d(channel_dim),
                nn.ReLU(),
            ])
        layers.append(
            nn.Conv2d(
                in_channels=channel_dim,
                out_channels=num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )
        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)
        return layers

    def forward(self, x):
        x = self.backbone(x)
        f = self.upsampling(x)
        fea1, fea2, y_adv2 = self.head_adv2(f)
        f_adv = self.gl_layer(fea2)
        y = self.head(fea2)
        y_adv = self.head_adv(f_adv)
        


        return y, y_adv, y_adv2, fea2


    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head.parameters(), 'lr': lr},
            {'params': self.head_adv.parameters(), 'lr': lr},
            {'params': self.head_adv2.parameters(), 'lr': lr},
        ]

    def step(self):
        """Call step() each iteration during training.
        Will increase :math:`\lambda` in GL layer.
        """
        self.gl_layer.step()




class DomainClassifier(nn.Module):
    def __init__(self, input_dim=256, ndf=64, with_bias=False):
        super(DomainClassifier, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, ndf, kernel_size=4, stride=2, padding=1, bias=with_bias)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=with_bias)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=with_bias)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=with_bias)
        self.conv5 = nn.Conv2d(ndf * 8, ndf * 16, kernel_size=4, stride=2, padding=1, bias=with_bias)
        self.conv6 = nn.Conv2d(ndf * 16, 1, kernel_size=2, stride=1, bias=with_bias)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.conv5(x)
        x = self.leaky_relu(x)
        x = self.conv6(x)
        return x