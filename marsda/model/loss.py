# ------------------------------------------------------------------------------
# Modified from https://github.com/thuml/Regressive-Domain-Adaptation-for-Unsupervised-Keypoint-Detection
# ------------------------------------------------------------------------------


import torch.nn as nn
import torch.nn.functional as F
import torch
from scipy.stats import wasserstein_distance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mixup(img_src, hm_src, weights_src, img_trg, hm_trg, weights_trg, beta):
    m = torch.distributions.beta.Beta(torch.tensor(beta), torch.tensor(beta))
    mix = m.rsample(sample_shape=(img_src.size(0), 1, 1, 1))
    # keep the max value such that the domain labels does not change
    mix = torch.max(mix, 1 - mix)
    mix = mix.to(device)
    img_src_mix = img_src * mix + img_trg * (1. - mix)
    hm_src_mix = hm_src * mix + hm_trg * (1. - mix)
    img_trg_mix = img_trg * mix + img_src * (1. - mix)
    hm_trg_mix = hm_trg * mix + hm_src * (1. - mix)
    weights = torch.max(weights_src, weights_trg)
    return img_src_mix, hm_src_mix, weights, img_trg_mix, hm_trg_mix, weights


class JointsMSELoss(nn.Module):
    """
    Typical MSE loss for keypoint detection.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``

    Inputs:
        - output (tensor): heatmap predictions
        - target (tensor): heatmap labels
        - target_weight (tensor): whether the keypoint is visible. All keypoint is visible if None. Default: None.

    Shape:
        - output: :math:`(minibatch, K, H, W)` where K means the number of keypoints,
          H and W is the height and width of the heatmap respectively.
        - target: :math:`(minibatch, K, H, W)`.
        - target_weight: :math:`(minibatch, K)`.
        - Output: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(minibatch, K)`.

    """
    def __init__(self, reduction='mean'):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.reduction = reduction

    def forward(self, output, target, target_weight=None):
        B, K, _, _ = output.shape
        heatmaps_pred = output.reshape((B, K, -1))
        heatmaps_gt = target.reshape((B, K, -1))
        loss = self.criterion(heatmaps_pred, heatmaps_gt) * 0.5
        if target_weight is not None:
            loss = loss * target_weight.view((B, K, 1))
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss.mean(dim=-1)


class JointsMSELoss0(nn.Module):
    """
    Typical MSE loss for keypoint detection.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``

    Inputs:
        - output (tensor): heatmap predictions
        - target (tensor): heatmap labels
        - target_weight (tensor): whether the keypoint is visible. All keypoint is visible if None. Default: None.

    Shape:
        - output: :math:`(minibatch, K, H, W)` where K means the number of keypoints,
          H and W is the height and width of the heatmap respectively.
        - target: :math:`(minibatch, K, H, W)`.
        - target_weight: :math:`(minibatch, K)`.
        - Output: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(minibatch, K)`.

    """
    def __init__(self, reduction='mean'):
        super(JointsMSELoss0, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.reduction = reduction

    def forward(self, output, target, target_weight=None):
        B, K, _, _ = output.shape
        heatmaps_pred = output.reshape((B, K, -1)) + 1e-7
        heatmaps_pred = heatmaps_pred / heatmaps_pred.sum(dim=-1, keepdims=True)
        heatmaps_gt = target.reshape((B, K, -1)) + 1e-7
        heatmaps_gt = heatmaps_gt / heatmaps_gt.sum(dim=-1, keepdims=True)
        loss = self.criterion(heatmaps_pred, heatmaps_gt) * 0.5
        if target_weight is not None:
            loss = loss * target_weight.view((B, K, 1))
            
        #loss = torch.sum(loss, dim=2)
      
        
        if self.reduction == 'mean':
           return loss.mean()
        elif self.reduction == 'none':
            return loss.mean(dim=-1)


class JointsKLLoss(nn.Module):
    """
    KL Divergence for keypoint detection proposed by
    `Regressive Domain Adaptation for Unsupervised Keypoint Detection <https://arxiv.org/abs/2103.06175>`_.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``

    Inputs:
        - output (tensor): heatmap predictions
        - target (tensor): heatmap labels
        - target_weight (tensor): whether the keypoint is visible. All keypoint is visible if None. Default: None.

    Shape:
        - output: :math:`(minibatch, K, H, W)` where K means the number of keypoints,
          H and W is the height and width of the heatmap respectively.
        - target: :math:`(minibatch, K, H, W)`.
        - target_weight: :math:`(minibatch, K)`.
        - Output: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(minibatch, K)`.

    """
    def __init__(self, reduction='mean', epsilon=0.):
        super(JointsKLLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, output, target, target_weight=None):
        B, K, _, _ = output.shape
        heatmaps_pred = output.reshape((B, K, -1))
        heatmaps_pred = F.log_softmax(heatmaps_pred, dim=-1)
        heatmaps_gt = target.reshape((B, K, -1))
        heatmaps_gt = heatmaps_gt + self.epsilon
        heatmaps_gt = heatmaps_gt / heatmaps_gt.sum(dim=-1, keepdims=True)
        loss = self.criterion(heatmaps_pred, heatmaps_gt).sum(dim=-1)
        if target_weight is not None:
            loss = loss * target_weight.view((B, K))
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss.mean(dim=-1)
            
class JointsKLLoss5(nn.Module):
    """
    KL Divergence for keypoint detection proposed by
    `Regressive Domain Adaptation for Unsupervised Keypoint Detection <https://arxiv.org/abs/2103.06175>`_.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``

    Inputs:
        - output (tensor): heatmap predictions
        - target (tensor): heatmap labels
        - target_weight (tensor): whether the keypoint is visible. All keypoint is visible if None. Default: None.

    Shape:
        - output: :math:`(minibatch, K, H, W)` where K means the number of keypoints,
          H and W is the height and width of the heatmap respectively.
        - target: :math:`(minibatch, K, H, W)`.
        - target_weight: :math:`(minibatch, K)`.
        - Output: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(minibatch, K)`.

    """
    def __init__(self, reduction='mean', epsilon=0.):
        super(JointsKLLoss5, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, output, target, target_weight=None):
        B, K, _, _ = output.shape
        
        f1 = (output / torch.max(output)).detach()
        f2 = (target / torch.max(target)).detach()
        w1 = torch.mul(f1, f2)
        w2 = torch.sum(w1, dim=2)
        w3 = torch.sum(w2, dim=2)
        w4 = w3 / torch.max(w3)
        w5 = w4.unsqueeze(-1)
        w5 = w5.unsqueeze(-1)
        output = torch.mul(output, w5)
        target = torch.mul(target, w5)
        
        heatmaps_pred = output.reshape((B, K, -1))
        heatmaps_pred = F.log_softmax(heatmaps_pred, dim=-1)
        heatmaps_gt = target.reshape((B, K, -1))
        heatmaps_gt = heatmaps_gt + self.epsilon
        heatmaps_gt = heatmaps_gt / heatmaps_gt.sum(dim=-1, keepdims=True)
        loss = self.criterion(heatmaps_pred, heatmaps_gt).sum(dim=-1)

        

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss.mean(dim=-1)





def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        # ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

def update_ema_variables2(main, ema, alpha, global_step):
        state_dict_main = main.state_dict()
        state_dict_ema = ema.state_dict()
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for (k_main, v_main), (k_ema, v_ema) in zip(state_dict_main.items(), state_dict_ema.items()):
            assert k_main == k_ema, "state_dict names are different!"
            assert v_main.shape == v_ema.shape, "state_dict shapes are different!"
            if 'num_batches_tracked' in k_ema:
                v_ema.copy_(v_main)
            else:
                v_ema.copy_(v_ema * alpha + (1. - alpha) * v_main)

def update_ema_variables3(main, ema, alpha):
        state_dict_main = main.state_dict()
        state_dict_ema = ema.state_dict()
        for (k_main, v_main), (k_ema, v_ema) in zip(state_dict_main.items(), state_dict_ema.items()):
            assert k_main == k_ema, "state_dict names are different!"
            assert v_main.shape == v_ema.shape, "state_dict shapes are different!"
            if 'num_batches_tracked' in k_ema:
                v_ema.copy_(v_main)
            else:
                v_ema.copy_(v_ema * alpha + (1. - alpha) * v_main)

def update_ema_variables5(main, ema, momentum):
        state_dict_main = main.state_dict()
        state_dict_ema = ema.state_dict()
        for (k_main, v_main), (k_ema, v_ema) in zip(state_dict_main.items(), state_dict_ema.items()):
            assert k_main == k_ema, "state_dict names are different!"
            assert v_main.shape == v_ema.shape, "state_dict shapes are different!"
            if 'num_batches_tracked' in k_ema:
                v_ema.copy_(v_main)
            else:
                v_ema.copy_(v_ema * momentum + (1. - momentum) * v_main)



def mt_loss(pre, label, weight, k):


   # global pre1, label1
    mtloss = torch.nn.MSELoss()
    if k < 100:
        pre1 = pre[:, 0, :, :].unsqueeze(1)
        label1 = label[:, 0, :, :].unsqueeze(1)
        weight1 = weight[:, 0, :].unsqueeze(1)
        return mtloss(pre1, label1)
    if k<200 and k > 99:
        a = [0, 1, 5, 9, 13, 17]
        pre1 = pre[:, a, :, :]
        label1 = label[:, a, :, :]
        weight1 = weight[:, a, :]
        return mtloss(pre1, label1)
    if k<300 and k > 199:
        a = [0, 1, 2, 5, 6, 9, 10, 13, 14, 17, 18]
        pre1 = pre[:, a, :, :]
        label1 = label[:, a, :, :]
        weight1 = weight[:, a, :]
        return mtloss(pre1, label1)
    if k < 400 and k > 299:
        a = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
        pre1 = pre[:, a, :, :]
        label1 = label[:, a, :, :]
        weight1 = weight[:, a, :]
        return mtloss(pre1, label1)
    if k >399 :
        pre1 = pre
        label1 = label

        return mtloss(pre1, label1)
  #  klloss = JointsKLLoss(epsilon=1e-7)
#    klloss = torch.nn.MSELoss()
#    loss = klloss(pre1, label1)

#    return loss

def wasserstein(output, target):
    B, K, _, _ = output.shape
    heatmaps_pred = output.reshape((B, K, -1))
    heatmaps_pred = F.log_softmax(heatmaps_pred, dim=-1)
    heatmaps_gt = target.reshape((B, K, -1))
    heatmaps_gt = F.log_softmax(heatmaps_gt, dim=-1)
    w = wasserstein_distance(heatmaps_pred, heatmaps_gt)
    
    return w
    
class wasserstein2(nn.Module):
   
    def __init__(self, reduction='mean', epsilon=0.):
        super(JointsKLLoss, self).__init__()
        self.wasserstein2 = wasserstein_distance
     

    def forward(self, output, target):
        B, K, _, _ = output.shape
        heatmaps_pred = output.reshape((B, K, -1))
        heatmaps_pred = F.log_softmax(heatmaps_pred, dim=-1)
        heatmaps_gt = target.reshape((B, K, -1))
        heatmaps_gt = F.log_softmax(heatmaps_gt, dim=-1)
        w = wasserstein_distance(heatmaps_pred, heatmaps_gt)
       
        return w
        
def loss1(feature, pre):
    b, c, h, w = feature.shape
    th = 6
    fea_c = []
    fea_c0 = []
   # pre = pre/4
    for i in range(b):
        feature1 = feature[i]
        pre0 = pre[i]
        fea_c0 = []
        for j in range(21):
            xy = pre0[j]
            x = xy[0]
            y = xy[1]
            upper = y + 6
            down = y - 6
            left = x - 6
            right = x + 6
            down = torch.clamp(down, min=0).int()
            left = torch.clamp(left, min=0).int()
            upper = torch.clamp(upper, max=63).int()
            right = torch.clamp(right, max=63).int()
            f1 = feature1[:, left:right, down:upper]
            a0 = f1.shape
            f2 = torch.sum(f1, dim=1)
            f3 = torch.sum(f2, dim=1)
            f4 = f3 / (13 * 13)
            af4 = f4.shape
            fea_c0.append(f4)
        fea_c1 = torch.stack(fea_c0)
        dd = fea_c1.shape
        fea_c.append(fea_c1)

    fea_c = torch.stack(fea_c)
    return fea_c

def loss3(f1, f2, pre1, pre2):
    epsilon=1e-7
    fea_c1 = loss1(f1, pre1)
    fea_c1 = F.log_softmax(fea_c1, dim=-1)
    fea_c2 = loss1(f2, pre2)
    fea_c2 = fea_c2 + 10e-7
    fea_c2 = fea_c2 / fea_c2.sum(dim=-1, keepdims=True)
  #  fea_c2 = F.log_softmax(fea_c2, dim=-1) + epsilon
    
    criterion = nn.KLDivLoss(reduction='none')
    loss = criterion(fea_c1, fea_c2).sum(dim=-1)
    return loss.mean()
 
 
class lossx(nn.Module):
    def __init__(self, reduction='mean', epsilon=0.):
        super(lossx, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.reduction = reduction
        self.epsilon = epsilon
        self.reset()
    def reset(self):
        self.val1 = 0.
        self.val2 = 0.
    def updata(self, val1, val2):
        self.val1 = val1
        self.val2 = val2
    def loss1(self, feature, pre, fea):
        b, c, h, w = feature.shape
        th = 6
        fea_c = []
        fea_c0 = []
        # pre = pre/4
        m = 0.999

        for i in range(b):
            feature1 = feature[i]
            pre0 = pre[i]
            fea_c0 = []
            for j in range(21):
                xy = pre0[j]
                x = xy[0]
                y = xy[1]
                upper = y + 6
                down = y - 6
                left = x - 6
                right = x + 6
                down = torch.clamp(down, min=0).int()
                left = torch.clamp(left, min=0).int()
                upper = torch.clamp(upper, max=63).int()
                right = torch.clamp(right, max=63).int()
                s1 = upper - down + 1
                s2 = right - left + 1

                f1 = feature1[:, left:right, down:upper]
                a0 = f1.shape
                f2 = torch.sum(f1, dim=1)
                f3 = torch.sum(f2, dim=1)
                f4 = f3 / (s1 * s2)
                af4 = f4.shape
                fea_c0.append(f4)
            fea_c1 = torch.stack(fea_c0)
            dd = fea_c1.shape
            #  fea_c2 = fea_c1 / (b*13*13)
            fea_c.append(fea_c1)

        fea_c = torch.stack(fea_c)

        d1 = fea_c.shape
        fea_c = torch.sum(fea_c, dim=0)
        fea_c = fea_c / (b)
        d2 = fea_c.shape
        #   fea_c = F.log_softmax(fea_c, dim=-1)


        # if fea == 0:
        #     return fea_c
        # else:
        #     fea_c = m*fea_c + (1-m)*fea
        #     return fea_c

        fea_c = m * fea_c + (1 - m) * fea
        return fea_c

    def forward(self, f1, f2, pre1, pre2):
        fea1 = self.val1
        fea_c1 = self.loss1(f1, pre1, fea1)
        self.val1 = fea_c1.detach()
        fea_c1 = F.log_softmax(fea_c1, dim=-1)


        fea2 = self.val2
        fea_c2 = self.loss1(f2, pre2, fea2)
        self.val2 = fea_c2.detach()
        fea_c2 = fea_c2 + 0.
        fea_c2 = fea_c2 / fea_c2.sum(dim=-1, keepdims=True)

        loss = self.criterion(fea_c1, fea_c2).sum(dim=-1)

        return loss.mean()


class lossx2(nn.Module):
    def __init__(self, reduction='mean', epsilon=0.):
        super(lossx2, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        self.reset()
    def reset(self):
        self.val1 = 0.
        self.val2 = 0.
    def updata(self, val1, val2):
        self.val1 = val1
        self.val2 = val2
    def loss1(self, feature, pre, fea):
        b, c, h, w = feature.shape
        th = 6
        fea_c = []
        fea_c0 = []
        # pre = pre/4
        m = 0.999

        for i in range(b):
            feature1 = feature[i]
            pre0 = pre[i]
            fea_c0 = []
            for j in range(21):
                xy = pre0[j]
                x = xy[0]
                y = xy[1]
                upper = y + 6
                down = y - 6
                left = x - 6
                right = x + 6
                down = torch.clamp(down, min=0).int()
                left = torch.clamp(left, min=0).int()
                upper = torch.clamp(upper, max=63).int()
                right = torch.clamp(right, max=63).int()
                s1 = upper - down + 1
                s2 = right - left + 1

                f1 = feature1[:, left:right, down:upper]
                a0 = f1.shape
                f2 = torch.sum(f1, dim=1)
                f3 = torch.sum(f2, dim=1)
                f4 = f3 / (s1 * s2)
                af4 = f4.shape
                fea_c0.append(f4)
            fea_c1 = torch.stack(fea_c0)
            dd = fea_c1.shape
            #  fea_c2 = fea_c1 / (b*13*13)
            fea_c.append(fea_c1)

        fea_c = torch.stack(fea_c)

        d1 = fea_c.shape
        fea_c = torch.sum(fea_c, dim=0)
        fea_c = fea_c / (b)
        d2 = fea_c.shape
        #   fea_c = F.log_softmax(fea_c, dim=-1)


        # if fea == 0:
        #     return fea_c
        # else:
        #     fea_c = m*fea_c + (1-m)*fea
        #     return fea_c

        fea_c = m * fea_c + (1 - m) * fea
        return fea_c

    def forward(self, f1, f2, pre1, pre2):
        fea1 = self.val1
        fea_c1 = self.loss1(f1, pre1, fea1)
        self.val1 = fea_c1.detach()
    #    fea_c1 = F.log_softmax(fea_c1, dim=-1)


        fea2 = self.val2
        fea_c2 = self.loss1(f2, pre2, fea2)
        self.val2 = fea_c2.detach()
    #    fea_c2 = fea_c2
    #    fea_c2 = fea_c2 / fea_c2.sum(dim=-1, keepdims=True)

        loss = mmd_rbf(fea_c1, fea_c2)

        # fea_c1 = F.log_softmax(fea_c1, dim=-1)
        # fea_c2 = fea_c2 / fea_c2.sum(dim=-1, keepdims=True)
        #
        # loss = self.criterion(fea_c1, fea_c2).sum(dim=-1)

        return loss

class lossx3(nn.Module):
    def __init__(self, reduction='mean', epsilon=0.):
        super(lossx3, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        self.reset()
    def reset(self):
        self.val1 = 0.
        self.val2 = 0.
    def updata(self, val1, val2):
        self.val1 = val1
        self.val2 = val2
    def loss1(self, feature, pre, fea):
        b, c, h, w = feature.shape
        th = 6
        fea_c = []
        fea_c0 = []
        # pre = pre/4
        m = 0.999

        for i in range(b):
            feature1 = feature[i]
            pre0 = pre[i]
            fea_c0 = []
            for j in range(21):
                xy = pre0[j]
                x = xy[0]
                y = xy[1]
                upper = y + 6
                down = y - 6
                left = x - 6
                right = x + 6
                down = torch.clamp(down, min=0).int()
                left = torch.clamp(left, min=0).int()
                upper = torch.clamp(upper, max=63).int()
                right = torch.clamp(right, max=63).int()
                s1 = upper - down + 1
                s2 = right - left + 1

                f1 = feature1[:, left:right, down:upper]
                a0 = f1.shape
                f2 = torch.sum(f1, dim=1)
                f3 = torch.sum(f2, dim=1)
                f4 = f3 / (s1 * s2)
                af4 = f4.shape
                fea_c0.append(f4)
            fea_c1 = torch.stack(fea_c0)
            dd = fea_c1.shape
            #  fea_c2 = fea_c1 / (b*13*13)
            fea_c.append(fea_c1)

        fea_c = torch.stack(fea_c)

        d1 = fea_c.shape
        fea_c = torch.sum(fea_c, dim=0)
        fea_c = fea_c / (b)
        d2 = fea_c.shape
        #   fea_c = F.log_softmax(fea_c, dim=-1)


        # if fea == 0:
        #     return fea_c
        # else:
        #     fea_c = m*fea_c + (1-m)*fea
        #     return fea_c

        fea_c = m * fea_c + (1 - m) * fea
        return fea_c

    def forward(self, f1, f2, pre1, pre2):
        fea1 = self.val1
        fea_c1 = self.loss1(f1, pre1, fea1)
        self.val1 = fea_c1.detach()
   #     fea_c1 = fea_c1 / fea_c1.sum(dim=-1, keepdims=True)



        fea2 = self.val2
        fea_c2 = self.loss1(f2, pre2, fea2)
        self.val2 = fea_c2.detach()
   #     fea_c2 = fea_c2
   #     fea_c2 = fea_c2 / fea_c2.sum(dim=-1, keepdims=True)

    #    loss = mmd_rbf(fea_c1, fea_c2)

        loss = F.kl_div(fea_c1.softmax(dim=-1).log(), fea_c2.softmax(dim=-1), reduction='sum')

        # fea_c1 = F.log_softmax(fea_c1, dim=-1)
        # fea_c2 = fea_c2 / fea_c2.sum(dim=-1, keepdims=True)
        #
        # loss = self.criterion(fea_c1, fea_c2).sum(dim=-1)

        return loss


class lossx4(nn.Module):
    def __init__(self, reduction='mean', epsilon=0.):
        super(lossx4, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.reduction = reduction
        self.epsilon = epsilon
        self.reset()
    def reset(self):
        self.val1 = 0.
        self.val2 = 0.
    def updata(self, val1, val2):
        self.val1 = val1
        self.val2 = val2
    def loss1(self, feature, pre):
        b, c, h, w = feature.shape
        th = 6
        fea_c = []
        fea_c0 = []
        # pre = pre/4
        m = 0.999

        for i in range(b):
            feature1 = feature[i]
            pre0 = pre[i]
            fea_c0 = []
            for j in range(21):
                xy = pre0[j]
                x = xy[0]
                y = xy[1]
                upper = y + 4
                down = y - 4
                left = x - 4
                right = x + 4
                down = torch.clamp(down, min=0).int()
                left = torch.clamp(left, min=0).int()
                upper = torch.clamp(upper, max=63).int()
                right = torch.clamp(right, max=63).int()
                s1 = upper - down + 1
                s2 = right - left + 1

                f1 = feature1[:, left:right, down:upper]
                a0 = f1.shape
                f2 = f1.reshape(c, -1)
                f4 = torch.mean(f2, dim=1)
             #   f2 = torch.sum(f1, dim=1)
             #   f3 = torch.sum(f2, dim=1)
             #   f4 = f3 / (s1 * s2)
                af4 = f4.shape
                fea_c0.append(f4)
            fea_c1 = torch.stack(fea_c0)
            dd = fea_c1.shape
            #  fea_c2 = fea_c1 / (b*13*13)
            fea_c.append(fea_c1)

        fea_c = torch.stack(fea_c)

        d1 = fea_c.shape
        fea_c = torch.sum(fea_c, dim=0)
        fea_c = fea_c / (b)
        d2 = fea_c.shape
        
        return fea_c

    def forward(self, f1, f2, pre1, pre2):
    
        f10 = (f1 / torch.max(f1)).detach()
        f20 = (f2 / torch.max(f2)).detach()
        w1 = torch.mul(f10, f20)
        w2 = torch.sum(w1, dim=2)
        w3 = torch.sum(w2, dim=2)
        w4 = w3 / torch.max(w3)
        w5 = w4.unsqueeze(-1)
        w5 = w5.unsqueeze(-1)
        f1 = torch.mul(f1, w5)
        f2 = torch.mul(f2, w5)
    
        m = 0.999
        fea1 = self.val1
        fea_c1 = self.loss1(f1, pre1)
        fea_c1 = m*fea_c1 + (1-m)*fea1
        self.val1 = fea_c1.detach()

        fea2 = self.val2
        fea_c2 = self.loss1(f2, pre2)
        fea_c2 = m*fea_c2 + (1-m)*fea2
        self.val2 = fea_c2.detach()

        loss = self.criterion(fea_c1, fea_c2)


        return loss


class lossx5(nn.Module):
    def __init__(self, reduction='mean', epsilon=0.):
        super(lossx5, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.reduction = reduction
        self.epsilon = epsilon
        self.reset()
    def reset(self):
        self.val1 = 0.
        self.val2 = 0.
    def updata(self, val1, val2):
        self.val1 = val1
        self.val2 = val2
    def loss1(self, feature, pre):
        b, c, h, w = feature.shape
        th = 6
        fea_c = []
        fea_c0 = []
        # pre = pre/4
        m = 0.999

        for i in range(b):
            feature1 = feature[i]
            pre0 = pre[i]
            fea_c0 = []
            for j in range(21):
                xy = pre0[j]
                x = xy[0]
                y = xy[1]
                upper = y + 6
                down = y - 6
                left = x - 6
                right = x + 6
                down = torch.clamp(down, min=0).int()
                left = torch.clamp(left, min=0).int()
                upper = torch.clamp(upper, max=63).int()
                right = torch.clamp(right, max=63).int()
                s1 = upper - down + 1
                s2 = right - left + 1

                f1 = feature1[:, left:right, down:upper]
                a0 = f1.shape
                f2 = torch.mean(f1)
                fea_c0.append(f2)
            fea_c1 = torch.stack(fea_c0)
            dd = fea_c1.shape
            #  fea_c2 = fea_c1 / (b*13*13)
            fea_c.append(fea_c1)

        fea_c = torch.stack(fea_c)

        d1 = fea_c.shape
        fea_c = torch.mean(fea_c, dim=0)
        d2 = fea_c.shape
        
        return fea_c

    def forward(self, f1, f2, pre1, pre2):
    
        m = 0.999
        fea1 = self.val1
        fea_c1 = self.loss1(f1, pre1)
        self.val1 = fea_c1.detach()

        fea2 = self.val2
        fea_c2 = self.loss1(f2, pre2)
        fea_c2 = m*fea_c2 + (1-m)*fea2
        self.val2 = fea_c2.detach()

        loss = self.criterion(fea_c1, fea_c2)


        return loss
        
        
class lossx6(nn.Module):
    def __init__(self, reduction='mean', epsilon=0.):
        super(lossx6, self).__init__()
        self.criterion = nn.MSELoss(reduction='sum')
        self.reduction = reduction
        self.epsilon = epsilon
        self.reset()
    def reset(self):
        self.val1 = 0.
        self.val2 = 0.
    def updata(self, val1, val2):
        self.val1 = val1
        self.val2 = val2
    def loss1(self, feature, pre):
        b, c, h, w = feature.shape
        th = 6
        fea_c = []
        fea_c0 = []
        # pre = pre/4
        m = 0.999

        for i in range(b):
            feature1 = feature[i]
            pre0 = pre[i]
            fea_c0 = []
            for j in range(21):
                xy = pre0[j]
                x = xy[0]
                y = xy[1]
                upper = y + 4
                down = y - 4
                left = x - 4
                right = x + 4
                down = torch.clamp(down, min=0).int()
                left = torch.clamp(left, min=0).int()
                upper = torch.clamp(upper, max=63).int()
                right = torch.clamp(right, max=63).int()
                s1 = upper - down + 1
                s2 = right - left + 1

                f1 = feature1[:, left:right, down:upper]
                a0 = f1.shape
                f2 = f1.reshape(c, -1)
                f4 = torch.mean(f2, dim=1)
             #   f2 = torch.sum(f1, dim=1)
             #   f3 = torch.sum(f2, dim=1)
             #   f4 = f3 / (s1 * s2)
                af4 = f4.shape
                fea_c0.append(f4)
            fea_c1 = torch.stack(fea_c0)
            dd = fea_c1.shape
            #  fea_c2 = fea_c1 / (b*13*13)
            fea_c.append(fea_c1)

        fea_c = torch.stack(fea_c)

        d1 = fea_c.shape
        fea_c = torch.sum(fea_c, dim=0)
        fea_c = fea_c / (b)
        d2 = fea_c.shape
        
        return fea_c

    def forward(self, f1, f2, pre1, pre2):
    
        f10 = (f1 / torch.max(f1)).detach()
        f20 = (f2 / torch.max(f2)).detach()
        w1 = torch.mul(f10, f20)
        w2 = torch.sum(w1, dim=2)
        w3 = torch.sum(w2, dim=2)
        w4 = w3 / torch.max(w3)
        w5 = w4.unsqueeze(-1)
        w5 = w5.unsqueeze(-1)
        f1 = torch.mul(f1, w5)
        f2 = torch.mul(f2, w5)
    
        m = 0.999
        fea1 = self.val1
        fea_c1 = self.loss1(f1, pre1)

        fea2 = self.val2
        fea_c2 = self.loss1(f2, pre2)
        fea_c2 = (m*fea_c2 + (1-m)*fea2).detach()
        self.val2 = fea_c2

        loss = self.criterion(fea_c1, fea_c2)


        return loss
        
 
class lossx7(nn.Module):
    def __init__(self, reduction='mean', epsilon=0.):
        super(lossx7, self).__init__()
        self.criterion = nn.MSELoss(reduction='sum')
        self.reduction = reduction
        self.epsilon = epsilon
        self.reset()
    def reset(self):
        self.val1 = 0.
        self.val2 = 0.
    def updata(self, val1, val2):
        self.val1 = val1
        self.val2 = val2
    def loss1(self, feature, pre):
        b, c, h, w = feature.shape
        th = 6
        fea_c = []
        fea_c0 = []
        # pre = pre/4
        m = 0.999

        for i in range(b):
            feature1 = feature[i]
            pre0 = pre[i]
            fea_c0 = []
            for j in range(21):
                xy = pre0[j]
                x = xy[0]
                y = xy[1]
                upper = y + 6
                down = y - 6
                left = x - 6
                right = x + 6
                down = torch.clamp(down, min=0).int()
                left = torch.clamp(left, min=0).int()
                upper = torch.clamp(upper, max=63).int()
                right = torch.clamp(right, max=63).int()
                s1 = upper - down + 1
                s2 = right - left + 1

                f1 = feature1[:, left:right, down:upper]
                a0 = f1.shape
                f2 = torch.mean(f1)
                fea_c0.append(f2)
            fea_c1 = torch.stack(fea_c0)
            dd = fea_c1.shape
            #  fea_c2 = fea_c1 / (b*13*13)
            fea_c.append(fea_c1)

        fea_c = torch.stack(fea_c)

        d1 = fea_c.shape
        fea_c = torch.mean(fea_c, dim=0)
        d2 = fea_c.shape
        
        return fea_c

    def forward(self, f1, f2, pre1, pre2):
    
    
        f10 = (f1 / torch.max(f1)).detach()
        f20 = (f2 / torch.max(f2)).detach()
        w1 = torch.mul(f10, f20)
        w2 = torch.sum(w1, dim=2)
        w3 = torch.sum(w2, dim=2)
        w4 = w3 / torch.max(w3)
        w5 = w4.unsqueeze(-1)
        w5 = w5.unsqueeze(-1)
        f1 = torch.mul(f1, w5)
        f2 = torch.mul(f2, w5)
    
        m = 0.999
        fea1 = self.val1
        fea_c1 = self.loss1(f1, pre1)
        self.val1 = fea_c1.detach()

        fea2 = self.val2
        fea_c2 = self.loss1(f2, pre2)
        fea_c2 = (m*fea_c2 + (1-m)*fea2).detach()
        self.val2 = fea_c2

        loss = self.criterion(fea_c1, fea_c2)


        return loss   
        

class JointsMMDLoss(nn.Module):
    """
    KL Divergence for keypoint detection proposed by
    `Regressive Domain Adaptation for Unsupervised Keypoint Detection <https://arxiv.org/abs/2103.06175>`_.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``

    Inputs:
        - output (tensor): heatmap predictions
        - target (tensor): heatmap labels
        - target_weight (tensor): whether the keypoint is visible. All keypoint is visible if None. Default: None.

    Shape:
        - output: :math:`(minibatch, K, H, W)` where K means the number of keypoints,
          H and W is the height and width of the heatmap respectively.
        - target: :math:`(minibatch, K, H, W)`.
        - target_weight: :math:`(minibatch, K)`.
        - Output: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(minibatch, K)`.

    """
    def __init__(self, reduction='mean', epsilon=0.):
        super(JointsMMDLoss, self).__init__()
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, output, target, target_weight=None):
        B, K, _, _ = output.shape
        
        f1 = (output / torch.max(output)).detach()
        f2 = (target / torch.max(target)).detach()
        w1 = torch.mul(f1, f2)
        w2 = torch.sum(w1, dim=2)
        w3 = torch.sum(w2, dim=2)
        w4 = w3 / torch.max(w3)
        w5 = w4.unsqueeze(-1)
        w5 = w5.unsqueeze(-1)
        output = torch.mul(output, w5)
        target = torch.mul(target, w5)
        
        ap = torch.nn.AdaptiveAvgPool2d((1,1))
        fx1 = ap(output).flatten(1)
        aa1 = fx1.shape
        fx2 = ap(target).flatten(1)
        loss = mmd_rbf(fx1, fx2)
        
        
    #    heatmaps_pred = output.reshape((B, K, -1))
    #    heatmaps_pred = F.log_softmax(heatmaps_pred, dim=-1)
    #    heatmaps_gt = target.reshape((B, K, -1))
    #    heatmaps_gt = heatmaps_gt + self.epsilon
   #     heatmaps_gt = heatmaps_gt / heatmaps_gt.sum(dim=-1, keepdims=True)
    #    loss = mmd_rbf(heatmaps_pred, heatmaps_gt)

        return loss


class MMD_loss3(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss3, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])

        B, K, _, _ = source.shape
        source = source.reshape((B, K, -1))
        target = target.reshape((B, K, -1))
        loss_x = 0
        for i in range(K):
            source1 = source[:, i, :]
            target1 = target[:, i, :]
            kernels = guassian_kernel(source1, target1, kernel_mul=self.kernel_mul,
                                      kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = kernels[:batch_size, :batch_size]
            YY = kernels[batch_size:, batch_size:]
            XY = kernels[:batch_size, batch_size:]
            YX = kernels[batch_size:, :batch_size]
            loss = torch.mean(XX + YY - XY - YX)
            loss_x = loss_x + loss

        loss_x = loss_x / K
        return loss_x


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵，即上文中的K
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])# 求矩阵的行数，一般source和target的尺度是一样的，这样便于计算
    total = torch.cat([source, target], dim=0)#将source,target按列方向合并
    #将total复制（n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #将total的每一行都复制成（n+m）行，即每个数据都扩展成（n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #求任意两个数据之间的和，得到的矩阵中坐标（i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0-total1)**2).sum(2)
    #调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    #以fix_sigma为中值，以kernel_mul为倍数取kernel_num个bandwidth值（比如fix_sigma为1时，得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    #高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    #得到最终的核矩阵
    return sum(kernel_val)#/len(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params:
	    source: 源域数据（n * len(x))
	    target: 目标域数据（m * len(y))
	    kernel_mul:
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		loss: MMD loss
    '''
    batch_size = int(source.size()[0])#一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    #根据式（3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss#因为一般都是n==m，所以L矩阵一般不加入计算


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])

        kernels = guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss



class MMD_loss2(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss2, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])

        heatmaps_pred = source
        heatmaps_pred = heatmaps_pred / heatmaps_pred.sum(dim=-1, keepdims=True)
        heatmaps_gt = target
        heatmaps_gt = heatmaps_gt / heatmaps_gt.sum(dim=-1, keepdims=True)

        kernels = guassian_kernel(heatmaps_pred, heatmaps_gt, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss



class JointsKLLoss2(nn.Module):
    """
    KL Divergence for keypoint detection proposed by
    `Regressive Domain Adaptation for Unsupervised Keypoint Detection <https://arxiv.org/abs/2103.06175>`_.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``

    Inputs:
        - output (tensor): heatmap predictions
        - target (tensor): heatmap labels
        - target_weight (tensor): whether the keypoint is visible. All keypoint is visible if None. Default: None.

    Shape:
        - output: :math:`(minibatch, K, H, W)` where K means the number of keypoints,
          H and W is the height and width of the heatmap respectively.
        - target: :math:`(minibatch, K, H, W)`.
        - target_weight: :math:`(minibatch, K)`.
        - Output: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(minibatch, K)`.

    """
    def __init__(self, reduction='mean', epsilon=0.):
        super(JointsKLLoss2, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='none')
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, output, target, target_weight=None):
    
  #      heatmaps_pred = output
  #      heatmaps_pred = F.log_softmax(heatmaps_pred, dim=-1)
 #       heatmaps_gt = target
 #       heatmaps_gt = heatmaps_gt + self.epsilon
 #       heatmaps_gt = heatmaps_gt / heatmaps_gt.sum(dim=-1, keepdims=True))


 #       loss = self.criterion(heatmaps_pred, heatmaps_gt).sum(dim=-1)
 #       if target_weight is not None:
 #           loss = loss * target_weight.view((B, K))
 #       if self.reduction == 'mean':
 #           return loss.mean()
 #       elif self.reduction == 'none':
 #           return loss.mean(dim=-1)

        kl = F.kl_div(output.softmax(dim=-1).log(), target.softmax(dim=-1), reduction='batchmean')

        return kl

