import numpy as np
import cv2
import torch
from math import exp, log, sqrt, ceil
from scipy.stats import multivariate_normal



def generate_target(joints, joints_vis, heatmap_size, sigma, image_size):
    """Generate heatamap for joints.

    Args:
        joints: (K, 2)
        joints_vis: (K, 1)
        heatmap_size: W, H
        sigma:
        image_size:

    Returns:

    """
    num_joints = joints.shape[0]
    target_weight = np.ones((num_joints, 1), dtype=np.float32)
    target_weight[:, 0] = joints_vis[:, 0]

    target = np.zeros((num_joints,
                       heatmap_size[1],
                       heatmap_size[0]),
                      dtype=np.float32)

    tmp_size = sigma * 3
    image_size = np.array(image_size)
    heatmap_size = np.array(heatmap_size)

    for joint_id in range(num_joints):
        feat_stride = image_size / heatmap_size
        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        if mu_x >= heatmap_size[0] or mu_y >= heatmap_size[1] \
                or mu_x < 0 or mu_y < 0:
            # If not, just return the image as is
            target_weight[joint_id] = 0
            continue

        # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

        v = target_weight[joint_id]
        if v > 0.5:
            target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return target, target_weight



def keypoint2d_to_3d(keypoint2d: np.ndarray, intrinsic_matrix: np.ndarray, Zc: np.ndarray):
    """Convert 2D keypoints to 3D keypoints"""
    uv1 = np.concatenate([np.copy(keypoint2d), np.ones((keypoint2d.shape[0], 1))], axis=1).T * Zc  # 3 x NUM_KEYPOINTS
    xyz = np.matmul(np.linalg.inv(intrinsic_matrix), uv1).T  # NUM_KEYPOINTS x 3
    return xyz


def keypoint3d_to_2d(keypoint3d: np.ndarray, intrinsic_matrix: np.ndarray):
    """Convert 3D keypoints to 2D keypoints"""
    keypoint2d = np.matmul(intrinsic_matrix, keypoint3d.T).T  # NUM_KEYPOINTS x 3
    keypoint2d = keypoint2d[:, :2] / keypoint2d[:, 2:3]  # NUM_KEYPOINTS x 2
    return keypoint2d


def scale_box(box, image_width, image_height, scale):
    """
    Change `box` to a square box.
    The side with of the square box will be `scale` * max(w, h)
    where w and h is the width and height of the origin box
    """
    left, upper, right, lower = box
    center_x, center_y = (left + right) / 2, (upper + lower) / 2
    w, h = right - left, lower - upper
    side_with = min(round(scale * max(w, h)), min(image_width, image_height))
    left = round(center_x - side_with / 2)
    right = left + side_with - 1
    upper = round(center_y - side_with / 2)
    lower = upper + side_with - 1
    if left < 0:
        left = 0
        right = side_with - 1
    if right >= image_width:
        right = image_width - 1
        left = image_width - side_with
    if upper < 0:
        upper = 0
        lower = side_with -1
    if lower >= image_height:
        lower = image_height - 1
        upper = image_height - side_with
    return left, upper, right, lower


def get_bounding_box(keypoint2d: np.array):
    """Get the bounding box for keypoints"""
    left = np.min(keypoint2d[:, 0])
    right = np.max(keypoint2d[:, 0])
    upper = np.min(keypoint2d[:, 1])
    lower = np.max(keypoint2d[:, 1])
    return left, upper, right, lower


def visualize_heatmap(image, heatmaps, filename):
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR).copy()
    H, W = heatmaps.shape[1], heatmaps.shape[2]
    resized_image = cv2.resize(image, (int(W), int(H)))
    heatmaps = heatmaps.mul(255).clamp(0, 255).byte().cpu().numpy()
    for k in range(heatmaps.shape[0]):
        heatmap = heatmaps[k]
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        masked_image = colored_heatmap * 0.7 + resized_image * 0.3
        cv2.imwrite(filename.format(k), masked_image)
        

def area(left, upper, right, lower):
    return max(right - left + 1, 0) * max(lower - upper + 1, 0)


def intersection(box_a, box_b):
    left_a, upper_a, right_a, lower_a = box_a
    left_b, upper_b, right_b, lower_b = box_b
    return max(left_a, left_b), max(upper_a, upper_b), min(right_a, right_b), min(lower_a, lower_b)


def uvd2xyz( keypoint2d, intrinsic_matrix, Zc):
    """Convert 2D keypoints to 3D keypoints"""
    # uv1 = np.concatenate([np.copy(keypoint2d), np.ones((keypoint2d.shape[0], 1))], axis=1).T * Zc  # 3 x NUM_KEYPOINTS
    # xyz = np.matmul(np.linalg.inv(intrinsic_matrix), uv1).T  # NUM_KEYPOINTS x 3


    uv = keypoint2d    # [B, 21, 2]
    z = Zc             # [B, 21]
    d1 = torch.ones_like(z).unsqueeze(-1)

    uv1 = torch.cat((uv, d1), -1)
    z0 = z.unsqueeze(-1)
    uv2 = uv1.mul(z0)
    uv3 = uv2.permute(0, 2, 1)

    qq = intrinsic_matrix.dtype
    ww = uv3.dtype
   # cam = intrinsic_matrix.double()
    cam = intrinsic_matrix
    xyz = torch.matmul(torch.inverse(cam), uv3).permute(0, 2, 1)

    return xyz



def generate_target2(uv_gts, uv_size=(256, 256), hm_size=(64, 64), std=4):
    heatmap_gt_list = []
    x, y = uv_gts.shape

    xres = 64
    yres = 64
    xlim = (0, xres)
    ylim = (0, yres)

    # x = np.linspace(xlim[0], xlim[1], xres)
    # y = np.linspace(ylim[0], ylim[1], yres)
    x = np.arange(xres, dtype=np.float)
    y = np.arange(yres, dtype=np.float)
    xx, yy = np.meshgrid(x, y)

    # evaluate kernels at grid points
    xxyy = np.c_[xx.ravel(), yy.ravel()]
    # heatmap_vis = np.zeros(hm_size)
     # 21 joints


    for uv_gt in uv_gts:  # 21 joints
        u_max = uv_gt[0] * hm_size[0] / uv_size[0]  # column

        if u_max < 0:
            # print(u_max)
            u_max = torch.tensor(0)

        v_max = uv_gt[1] * hm_size[1] / uv_size[1]  # row

        if v_max < 0:
            #  print(u_max)
            v_max = torch.tensor(0)

        m1 = (u_max, v_max)
        s1 = np.eye(2) * pow(4, 2)
        # k1 = multivariate_normal(mean=m1, cov=593.109206084)
        k1 = multivariate_normal(mean=m1, cov=s1)
        #     zz = k1.pdf(array_like_hm)
        zz = gaussian(xxyy.copy(), m1, 1)
        heatmap_gt = zz.reshape((64, 64))


    # plt.imshow(heatmap_gt)
        # plt.show()

        heatmap_gt_list.append(heatmap_gt.astype(np.float16))
    # plt.imshow(heatmap_vis)
    # plt.savefig("hm" + ".jpg")
    heatmap_gt_list = torch.Tensor(heatmap_gt_list)
    return heatmap_gt_list  # 21 x 64 x 64

def cal_sigma(dmax, edge_value):
    return sqrt(- pow(dmax, 2) / log(edge_value))

def gaussian(array_like_hm, mean, sigma):
    """modifyed version normal distribution pdf, vector version"""
    array_like_hm -= mean
    x_term = array_like_hm[:,0] ** 2
    y_term = array_like_hm[:,1] ** 2
    exp_value = - (x_term + y_term) / 2 / pow(sigma, 2)
    return np.exp(exp_value)

def crop_img(im, bb):
    """
    :param im: H x W x C
    :param bb: x, y, w, h (may exceed the image region)
    :return: cropped image
    """
    crop_im = im[max(0, bb[1]):min(bb[1] + bb[3], im.shape[0]), max(0, bb[0]):min(bb[0] + bb[2], im.shape[1]), :]

    if bb[1] < 0:
        crop_im = cv2.copyMakeBorder(crop_im, -bb[1], 0, 0, 0,  # top, bottom, left, right, bb[3]-crop_im.shape[0]
                                     borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    if bb[1] + bb[3] > im.shape[0]:
        crop_im = cv2.copyMakeBorder(crop_im, 0, bb[1] + bb[3] - im.shape[0], 0, 0,
                                     borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))

    if bb[0] < 0:
        crop_im = cv2.copyMakeBorder(crop_im, 0, 0, -bb[0], 0,  # top, bottom, left, right
                                     borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    if bb[0] + bb[2] > im.shape[1]:
        crop_im = cv2.copyMakeBorder(crop_im, 0, 0, 0, bb[0] + bb[2] - im.shape[1],
                                     borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    return crop_im


def BHWC_to_BCHW(x):
    """
    :param x: torch tensor, B x H x W x C
    :return:  torch tensor, B x C x H x W
    """
    return x.unsqueeze(0).transpose(0, -1).squeeze(-1)
   # return x.transpose(0, -1)


def normalize_image(im):
    """
    byte -> float, / pixel_max, - 0.5
    :param im: torch byte tensor, B x C x H x W, 0 ~ 255
    :return:   torch float tensor, B x C x H x W, -0.5 ~ 0.5
    """
    return ((im.float() / 255.0) - 0.5)


def scale_box2(box, image_width, image_height, scale):
    """
    Change `box` to a square box.
    The side with of the square box will be `scale` * max(w, h)
    where w and h is the width and height of the origin box
    """
    left, upper, right, lower = (box)
    bb = [left, upper, right - left, lower - upper]
    bb = torch.tensor(bb).float()
    bb2 = pad_bounding_rect(bb, 40, [640, 480])
    bbx = bounding_replenish(bb2)
    bbox2 = expand_bounding_rect(bbx, [640, 480], [256, 256])
    left, upper, sw, sh = bbox2
    right = left + sw+1
    lower = upper + sh+1

    if left < 0:
        left = 0
        right = sw
    if right >= image_width:
        right = image_width - 1
        left = image_width - sw
    if upper < 0:
        upper = 0
        lower = sh
    if lower >= image_height:
        lower = image_height - 1
        upper = image_height - sh

    return left.float().item(), upper.float().item(), right.float().item(), lower.float().item()


def pad_bounding_rect(bbox, pad_sz, image_shape):
    """
    :param bbox: 1 x 4, [x, y, w, h]
    :param pad_sz
    :param image_shape: [H, W]
    :return: 1 x 4, [x, y, w, h]
    """
    x_upper = bbox[0] + bbox[2]  # N
    y_upper = bbox[1] + bbox[3]  # N

    xy_pad = torch.clamp(bbox[:2] - pad_sz, min=0.0)


    # w_pad = torch.min(x_upper - xy_pad[:, 0] + pad_sz, image_shape[1] - xy_pad[:, 0])  # N
    # h_pad = torch.min(y_upper - xy_pad[:, 1] + pad_sz, image_shape[0] - xy_pad[:, 1])  # N

    w_pad = torch.min(x_upper - xy_pad[0] + pad_sz, image_shape[0] - xy_pad[0]) # N
    h_pad = torch.min(y_upper - xy_pad[1] + pad_sz, image_shape[1] - xy_pad[1])  # N

    # bbox0 = [xy_pad, w_pad, h_pad]
    # bbox1 = torch.stack(bbox0)
    #w_pad = w_pad.unsqueeze(-1)

    return torch.cat((xy_pad, w_pad.unsqueeze(-1), h_pad.unsqueeze(-1)))  # 1 x 4


def expand_bounding_rect(bbox, image_dim, resize_dim):
    """
    :param bbox: [x, y, w, h]
    :param image_dim: [H, W]
    :param resize_dim: [H_r, W_r]
    :return: N x 4, [x, y, w, h]
    """
    place_ratio = 0.5
    bbox_expand = bbox
    if resize_dim[0] / bbox[3] > resize_dim[1] / bbox[2]:  # keep width
        bbox_expand[3] = resize_dim[0] * bbox[2] / resize_dim[1]
        bbox_expand[1] = max(min(bbox[1] - (bbox_expand[3] - bbox[3]) * place_ratio,
                                 image_dim[1] - bbox_expand[3]), 0.0)
    else:  # keep height
        bbox_expand[2] = resize_dim[1] * bbox[3] / resize_dim[0]
        bbox_expand[0] = max(min(bbox[0] - (bbox_expand[2] - bbox[2]) * place_ratio,
                                 image_dim[0] - bbox_expand[2]), 0.0)

    return bbox_expand.int()

def bounding_replenish(bbx):
    a=bbx[2]
    b=bbx[3]
    s=(a-b)/2
    if a<b:
        t=bbx[0]+s
        bbx[0] = t
    else:
        t=bbx[1]-s
        bbx[1]=t

    return bbx


def uvd2xyz2(P, CamI, bl, root_deep):
        '''
        :param P: Bz,21,3. (u1,v1,d1)
        :param camI: Bz,3,3
        :param bl: Bz,1
        :return: Bz, 21,3. (x0,y0,z0)
        '''
        fx, fy, u0, v0 = CamI[:, 0, 0].unsqueeze(-1), CamI[:, 1, 1].unsqueeze(-1), \
                         CamI[:, 0, 2].unsqueeze(-1), CamI[:, 1, 2].unsqueeze(-1)
        z_cs = P[:, :, -1] * (bl.unsqueeze(-1)) + root_deep

        res = torch.clone(P)

        res[:, :, 0] = z_cs * ((P[:, :, 0] - u0) / fx)
        res[:, :, 1] = z_cs * ((P[:, :, 1] - v0) / fy)
        res[:, :, 2] = z_cs
        return res


def uvd2xyz3(P, CamI, root):
        '''
        :param P: Bz,21,3. (u1,v1,d1)
        :param camI: Bz,3,3
        :param bl: Bz,1
        :return: Bz, 21,3. (x0,y0,z0)
        '''
        fx, fy, u0, v0 = CamI[:, 0, 0].unsqueeze(-1), CamI[:, 1, 1].unsqueeze(-1), \
                         CamI[:, 0, 2].unsqueeze(-1), CamI[:, 1, 2].unsqueeze(-1)
        z_cs = P[:, :, -1] + root

        res = torch.clone(P)

        res[:, :, 0] = z_cs * ((P[:, :, 0] - u0) / fx)
        res[:, :, 1] = z_cs * ((P[:, :, 1] - v0) / fy)
        res[:, :, 2] = z_cs
        return res


def uvd_root(P, CamI, depth):
        '''
        :param P: Bz,21,3. (u1,v1,d1)
        :param camI: Bz,3,3
        :param bl: Bz,1
        :return: Bz, 21,3. (x0,y0,z0)
        '''
        # fx, fy, u0, v0 = CamI[:, 0, 0].unsqueeze(-1), CamI[:, 1, 1].unsqueeze(-1), \
        #                  CamI[:, 0, 2].unsqueeze(-1), CamI[:, 1, 2].unsqueeze(-1)
        # z_cs = P[:, :, -1] * (bl.unsqueeze(-1)) + root_deep
        #
        # res = torch.clone(P)
        #
        # res[:, :, 0] = z_cs * ((P[:, :, 0] - u0) / fx)
        # res[:, :, 1] = z_cs * ((P[:, :, 1] - v0) / fy)
        # res[:, :, 2] = z_cs
        # for i in range(2):
        #     uvd = P[i]
      #  ex = P[:, :, 2]
      #  aa = P[:, 9, :] - P[:, 0, :]
      #  s = torch.norm(aa, p=2, dim=1)
      #  s = s.unsqueeze(-1)
      #  s = s.unsqueeze(1)
    #    P[:, :, 2] = P[:, :, 2] / s
        fx = CamI[:, 0, 0]
        fy = CamI[:, 1, 1]
        u0 = CamI[:, 0, 2]
        v0 = CamI[:, 1, 2]
  
        P0 = P
        Xn = P0[:, 9, :]
        xn = Xn[:, 0]
        yn = Xn[:, 1]
        zn = Xn[:, 2]

        Xm = P0[:, 0, :]
        xm = Xm[:, 0]
        ym = Xm[:, 1]
        zm = Xm[:, 2]

      #  a = (xn - xm)**2 + (yn - ym)**2
      #  b = zn*(xn*xn + yn*yn - xn*xm - yn*ym) + zm*(xm*xm + ym*ym - xn*xm - yn*ym)
      #  c = (xn*zn - xm*zm)**2 + (yn*zn - ym*zm)**2 + (zn - zm)**2 - 1
        
        a = ((xn - xm)/fx)**2 + ((yn - ym)/fy)**2
        b = 2*(((xn-xm)/fx)*(((xn-u0)/fx)*zn - ((xm-u0)/fx)*zm) + ((yn-ym)/fy)*(((yn-v0)/fy)*zn - ((ym-v0)/fy)*zm))
        c = (((xn-u0)/fx)*zn - ((xm-u0)/fx)*zm)**2 + (((yn-v0)/fy)*zn - ((ym-v0)/fy)*zm)**2 + (zn - zm)**2 - 1
     
        xx0 = b*b - 4*a*c
        xx0 = torch.clamp(xx0, min=0.0)

    #    k = torch.tensor([0]).cuda()
    #    if xx0 <= k:
    #        root = 0.5*(-b/a)
    #    if xx0 > k:
        xx = -b + (xx0)**(0.5)
        root = 0.5*(xx/a)
      #  depth = depth + root
        P[:, :, 2] = P[:, :, 2] + root.unsqueeze(-1)




        return P


def P2W(P, CamI, depth):
        '''
        :param P: Bz,21,3. (u1,v1,d1)
        :param camI: Bz,3,3
        :param bl: Bz,1
        :return: Bz, 21,3. (x0,y0,z0)
        '''
        # fx, fy, u0, v0 = CamI[:, 0, 0].unsqueeze(-1), CamI[:, 1, 1].unsqueeze(-1), \
        #                  CamI[:, 0, 2].unsqueeze(-1), CamI[:, 1, 2].unsqueeze(-1)
        # z_cs = P[:, :, -1] * (bl.unsqueeze(-1)) + root_deep
        #
        # res = torch.clone(P)
        #
        # res[:, :, 0] = z_cs * ((P[:, :, 0] - u0) / fx)
        # res[:, :, 1] = z_cs * ((P[:, :, 1] - v0) / fy)
        # res[:, :, 2] = z_cs
        # for i in range(2):
        #     uvd = P[i]
      #  ex = P[:, :, 2]
      #  aa = P[:, 9, :] - P[:, 0, :]
      #  s = torch.norm(aa, p=2, dim=1)
      #  s = s.unsqueeze(-1)
      #  s = s.unsqueeze(1)
    #    P[:, :, 2] = P[:, :, 2] / s
        fx = CamI[:, 0, 0] 
        fy = CamI[:, 1, 1] 
        u0 = CamI[:, 0, 2] 
        v0 = CamI[:, 1, 2] 

        


        P0 = P 
        Xn = P0[:, 9, :]
        xn = Xn[:, 0]
        yn = Xn[:, 1]
        zn = Xn[:, 2]

        Xm = P0[:, 0, :]
        xm = Xm[:, 0]
        ym = Xm[:, 1]
        zm = Xm[:, 2]

     #   a = (xn - xm)**2 + (yn - ym)**2
     #   b = zn*(xn*xn + yn*yn - xn*xm - yn*ym) + zm*(xm*xm + ym*ym - xn*xm - yn*ym)
     #   c = (xn*zn - xm*zm)**2 + (yn*zn - ym*zm)**2 + (zn - zm)**2 - 1
        a = ((xn - xm)/fx)**2 + ((yn - ym)/fy)**2
        b = 2*(((xn-xm)/fx)*(((xn-u0)/fx)*zn - ((xm-u0)/fx)*zm) + ((yn-ym)/fy)*(((yn-v0)/fy)*zn - ((ym-v0)/fy)*zm))
        c = (((xn-u0)/fx)*zn - ((xm-u0)/fx)*zm)**2 + (((yn-v0)/fy)*zn - ((ym-v0)/fy)*zm)**2 + (zn -zm)**2 - 1

        xx0 = b*b - 4*a*c
        xx0 = torch.clamp(xx0, min=0.0)
        

    #    b = torch.where(b > 0, 0.5*b, -0.5*b)
        xx = -b + (xx0)**(0.5)
        root = (0.5*(xx/a)) * 1
        depth2 = depth + root.unsqueeze(-1) 
       # P[:, :, 2] = P[:, :, 2] + root.unsqueeze(-1)




        return root



def uvd2xyz4(P, CamI, bl):
        '''
        :param P: Bz,21,3. (u1,v1,d1)
        :param camI: Bz,3,3
        :param bl: Bz,1
        :return: Bz, 21,3. (x0,y0,z0)
        '''
        fx, fy, u0, v0 = CamI[:, 0, 0].unsqueeze(-1), CamI[:, 1, 1].unsqueeze(-1), \
                         CamI[:, 0, 2].unsqueeze(-1), CamI[:, 1, 2].unsqueeze(-1)
        z_cs = P[:, :, -1] * (bl.unsqueeze(-1))

        res = torch.clone(P)

        res[:, :, 0] = z_cs * ((P[:, :, 0] - u0) / fx)
        res[:, :, 1] = z_cs * ((P[:, :, 1] - v0) / fy)
        res[:, :, 2] = z_cs
        return res
