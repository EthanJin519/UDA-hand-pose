import torch
import os
import pickle
import numpy.linalg as LA
from utils._util import download as download_data, check_exits
from marsda.dataset.keypoint_detection import *
from .keypoint_dataset import Hand21KeypointDataset
from .util import *
import scipy.io as sio
import os.path as osp
from torchvision import transforms


SK_fx_color = 607.92271
SK_fy_color = 607.88192
SK_tx_color = 314.78337
SK_ty_color = 236.42484

def SK_rot_mx(rot_vec):
    """
    use Rodrigues' rotation formula to transform the rotation vector into rotation matrix
    :param rot_vec:
    :return:
    """
    theta = LA.norm(rot_vec)
    vector = np.array(rot_vec) * math.sin(theta / 2.0) / theta
    a = math.cos(theta / 2.0)
    b = -vector[0]
    c = -vector[1]
    d = -vector[2]
    return np.array([[a * a + b * b - c * c - d * d, 2 * (b * c + a * d), 2 * (b * d - a * c)],
                     [2 * (b * c - a * d), a * a + c * c - b * b - d * d, 2 * (c * d + a * b)],
                     [2 * (b * d + a * c), 2 * (c * d - a * b), a * a + d * d - b * b - c * c]])


SK_rot_vec = [0.00531, -0.01196, 0.00301]
SK_trans_vec = [-24.0381, -0.4563, -1.2326]  # mm
SK_rot = SK_rot_mx(SK_rot_vec)




intrinsic_matrix0 = np.asarray([
  [SK_fx_color, 0, SK_tx_color],
  [0, SK_fy_color, SK_ty_color],
  [0, 0, 1]])


class STBx1(Hand21KeypointDataset):
    """`Hand-3d-Studio Dataset <https://www.yangangwang.com/papers/ZHAO-H3S-2020-02.html>`_

    Args:
        root (str): Root directory of dataset
        split (str, optional): The dataset split, supports ``train``, ``test``, or ``all``.
        task (str, optional): The task to create dataset. Choices include ``'noobject'``: only hands without objects, \
            ``'object'``: only hands interacting with hands, and ``'all'``: all hands. Default: 'noobject'.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transforms (callable, optional): A function/transform that takes in a dict (which contains PIL image and
            its labels) and returns a transformed version. E.g, :class:`~common.vision.transforms.keypoint_detection.Resize`.
        image_size (tuple): (width, height) of the image. Default: (256, 256)
        heatmap_size (tuple): (width, height) of the heatmap. Default: (64, 64)
        sigma (int): sigma parameter when generate the heatmap. Default: 2

    .. note::
        We found that the original H3D image is in high resolution while most part in an image is background,
        thus we crop the image and keep only the surrounding area of hands (1.5x bigger than hands) to speed up training.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            H3D_crop/
                annotation.json
                part1/
                part2/
                part3/
                part4/
                part5/
    """
    def __init__(self, root, split='train', task='noobject', download=True, **kwargs):

        # if download:
        #     download_data(root, "H3D_crop", "H3D_crop.tar", "https://cloud.tsinghua.edu.cn/f/d4e612e44dc04d8eb01f/?dl=1")
        # else:
        #     check_exits(root, "H3D_crop")

        root = os.path.join(root, "STB")
        # load labels
        assert split in ['train', 'test', 'all']
        self.split = split

        image_list = ["B1Counting", "B1Random", "B2Counting", "B2Random", "B3Counting", "B3Random", "B4Counting",
                      "B4Random", "B5Counting", "B5Random", "B6Counting", "B6Random"]
        if split == 'train':
            image_list = image_list[2:]
            samples = self.get_samples(root, image_list)
        if split == 'test':
            image_list = image_list[:2]
            samples = self.get_samples(root, image_list)

        example = samples[-1]

        super(STBx1, self).__init__(root, samples, **kwargs)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_name = sample['name']
        image_path = os.path.join(self.root, image_name)
        image = Image.open(image_path)
        keypoint3d_camera = np.array(sample['keypoint3d'])  # NUM_KEYPOINTS x 3
        keypoint2d = np.array(sample['keypoint2d'])  # NUM_KEYPOINTS x 2
        keypoint2d2 = np.array(sample['keypoint2d2'])  # NUM_KEYPOINTS x 2
        intrinsic_matrix = np.array(sample['intrinsic_matrix'])
        Zc = keypoint3d_camera[:, 2]
    

        # Crop the images such that the hand is at the center of the image
        # The images will be 1.5 times larger than the hand
        # The crop process will change Xc and Yc, leaving Zc with no changes
        bounding_box = get_bounding_box(keypoint2d2)
        w, h = image.size

        left, upper, right, lower = scale_box(bounding_box, w, h, 1.6)
        image, keypoint2d = crop(image, upper, left, lower - upper, right - left, keypoint2d)
   #     w1, h1 = image.size
   #     resize1 = transforms.Resize([256,256])
   #     image = resize1(image)
   #     factor = float(255) / float(w1)
   #     keypoint2d = factor * keypoint2d
        
      #  bounding_box = get_bounding_box(keypoint2d)
       # left, upper, right, lower = scale_box(bounding_box, 512, 512, 1.5)
      #  image, keypoint2d = crop(image, upper, left, lower - upper, right - left, keypoint2d)
        
        

        image, data = self.transforms(image, keypoint2d=keypoint2d, intrinsic_matrix=intrinsic_matrix)
        keypoint2d = data['keypoint2d']
        intrinsic_matrix = data['intrinsic_matrix']
        if 'image_ema' in data:
             image_ema = data['image_ema']
        else:
             image_ema = image
        keypoint3d_camera = keypoint2d_to_3d(keypoint2d, intrinsic_matrix, Zc)
        zc = keypoint3d_camera[:, 2]

        # noramlize 2D pose:
        visible = np.ones((self.num_keypoints,), dtype=np.float32)
        visible = visible[:, np.newaxis]
        # 2D heatmap
        target, target_weight = generate_target(keypoint2d, visible, self.heatmap_size, self.sigma, self.image_size)
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        # normalize 3D pose:
        # put middle finger metacarpophalangeal (MCP) joint in the center of the coordinate system
        # and make distance between wrist and middle finger MCP joint to be of length 1
        keypoint3d_n = keypoint3d_camera - keypoint3d_camera[9:10, :]
        keypoint3d_n = keypoint3d_n / np.sqrt(np.sum(keypoint3d_n[0, :] ** 2))

        meta = {
            'image': image_name,
            'keypoint2d': keypoint2d,  # （NUM_KEYPOINTS x 2）
            'keypoint3d': keypoint3d_n,  # （NUM_KEYPOINTS x 3）
            'z': zc,
            'keypoint3d_camera': keypoint3d_camera,
            'cam_param': intrinsic_matrix,
            'image_ema': image_ema,
        }

        return image, target, target_weight, meta

    def get_samples(self, root, image_list):

        # load annotations of this set
        ann_dir = os.path.join(root, "labels")
        image_prefix = "SK_color"
        image_dir_list = [os.path.join(root, image_dir) for image_dir in image_list]
        ann_file_list = [os.path.join(ann_dir, image_dir + "_" + image_prefix[:2] + ".mat")
                           for image_dir in image_list]
      #  ann_file_list = [e for e in ann_file_list0]
        samples = []
        hand_index = [0, 17, 18, 19, 20, 13, 14, 15, 16, 9, 10, 11, 12, 5, 6, 7, 8, 1, 2, 3, 4]
        for image_dir, ann_file, image in zip(image_dir_list, ann_file_list, image_list):
            mat_gt = sio.loadmat(ann_file)
            curr_pose_gts = mat_gt["handPara"].transpose((2, 1, 0))
            curr_pose_gts = self.SK_xyz_depth2color(curr_pose_gts, SK_trans_vec, SK_rot)
            curr_pose_gts = curr_pose_gts[:, hand_index, :] / 10.0
            curr_pose_gts1 = self.palm2wrist(curr_pose_gts.copy())
            curr_pose_gts0 = self.palm2wrist0(curr_pose_gts.copy())


            for image_id in range(curr_pose_gts1.shape[0]):
                image_name = osp.join(image, "%s_%d.png" % (image_prefix, image_id))
                keypoint3d = curr_pose_gts1[image_id]
                keypoint3d2 = curr_pose_gts0[image_id]
                keypoint2d = keypoint3d_to_2d(keypoint3d, intrinsic_matrix0)
                keypoint2d2 = keypoint3d_to_2d(keypoint3d2, intrinsic_matrix0)

                sample = {
                    'name': image_name,
                    'keypoint2d': keypoint2d,
                    'keypoint2d2': keypoint2d2,
                    'keypoint3d': keypoint3d,
                    'intrinsic_matrix': intrinsic_matrix0
                }
                samples.append(sample)

        return samples
    def palm2wrist(self, pose_xyz):
        root_id = 0
        mid_root_id = 9
        pose_xyz[:, root_id, :] = pose_xyz[:, mid_root_id, :] + \
                                  2.1 * (pose_xyz[:, root_id, :] - pose_xyz[:, mid_root_id, :])  # N x K x 3
        return pose_xyz

    def palm2wrist0(self, pose_xyz):
        root_id = 0
        mid_root_id = 13
        pose_xyz[:, root_id, :] = pose_xyz[:, mid_root_id, :] + \
                                  2.3 * (pose_xyz[:, root_id, :] - pose_xyz[:, mid_root_id, :])  # N x K x 3
        return pose_xyz
        
    def palm2wrist3(self, pose_xyz):
        root_id = 0
        mid_root_id = 9
        pose_xyz[:, root_id, :] = pose_xyz[:, root_id, :] + \
                                  1.1 * (pose_xyz[:, root_id, :] - pose_xyz[:, mid_root_id, :])  # N x K x 3
        return pose_xyz

    def SK_xyz_depth2color(self, depth_xyz, trans_vec, rot_mx):
        """
        :param depth_xyz: N x 21 x 3, trans_vec: 3, rot_mx: 3 x 3
        :return: color_xyz: N x 21 x 3
        """
        color_xyz = depth_xyz - np.tile(trans_vec, [depth_xyz.shape[0], depth_xyz.shape[1], 1])
        return color_xyz.dot(rot_mx)
