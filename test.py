import random
import time
import warnings
import sys
import argparse
import shutil

import torch
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToPILImage
import torch.nn.functional as F

#sys.path.append('../../..')
from marsda.model.regda_4 import PoseResNet as RegDAPoseResNet, \
    PseudoLabelGenerator, RegressionDisparity4, PoseResNet3 as RegDAPoseResNet3, PoseResNet2 as RegDAPoseResNet2, RegressionDisparity3
import marsda.model as models
from marsda.model.pose_resnet2 import Upsampling, PoseResNet
from marsda.model.loss import JointsKLLoss, update_ema_variables5, loss3
import marsda.dataset as datasets
import marsda.dataset.keypoint_detection as T
from utils import Denormalize
from utils.data import ForeverDataIterator
from utils.meter import AverageMeter, ProgressMeter, AverageMeterDict
from utils.keypoint_detection import accuracy, compute_uv_from_heatmaps
from utils.logger import CompleteLogger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

global_step = 0
def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)
    global global_step

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = T.Compose([
        T.RandomRotation(args.rotation),
        T.RandomResizedCrop(size=args.image_size, scale=args.resize_scale),
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
        T.GaussianBlur(),
        T.ToTensor(),
        normalize
    ])
    
    
    val_transform = T.Compose([
        T.Resize(args.image_size),
        T.ToTensor(),
        normalize
    ])
    image_size = (args.image_size, args.image_size)
    heatmap_size = (args.heatmap_size, args.heatmap_size)
    source_dataset = datasets.__dict__[args.source]
    train_source_dataset = source_dataset(root=args.source_root, transforms=train_transform,
                                          image_size=image_size, heatmap_size=heatmap_size)
    train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_source_dataset = source_dataset(root=args.source_root, split='test', transforms=val_transform,
                                        image_size=image_size, heatmap_size=heatmap_size)
    val_source_loader = DataLoader(val_source_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    target_dataset = datasets.__dict__[args.target]
    train_target_dataset = target_dataset(root=args.target_root, transforms=train_transform,
                                          image_size=image_size, heatmap_size=heatmap_size)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
                                                                   
                                     
    val_target_dataset = target_dataset(root=args.target_root, split='test', transforms=val_transform,
                                        image_size=image_size, heatmap_size=heatmap_size)
    val_target_loader = DataLoader(val_target_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    print("Source train:", len(train_source_loader))
    print("Target train:", len(train_target_loader))
    print("Source test:", len(val_source_loader))
    print("Target test:", len(val_target_loader))

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    def creat_ema(model):

        backbone = models.__dict__[args.arch](pretrained=True)
   #     backbone2 = models.__dict__[args.arch2]()
        upsampling = Upsampling(backbone.out_features)
        num_keypoints = train_source_dataset.num_keypoints
        model_ema = RegDAPoseResNet2(backbone, upsampling, 256, num_keypoints, num_head_layers=args.num_head_layers, finetune=True).to(device)
        # pretrained_dict = torch.load(args.pretrain01, map_location='cpu')['model']
        # model.load_state_dict(pretrained_dict, strict=False)
      
        #for param in model.parameters():
        #    param.detach_()

        for param_main, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.data.copy_(param_main.data)  # initialize
                param_ema.requires_grad = False  # not update by gradient

        return model_ema



    backbone = models.__dict__[args.arch](pretrained=True)
    upsampling = Upsampling(backbone.out_features)
    num_keypoints = train_source_dataset.num_keypoints
    model = RegDAPoseResNet3(backbone, upsampling, 256, num_keypoints, num_head_layers=args.num_head_layers, finetune=True).to(device)
    model_ema = creat_ema(model)

    # define loss function
    criterion = JointsKLLoss()
    pseudo_label_generator = PseudoLabelGenerator(num_keypoints, args.heatmap_size, args.heatmap_size)
    regression_disparity = RegressionDisparity4(pseudo_label_generator, JointsKLLoss(epsilon=1e-7))
    regression_disparity2 = RegressionDisparity3(pseudo_label_generator, JointsKLLoss(epsilon=1e-7))



   
    # optionally resume from a checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
 
   

    # define visualization function
    tensor_to_image = Compose([
        Denormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToPILImage()
    ])

    def visualize(image, keypoint2d, name, heatmaps=None):
        """
        Args:
            image (tensor): image in shape 3 x H x W
            keypoint2d (tensor): keypoints in shape K x 2
            name: name of the saving image
        """
        train_source_dataset.visualize(tensor_to_image(image),
                                       keypoint2d, logger.get_image_path("{}.jpg".format(name)))


    # start test

    print("test step.")

    target_val_acc = validate(val_target_loader, model, criterion, visualize if args.debug else None, args)

    print("Target: {:4.3f}".format(target_val_acc['all']))

    for name, acc in target_val_acc.items():
         print("{}: {:4.3f}".format(name, acc))

    logger.close()


def validate(val_loader, model, criterion, visualize, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.2e')
    acc = AverageMeterDict(val_loader.dataset.keypoints_group.keys(), ":3.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, acc['all']],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (x, label, weight, meta) in enumerate(val_loader):
            x = x.to(device)
            label = label.to(device)
            weight = weight.to(device)

            # compute output
            y = model(x)
            loss = criterion(y, label, weight)

            # measure accuracy and record loss
            losses.update(loss.item(), x.size(0))
            acc_per_points, avg_acc, cnt, pred = accuracy(y.cpu().numpy(),
                                                          label.cpu().numpy())

            group_acc = val_loader.dataset.group_accuracy(acc_per_points)
            acc.update(group_acc, x.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
                if visualize is not None:
                    visualize(x[0], pred[0] * args.image_size / args.heatmap_size, "val_{}_pred.jpg".format(i))
                    visualize(x[0], meta['keypoint2d'][0], "val_{}_label.jpg".format(i))

    return acc.average()



def update_ema_variables(model, ema_model, alpha, global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            # ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='Source Only for Keypoint Detection Domain Adaptation')
    # dataset parameters
    parser.add_argument('--source_root', default='data/RHD', help='root path of the source dataset')
    parser.add_argument('target_root', help='root path of the target dataset')
    parser.add_argument('-s', '--source', default='RenderedHandPose', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('--resize-scale', nargs='+', type=float, default=(0.6, 1.3),
                        help='scale range for the RandomResizeCrop augmentation')
    parser.add_argument('--rotation', type=int, default=180,
                        help='rotation range of the RandomRotation augmentation')
    parser.add_argument('--image-size', type=int, default=256,
                        help='input image size')
    parser.add_argument('--heatmap-size', type=int, default=64,
                        help='output heatmap size')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet101',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet101)')
    parser.add_argument('-a2', '--arch2', metavar='ARCH', default='net_hg',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet101)')
    parser.add_argument("--pretrain", type=str, default='models/pretrain_rhd.pth',
                        help="Where restore pretrained model parameters from.")

    parser.add_argument("--ema_model", type=str, default=None,
                        help="Where restore pretrained model parameters from.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="where restore model parameters from.")
    parser.add_argument('--num-head-layers', type=int, default=2)
    parser.add_argument('--margin', type=float, default=4., help="margin gamma")
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.0001, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-gamma', default=0.0001, type=float)
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-step', default=[45, 60], type=tuple, help='parameter for lr scheduler')
    parser.add_argument('--lr-factor', default=0.1, type=float, help='parameter for lr scheduler')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--pretrain_epochs', default=70, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=500, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument("--log", type=str, default='logs/mt',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test'],
                        help="When phase is 'test', only test the model.")
    parser.add_argument('--debug', action="store_true",
                        help='In the debug mode, save images and predictions')
    parser.add_argument('--ema-decay', default=0.999, type=float, metavar='ALPHA',
                        help='ema variable decay rate (default: 0.999)')
    args = parser.parse_args()
    main(args)



