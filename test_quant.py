import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.utils
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

from config import Config
from models import *

parser = argparse.ArgumentParser(description='FQ-ViT')

parser.add_argument('--model',
                    default='deit_tiny',
                    choices=[
                        'deit_tiny', 'deit_small', 'deit_base', 'vit_base',
                        'vit_large', 'swin_tiny', 'swin_small', 'swin_base'
                    ],
                    help='model')
parser.add_argument('--data', metavar='DIR',
                    default='/media/dinger/inner/Dataset/ImageNet',
                    help='path to dataset')
parser.add_argument('--quant', default=True, action='store_true')
parser.add_argument('--ptf', default=True, action='store_true')
parser.add_argument('--lis', default=True, action='store_true')
parser.add_argument('--quant-method',
                    default='minmax',
                    choices=['minmax', 'ema', 'omse', 'percentile'])
parser.add_argument('--calib-batchsize',
                    default=100,
                    type=int,
                    help='batchsize of calibration set')
parser.add_argument('--calib-iter', default=10, type=int)
parser.add_argument('--val-batchsize',
                    default=100,
                    type=int,
                    help='batchsize of validation set')
parser.add_argument('--num-workers',
                    default=8,
                    type=int,
                    help='number of data loading workers (default: 8)')
parser.add_argument('--device', default='cuda', type=str, help='device')
parser.add_argument('--print-freq',
                    default=100,
                    type=int,
                    help='print frequency')
parser.add_argument('--seed', default=0, type=int, help='seed')


def str2model(name):
    d = {
        'deit_tiny': deit_tiny_patch16_224,
        'deit_small': deit_small_patch16_224,
        'deit_base': deit_base_patch16_224,
        'vit_base': vit_base_patch16_224,
        'vit_large': vit_large_patch16_224,
        'swin_tiny': swin_tiny_patch4_window7_224,
        'swin_small': swin_small_patch4_window7_224,
        'swin_base': swin_base_patch4_window7_224,
    }
    print('Model: %s' % d[name].__name__)
    return d[name]


def seed(seed=0):
    import os
    import random
    import sys

    import numpy as np
    import torch
    sys.setrecursionlimit(100000)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


class my_val_dataset(torch.utils.data.Dataset):
    def __init__(self, valdir, val_transform):
        self.file_path = valdir
        self.transform = val_transform
        self.files = []
        with open(os.path.join(self.file_path, "val_list.txt"), "r") as f:
            for line in f.readlines():
                name, label = line.strip().split()
                self.files.append((name, int(label)))

    def __getitem__(self, index):
        name, label = self.files[index]
        path = os.path.join(self.file_path, name)
        img = Image.open(path)
        img = img.convert("RGB")
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.files)


def main():
    args = parser.parse_args()
    seed(args.seed)

    device = torch.device(args.device)
    cfg = Config(args.ptf, args.lis, args.quant_method)
    model = str2model(args.model)(pretrained=True, cfg=cfg)
    model = model.to(device)

    # Note: Different models have different strategies of data preprocessing.
    model_type = args.model.split('_')[0]
    if model_type == 'deit':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        crop_pct = 0.875
    elif model_type == 'vit':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        crop_pct = 0.9
    elif model_type == 'swin':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        crop_pct = 0.9
    else:
        raise NotImplementedError

    train_transform = build_transform(mean=mean, std=std, crop_pct=crop_pct)
    val_transform = build_transform(mean=mean, std=std, crop_pct=crop_pct)

    # Data
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    val_dataset = my_val_dataset(valdir, val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    # switch to evaluate mode
    model.eval()

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().to(device)

    if args.quant:
        train_dataset = datasets.ImageFolder(traindir, train_transform)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.calib_batchsize,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        # Get calibration set.
        image_list = []
        print(f'Get {len(train_loader)} batches of training data.')
        for i, (data, target) in enumerate(train_loader):
            if i == args.calib_iter:
                break
            data = data.to(device)
            image_list.append(data)

        print(f'Calibrating using {len(image_list)} batches.')
        model.model_open_calibrate()
        with torch.no_grad():
            for i, image in enumerate(image_list):
                if i == len(image_list) - 1:
                    # This is used for OMSE method to
                    # calculate minimum quantization error
                    model.model_open_last_calibrate()
                output = model(image)
        model.model_close_calibrate()
        model.model_quant()

    val_loss, val_prec1, val_prec5 = validate(args, val_loader, model,
                                              criterion, device)
    assert int(val_prec1) >= 70
    assert int(val_prec5) >= 90
    export_model(args.model, model)


def export_model(model_name, model):
    if not os.path.exists('export'):
        os.makedirs('export')
    arrays = {}
    arrays["cls_token"] = model.cls_token.detach().cpu().numpy()
    pos_embed = model.qact_pos(model.pos_embed)
    arrays["pos_embed"] = pos_embed.detach().cpu().numpy()
    with torch.no_grad():
        for name, m in model.named_modules():
            if type(m) in [QConv2d, QLinear]:
                weight = m.weight
                scale = m.quantizer.scale
                zero_point = m.quantizer.zero_point
                bias = m.bias
                # print(name, type(m).__name__, m.bit_type.name, weight.shape, weight.dtype)
                assert m.bit_type.name == 'int8'
                arrays[f"{name}_weight"] = weight.detach().cpu().numpy()
                arrays[f"{name}_scale"] = scale.detach().cpu().numpy()
                arrays[f"{name}_zero_point"] = zero_point.detach().cpu().numpy()
                arrays[f"{name}_bias"] = bias.detach().cpu().numpy()
            if type(m) is QAct:
                scale = m.quantizer.scale
                zero_point = m.quantizer.zero_point
                assert m.bit_type.name == 'int8'
                # print(name, type(m).__name__, m.bit_type.name, scale.shape, zero_point.shape)
                arrays[f"{name}_scale"] = scale.detach().cpu().numpy()
                arrays[f"{name}_zero_point"] = zero_point.detach().cpu().numpy()
            if type(m) is QIntSoftmax:
                # print(name, type(m).__name__, m.bit_type.name)
                assert m.bit_type.name == 'uint4'
            if type(m) is QIntLayerNorm:
                weight = m.weight
                bias = m.bias
                # print(name, type(m).__name__, weight.shape, weight.dtype)
                arrays[f"{name}_weight"] = weight.detach().cpu().numpy()
                arrays[f"{name}_bias"] = bias.detach().cpu().numpy()
    import numpy as np
    np.savez(f"export/{model_name}.npz", **arrays)
    print(f"Save model to {model_name}.npz")


def validate(args, val_loader, model, criterion, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    print(f'Validating on {len(val_loader)} batches.')
    val_start_time = end = time.time()
    for i, (data, target) in enumerate(val_loader):
        data = data.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      top1=top1,
                      top5=top5,
                  ))
    val_end_time = time.time()
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {time:.3f}'.
          format(top1=top1, top5=top5, time=val_end_time - val_start_time))

    return losses.avg, top1.avg, top5.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def build_transform(input_size=224,
                    interpolation='bicubic',
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    crop_pct=0.875):

    def _pil_interp(method):
        if method == 'bicubic':
            return Image.BICUBIC
        elif method == 'lanczos':
            return Image.LANCZOS
        elif method == 'hamming':
            return Image.HAMMING
        else:
            return Image.BILINEAR

    resize_im = input_size > 32
    t = []
    if resize_im:
        size = int(math.floor(input_size / crop_pct))
        ip = _pil_interp(interpolation)
        t.append(
            transforms.Resize(
                size,
                interpolation=ip),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


if __name__ == '__main__':
    main()
