# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
import numpy as np

from torchvision import datasets, transforms
import torch
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset
from PIL import Image

PIL.Image.MAX_IMAGE_PIXELS = 933120000

import sys, os
# 根据实际项目目录结构，将运行目录加入到环境变量中
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 注意，此处不要使用append
sys.path.insert(0, BASE_DIR)


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

# use PIL Image to read image
def default_loader(path: str):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class HMAEDataset(Dataset):

    def __init__(self, data_path, loader=default_loader):

        cancer_types = [
            "BRCA",
            "COAD",
            "LGG",
            "LIHC",
            "LUAD",
            "STAD"
        ]

        self.image_dirs = []
        for cancer_type in cancer_types:
            for i, line in enumerate(open(f"{data_path}/patches_order_{cancer_type}.txt", "r", encoding="utf8")):
                self.image_dirs.append(line.strip())
                # if i > 100000:
                #     break


        self.data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.2, 1.0)),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.loader = loader

    def __getitem__(self, index):

        img = self.loader(self.image_dirs[index])
        img = self.data_transforms(img)

        dir = self.image_dirs[index]

        return img, dir

    def __len__(self):
        return len(self.image_dirs)



