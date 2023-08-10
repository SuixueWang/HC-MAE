# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import PIL
import numpy as np
import pickle
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
sys.path.append(os.path.dirname(os.getcwd()))

from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset
from PIL import Image
from matplotlib import pyplot as plt
from survival.util.options import get_args_parser_finetune, logger
args = get_args_parser_finetune().parse_args()



PIL.Image.MAX_IMAGE_PIXELS = 933120000

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

    def __init__(self, data, split, loader = default_loader):

        if split == "all":
            self.X_mrna = np.concatenate((data["train"]['x_mrna'], data["validation"]['x_mrna'], data["test"]['x_mrna']))
            self.X_mirna = np.concatenate((data["train"]['x_mirna'], data["validation"]['x_mirna'], data["test"]['x_mirna']))
            self.X_meth = np.concatenate((data["train"]['x_meth'], data["validation"]['x_meth'], data["test"]['x_meth']))
            self.X_path = np.concatenate((data["train"]['x_path'], data["validation"]['x_path'], data["test"]['x_path']))
            self.censored = np.concatenate((data["train"]['censored'], data["validation"]['censored'], data["test"]['censored']))
            self.survival = np.concatenate((data["train"]['survival'], data["validation"]['survival'], data["test"]['survival']))
            self.region_pixel_5x = np.concatenate((data["train"]['region_pixel_5x'], data["validation"]['region_pixel_5x'], data["test"]['region_pixel_5x']))
            self.region_mae_no_cross = np.concatenate((data["train"]['region_mae_no_cross'], data["validation"]['region_mae_no_cross'], data["test"]['region_mae_no_cross']))
            self.region_mae_cross = np.concatenate((data["train"]['region_mae_cross'], data["validation"]['region_mae_cross'], data["test"]['region_mae_cross']))
            self.region_mae_no_cross = np.concatenate((data["train"]['region_mae_no_cross'], data["validation"]['region_mae_no_cross'], data["test"]['region_mae_no_cross']))
            self.patch_mae = np.concatenate((data["train"]['patch_mae'], data["validation"]['patch_mae'], data["test"]['patch_mae']))
        else:
            self.X_mrna = data[split]['x_mrna']
            self.X_mirna = data[split]['x_mirna']
            self.X_meth = data[split]['x_meth']
            self.X_path = data[split]['x_path']
            self.censored = data[split]['censored']
            self.survival = data[split]['survival']
            self.region_pixel_5x = data[split]['region_pixel_5x']
            self.region_mae_no_cross = data[split]['region_mae_no_cross']
            self.region_mae_cross = data[split]['region_mae_cross']
            self.region_mae_no_cross = data[split]['region_mae_no_cross']
            self.patch_mae = data[split]['patch_mae']

        img_pixels = []
        for i, img_dir in enumerate(self.region_pixel_5x):
            img = np.load(img_dir)
            img_pixels.append(img)
        self.img_pixels = img_pixels

        vecs_reg = []
        for i, vec_dir in enumerate(self.region_mae_no_cross):
            vecs = np.load(vec_dir)
            vecs_reg.append(vecs)
        self.vecs_reg = vecs_reg

        vecs_reg_pat = []
        for i, vec_dir in enumerate(self.region_mae_cross):
            vecs = np.load(vec_dir)
            vecs_reg_pat.append(vecs)
        self.vecs_reg_pat = vecs_reg_pat

        self.loader = loader
        self.max_seq_len = args.max_seq_length
        logger.info(f"max sequence length: {self.max_seq_len}")

    def __getitem__(self, index):

        single_censored = torch.tensor(self.censored[index]).type(torch.LongTensor)
        single_survival = torch.tensor(self.survival[index]).type(torch.FloatTensor)
        single_X_mrna = torch.tensor(self.X_mrna[index]).type(torch.FloatTensor)
        single_X_mirna = torch.tensor(self.X_mirna[index]).type(torch.FloatTensor)
        single_X_meth = torch.tensor(self.X_meth[index]).type(torch.FloatTensor)

        seq_len = len(self.img_pixels[index])

        imgs = self.img_pixels[index]
        imgs = torch.FloatTensor(imgs)

        vecs_pat = []
        for i, vec_dir in enumerate(self.patch_mae[index]):
            vecs = np.load(vec_dir)
            vecs_pat.extend(vecs)
        # vecs_pat = self.vecs_reg[index]
        vecs_pat = torch.FloatTensor(vecs_pat)
        vecs_reg = torch.FloatTensor(self.vecs_reg[index])
        vecs_reg_pat = torch.FloatTensor(self.vecs_reg_pat[index])

        assert seq_len == vecs_reg.shape[0]

        if seq_len >= self.max_seq_len:
            pixel_sum = torch.sum(imgs, dim=[1,2,3])
            top_pixel_id = torch.argsort(pixel_sum, descending=False)[:self.max_seq_len]
            imgs = imgs[top_pixel_id]
            vecs_reg = vecs_reg[top_pixel_id]
            vecs_pat = vecs_pat[top_pixel_id]
            vecs_reg_pat = vecs_reg_pat[top_pixel_id]

        else:
            # zero_pad = torch.zeros((self.max_seq_len-seq_len, 3, 256, 256))
            # imgs = torch.cat((imgs, zero_pad), 0)
            # vecs_reg = torch.nn.functional.pad(vecs_reg, pad=(0, 0, 0, self.max_seq_len - seq_len), mode='constant', value=0)
            pass
        seq_len = torch.tensor(seq_len).type(torch.FloatTensor)

        return imgs, vecs_reg, vecs_pat, vecs_reg_pat, seq_len, single_X_mrna, single_X_mirna, single_X_meth, \
               single_censored, single_survival

    def __len__(self):
        return len(self.X_mrna)




class save_img_npy(Dataset):

    def __init__(self, data, split, loader = default_loader):

        if split == "all":
            self.region_img_5x = np.concatenate(
                (data["train"]['region_img_5x'], data["validation"]['region_img_5x'], data["test"]['region_img_5x']))
            self.region_pixel_5x = np.concatenate(
                (data["train"]['region_pixel_5x'], data["validation"]['region_pixel_5x'], data["test"]['region_pixel_5x']))

        self.data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.2, 1.0)),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.loader = loader

    def save_npy(self):

        for i, region_img_dir in enumerate(self.region_img_5x):
            # if i < 160: continue
            imgs = torch.empty((0, 3, 256, 256))
            for j, img_dir in enumerate(region_img_dir):

                img = self.loader(img_dir)
                img = self.data_transforms(img)
                imgs = torch.cat((imgs, img.unsqueeze(0)), 0)

            folder_npy = "/".join(region_img_dir[0].replace("wsi-crop", "vectors/survival_5x").split("/")[:-2])
            if not os.path.exists(folder_npy):
                os.makedirs(folder_npy)

            np.save(f"{folder_npy}/regions.npy", imgs)
            print(f"i = {i}, j = {j}")


if __name__ == "__main__":

    cancer_type = "STAD"
    data_cv_splits = pickle.load(open(f"../datasets/{cancer_type.lower()}_cv_splits.pkl", 'rb'))

    obj = save_img_npy(
        data=data_cv_splits[0],
        split="all",
    )

    obj.save_npy()

    # npy_dir = "/data/wsxdata/TCGA-LIHC/vectors/survival_5x/TCGA-DD-AAE0-01Z-00-DX1.0F30D65A-A9B0-4098-940E-B9DA702B692F/regions.npy"
    # data = np.load(npy_dir)
    # pass


