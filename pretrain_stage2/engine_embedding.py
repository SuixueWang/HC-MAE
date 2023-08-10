# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import math
import sys
import torch
import util.misc as misc
from util.options import logger
import numpy as np
import os


@torch.no_grad()
def embedding(data_loader, model, device):

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    dirs_all, vectors_all = [], []
    for data_iter_step, (img, dirs, vectors_cell) in enumerate(metric_logger.log_every(data_loader, 10, header)):

        img = img.to(device, non_blocking=True)
        vectors_cell = vectors_cell.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            samples = [img, vectors_cell]
            vectors = model(samples)

            vectors = list(vectors.detach().cpu().numpy())
            dirs_all.extend(dirs)
            vectors_all.extend(vectors)

    slide_vecs = {}
    for _dir, _vec in zip(dirs_all, vectors_all):
        slide_name = _dir.split("/")[5]
        vec_dir = os.path.join("/".join(_dir.split("/")[:4]), f"vectors/stage2-cross/{slide_name}")
        if vec_dir not in slide_vecs:
            slide_vecs[vec_dir] = [_vec]
        else:
            temp = slide_vecs[vec_dir]
            temp.append(_vec)
            slide_vecs[vec_dir] = temp

    for _vec_dir, _vec in slide_vecs.items():

        if not os.path.exists(_vec_dir):
            os.makedirs(_vec_dir)
        np.save(f"{_vec_dir}/regions.npy", _vec)




