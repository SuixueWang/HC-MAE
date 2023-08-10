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
    for data_iter_step, (img, dirs) in enumerate(metric_logger.log_every(data_loader, 100, header)):

        img = img.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            vectors = model(img)

            _dirs = []
            for dir in dirs:
                _dirs.append("/".join(dir.split("/")[:8]))

            assert len(set(_dirs)) == 1, dirs

            dir = dirs[0].split("/")

            vec_dir = os.path.join("/".join(dir[:4]), "vectors/stage1", "/".join(dir[5:6]))
            if not os.path.exists(vec_dir):
                # os.system("rm -rf vec_dir")
                os.makedirs(vec_dir)

            np.save(f"{vec_dir}/{dir[6]}_{dir[7]}.npy", vectors.detach().cpu().numpy())


