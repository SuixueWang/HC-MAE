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
from typing import Iterable, Optional
import torch
from timm.data import Mixup
import util.misc as misc
import util.lr_sched as lr_sched
from model.cox_loss import PartialLogLikelihood, calc_concordance_index, cox_log_rank
from util.options import logger

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 5

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    outputs_accum = None
    censored_accum = None
    survival_accum = None
    k = -1
    for data_iter_step, (imgs, vecs_reg, vecs_pat, vecs_reg_pat, seq_len, X_mrna, X_mirna, X_meth, censored, survival) \
                                          in enumerate(data_loader):

        # we use a per iteration (instead of per epoch) lr scheduler
        if (data_iter_step+1) % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        imgs = imgs.to(device, non_blocking=True)
        vecs_reg = vecs_reg.to(device, non_blocking=True)
        vecs_pat = vecs_pat.to(device, non_blocking=True)
        vecs_reg_pat = vecs_reg_pat.to(device, non_blocking=True)
        seq_len = seq_len.to(device, non_blocking=True)
        X_mrna = X_mrna.to(device, non_blocking=True)
        X_mirna = X_mirna.to(device, non_blocking=True)
        X_meth = X_meth.to(device, non_blocking=True)
        censored = censored.to(device, non_blocking=True)
        survival = survival.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            samples = [imgs, vecs_reg, vecs_pat, vecs_reg_pat, seq_len, X_mrna, X_mirna, X_meth]
            outputs = model(samples)

        k += 1
        if k == 0:
            outputs_accum = outputs
            censored_accum = censored
            survival_accum = survival
        else:
            outputs_accum = torch.cat((outputs_accum, outputs), 0)
            censored_accum = torch.cat((censored_accum, censored), 0)
            survival_accum = torch.cat((survival_accum, survival), 0)

        if k == accum_iter - 1:
            k = -1

            outputs_accum = outputs_accum[torch.argsort(survival_accum, descending=True)]
            censored_accum = censored_accum[torch.argsort(survival_accum, descending=True)]
            survival_accum = survival_accum[torch.argsort(survival_accum, descending=True)]

            loss = PartialLogLikelihood(outputs_accum, censored_accum, survival_accum)
            try:
                c_index = calc_concordance_index(outputs_accum, censored_accum, survival_accum)
            except Exception as e:
                logger.info(e.args)
                continue
            p_value = cox_log_rank(outputs_accum.flatten(0), censored_accum, survival_accum)
            logger.info(f"---------- training c-index: {c_index:.4f}, p-value: {p_value:.30f} ------------------")
            metric_logger.meters['c-index'].update(c_index, n=data_loader.batch_size)
            metric_logger.meters['p-value'].update(p_value, n=data_loader.batch_size)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                logger.info("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss /= accum_iter

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            metric_logger.update(loss=loss_value)
            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)

            loss_value_reduce = misc.all_reduce_mean(loss_value)
            if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
                log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, model


@torch.no_grad()
def evaluate(data_loader, model, device):

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    output_all, censored_all, survival_all = torch.tensor([], device=device), \
                                             torch.tensor([], device=device), \
                                             torch.tensor([], device=device)
    for data_iter_step, (imgs, vecs_reg, vecs_pat, vecs_reg_pat, seq_len, X_mrna, X_mirna, X_meth, censored, survival) in enumerate(metric_logger.log_every(data_loader, 10, header)):

        imgs = imgs.to(device, non_blocking=True)
        vecs_reg = vecs_reg.to(device, non_blocking=True)
        vecs_pat = vecs_pat.to(device, non_blocking=True)
        vecs_reg_pat = vecs_reg_pat.to(device, non_blocking=True)
        seq_len = seq_len.to(device, non_blocking=True)
        X_mrna = X_mrna.to(device, non_blocking=True)
        X_mirna = X_mirna.to(device, non_blocking=True)
        X_meth = X_meth.to(device, non_blocking=True)
        censored = censored.to(device, non_blocking=True)
        survival = survival.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            samples = [imgs, vecs_reg, vecs_pat, vecs_reg_pat, seq_len, X_mrna, X_mirna, X_meth]
            outputs = model(samples)
            # loss = PartialLogLikelihood(output[:,0].unsqueeze(1), censored, survival)

            output_all = torch.cat((output_all, outputs), 0)
            censored_all = torch.cat((censored_all, censored), 0)
            survival_all = torch.cat((survival_all, survival), 0)

    c_index = calc_concordance_index(output_all, censored_all, survival_all)
    p_value = cox_log_rank(output_all.flatten(0), censored_all, survival_all)

    metric_logger.meters['c-index'].update(c_index.item(), n=data_loader.batch_size)
    metric_logger.meters['p-value'].update(p_value.item(), n=data_loader.batch_size)
    # print(f"-------- val --  c-index: {c_index:.4f}, p-value: {p_value:.8f} -----------")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, output_all, censored_all, survival_all