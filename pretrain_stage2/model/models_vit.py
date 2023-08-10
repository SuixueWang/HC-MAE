# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial
import torch
import torch.nn as nn
import timm.models.vision_transformer

from pretrain_stage2.util.options import get_args_parser_pretrain
args = get_args_parser_pretrain()
args = args.parse_args()

class CrossAttention(nn.Module):
    def __init__(self, input_dim):
        super(CrossAttention, self).__init__()
        self.query_linear = nn.Linear(input_dim, input_dim)
        self.key_linear = nn.Linear(input_dim, input_dim)
        self.value_linear = nn.Linear(input_dim, input_dim)
        self.scale_factor = 1.0 / (input_dim ** 0.5)

    def forward(self, query, key, value):
        query_proj = self.query_linear(query)  # Query的线性变换
        key_proj = self.key_linear(key)  # Key的线性变换
        value_proj = self.value_linear(value)  # Value的线性变换

        attention_scores = torch.matmul(query_proj, key_proj.transpose(-1, -2))  # 计算注意力分数
        attention_scores = attention_scores * self.scale_factor # 缩放点积
        attention_weights = torch.softmax(attention_scores, dim=-1)  # 计算注意力权重

        attended_values = torch.matmul(attention_weights, value_proj)  # 对Value加权求和

        return attended_values

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.co_attn = CrossAttention(384)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            # self.fc_norm = norm_layer(embed_dim)

            # del self.norm  # remove the original norm

    def forward_features(self, samples):
        x, vectors_cell = samples[:]
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        # cross-attention
        if args.with_cross_attn:
            x = torch.cat((x[:, :1, :], self.co_attn(x[:, 1:, :], vectors_cell, vectors_cell)), dim=1)

        if self.global_pool:
            vectors = x[:, 1:, :].mean(dim=1)  # global pool without cls token

        else:
            vectors = x[:, 0]

        return vectors

    def forward(self, samples):
        vectors = self.forward_features(samples)
        return vectors


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        img_size=4096, patch_size=256, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model