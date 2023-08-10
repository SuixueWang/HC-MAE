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
from timm.models.vision_transformer import Block

num_omics = 3


class GlobalAttentionPooling(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GlobalAttentionPooling, self).__init__()
        self.linear_in = nn.Linear(input_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_tensor):
        # 输入张量的形状：[batch_size, sequence_length, input_size]
        hidden = torch.relu(self.linear_in(input_tensor))
        attention_scores = self.linear_out(hidden).squeeze(dim=2)
        attention_weights = self.softmax(attention_scores)
        context_vector = torch.bmm(attention_weights.unsqueeze(dim=1), input_tensor).squeeze(dim=1)
        # 返回上下文向量和注意力权重
        return context_vector, attention_weights


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

        self.encoder_omics = nn.Linear(300, self.embed_dim, bias=True)

        self.cross_attn = CrossAttention(self.embed_dim)
        self.gap = GlobalAttentionPooling(384, 384)

        # self.vit_encoder11 = nn.ModuleList([
        #     Block(self.embed_dim, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)
        #     for i in range(2)])  # 2
        self.norm11 = nn.LayerNorm(self.embed_dim)

        self.vit_encoder12 = nn.ModuleList([
            Block(self.embed_dim, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)
            for i in range(12)])
        self.norm12 = nn.LayerNorm(self.embed_dim)

        self.vit_encoder2 = nn.ModuleList([
            Block(self.embed_dim, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)
            for i in range(12)]) # 12
        self.norm2 = nn.LayerNorm(self.embed_dim)

        self.classifier = nn.Sequential(nn.Linear(self.embed_dim*2, 1), nn.Sigmoid())
        self.output_range = nn.Parameter(torch.FloatTensor([8]), requires_grad=False)
        self.output_shift = nn.Parameter(torch.FloatTensor([-4]), requires_grad=False)

    def forward_features(self, samples):
        imgs, vec_slide, seq_len, X_mrna, X_mirna, X_meth = samples[:]

        X_img = torch.empty((0, imgs.shape[1], self.embed_dim)).cuda()
        for img, s_len in zip(imgs, seq_len):
            vec = self.patch_embed(img)
            X_img = torch.cat((X_img, vec.flatten(1).unsqueeze(0)), 0)

        # X_img = self.pos_drop(X_img)
        # for blk in self.vit_encoder11:
        #     X_img = blk(X_img)
        X_img = self.norm11(X_img)

        # append omics token
        X_mrna = self.encoder_omics(X_mrna).unsqueeze(1)
        X_mirna = self.encoder_omics(X_mirna).unsqueeze(1)
        X_meth = self.encoder_omics(X_meth).unsqueeze(1)
        X_omics = torch.cat((X_mrna, X_mirna, X_meth), dim=1)

        # X_comb = self.cross_attn(query=X_omics, key=vec_slide, value=X_img)
        X_comb = self.cross_attn(query=X_omics, key=X_img, value=vec_slide)

        X_omics = self.pos_drop(X_omics)
        for blk in self.vit_encoder12:
            X_omics = blk(X_omics)
        X_omics = self.norm12(X_omics)

        # X_comb = self.cross_attn(query=X_omics, key=vec_slide, value=X_img)

        X_comb = self.pos_drop(X_comb)
        for blk in self.vit_encoder2:
            X_comb = blk(X_comb)
        X_comb = self.norm2(X_comb)

        # X_omics = X_omics.mean(dim=1)
        # X_comb = X_comb.mean(dim=1)

        X_omics, _ = self.gap(X_omics)
        X_comb, _ = self.gap(X_comb)

        vec_slide = vec_slide.mean(dim=1)
        corr = torch.nn.functional.cosine_similarity(X_omics, vec_slide)

        return torch.cat((X_comb, X_omics), dim=1), corr.unsqueeze(1)

    def forward(self, x):
        x, corr = self.forward_features(x)
        x = self.classifier(x)
        x = x * self.output_range + self.output_shift

        return x



def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        img_size=256, patch_size=256, embed_dim=384, depth=6, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model