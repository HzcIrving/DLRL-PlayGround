#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


"""Package Usage"""
import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

"""Config"""
from Utils import swish,np2th

logger = logging.getLogger(__name__)
ACT2FN = {"gelu": torch.nn.functional.gelu,
          "relu": torch.nn.functional.relu,
          "swish": swish}

ATTENTION_Q = "MultiHeadDotProductAttention_1/query/"
ATTENTION_K = "MultiHeadDotProductAttention_1/key/"
ATTENTION_V = "MultiHeadDotProductAttention_1/value/"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out/"
FC_0 = "MlpBlock_3/Dense_0/"
FC_1 = "MlpBlock_3/Dense_1/"
ATTENTION_NORM = "LayerNorm_0/"
MLP_NORM = "LayerNorm_2/"

# Image Embedding Module
"""
Image2Tokens (Embedding Layer) 
- 16 patches as an example 
- Image: 224 x 224 x 3;  Patch: 16 x 16 
- 图像Patches数量: 224/16 x 224/16 = 196 
- 224 x 224 x 3 -> 14 x 14 x (16 x 16 x 3) => 14 x 14 x 768  
- Conv2d(in_channels:3; out_channels:768; kernel_size:16, stride 16) 即可实现  
>>>  Patch Dividing 
- 每个Patch(768,)  
- 测试代码:
-     input = torch.randn((16,3,224,224))
      layer = Conv2d(in_channels=3,
                   out_channels=768,
                   kernel_size=16,
                   stride=16)
      output = layer(input)
      print(output.shape) # 16, 768, 14, 14 
>>> 【CLS】 Flag adding 
- 第一个单词前面加上[cls] Token,使用这个token最顶端训练出来的向量进行分类 
- [196 x 768] + [1, 768] -> [197 x 768]  
>>> Position Embedding 
- 生成[197 x 768]的位置向量
"""
class ImageEmbeddings(nn.Module):
    def __init__(self,config,img_size,in_channels=3):
        super(ImageEmbeddings, self).__init__()
        img_size = _pair(img_size) # (img_size x img_size)

        # No Hybrid Situation
        self.hybrid = False
        patch_size = _pair(config.patches["size"])
        # patch的数量
        # 256/16 * 256/16
        n_patch = (img_size[0]//patch_size[0]) * (img_size[1]//patch_size[1])

        # Input (Bs,chs,H,W)
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        # Position embedding
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patch+1,config.hidden_size))
        # [CLS] adding
        self.cls_token = nn.Parameter(torch.zeros(1,1,config.hidden_size))

        # Dropout Trick
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self,x):
        # X:
        # Batch_size = 256
        # Hidden_size = 768
        B = x.shape[0]

        # Cls Tokens: Batch_size, 1, hidden_size
        cls_tokens = self.cls_token.expand(B,-1,-1)

        # ResNet Feature Extractor
        if self.hybrid:
            pass

        # Output: (Bs, 768, 14, 14)
        x = self.patch_embeddings(x)
        # Output: (Bs, 768, 196)
        x = x.flatten(2)
        # Output: (Bs, 196, 768)
        x = x.transpose(-1,-2)

        # Output: (Bs, 196 + 1, 768)
        x = torch.cat((cls_tokens, x), dim=1)

        # (Bs, 197, 768)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

"""
VIT Encoder
>>> 输入(Bs, 197, 768) (以16 patches, 224 x 224图像输入为例) 
一个Encoder模块包含四类子模块
- Layer Norm 
- Multi-head Attention 
- MLP Block 
- Dropout    
"""
"""
Sub-Module 1 MLP 
"""
class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

"""
Sub-Module 2 Multi-Head Attention 
"""
class Attention(nn.Module):
    def __init__(self, config, vis=False):
        super(Attention, self).__init__()
        self.vis = vis

        # 头数（12）
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)  # 768 / 12 = 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 12 * 64 = 768

        # Q,K,V
        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self,x):
        # Input Dims of X : (BS, 197, 768)
        # 768 = self.attention_head_size x self.num_attention_heads
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # Output Dims of X : (BS, 197, 12(heads), 64(head_sizes)
        x = x.view(*new_x_shape)

        return x.permute(0,2,1,3) # (Bs,12,197,64)

    def forward(self, hidden_states):
        # Hidden States: (Bs, 197, 768)
        # Output: (Bs, 197, 768)
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        # Multi-heads Dim Transform
        # Output(Bs, heads, 197, hidden_dim/heads)
        #       - (Bs, 12, 197, 64)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Attention_score
        # Dim (16, 12, 197, 197)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs) # Dim (16, 12, 197, 197)

        # 计算Context
        # Output: (16, 12, 197, 64)
        context_layer = torch.matmul(attention_probs, value_layer)
        # Output:(16, 197, 12, 64)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # Output: (16,197,768)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer) # FC
        attention_output = self.proj_dropout(attention_output) # Dropout
        return attention_output, weights

"""
整合上述Attention、MLP,生成Block 
Transformer Encoder Block 
"""
class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self,x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h #Skip Connect (ResNet)

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self,weights,n_block):
        """Pre-trained Model Load """
        ROOT = f"Transformer/encoderblock_{n_block}/"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))

"""
整合Block模块，生成Encoder 
"""
class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

"""
集成上述ImageEmbedding与Encoder模块
Transformer Encoder
"""
class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = ImageEmbeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


"""整体架构 
Transformer Decoder + MLP Head -> Class 
Image Embedding -> Transformer Encoder -> ... 
输入: img_size, num_classes,zero_head(是否MLP Head初始为0),vis(Attention_weight可视化flag)
输出: logits = self.head(x[:,0]) 只需要第一个[CLS]分类Token的值 
"""
class VITransModel(nn.Module):
    def __init__(self,config,img_size=224,num_classes=21843,zero_head=False,vis=False):
        super(VITransModel, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size, vis)
        self.head = Linear(config.hidden_size, num_classes)

    def forward(self,x,labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:,0]) # 只取[cls]对应的logits
        # print(logits.shape)
        # print(labels.shape)
        # print("=====================")
        # print(logits.view(-1,self.num_classes).shape)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1,self.num_classes), labels.view(-1))
            return loss
        else:
            return logits, attn_weights

    def load_from(self,weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                print("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                # Not Hybrid Mode (Vit + Pre-trained Vit)
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname) # Load_from Pre-trained Model

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname) # Load from Pre-trained Model




if __name__ == "__main__":
    img_size = _pair(225)
    print(img_size)

    cls_token = torch.zeros(1,1,768)
    cls_token = cls_token.expand(15,-1,-1)
    print(cls_token.shape)

    x = torch.randn((15,768,14,14))
    x = x.flatten(2)
    x = x.transpose(-1,-2)
    print(x.shape)

    x = torch.cat([cls_token,x],dim=1)
    print(x.shape)

    x1 = torch.randn((15, 197, 768))
    x2 = torch.randn((15, 197, 768))
    print((x1+x2).shape)