# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
from .swin_transformer3d import SwinTransformer3D
from .swin_mlp import SwinMLP
from .volo import *
from .volo_utils import load_pretrained_weights
import torch
import torch.nn as nn
def build_model():

    model = SwinTransformer(img_size=224,
                                patch_size=4,
                                in_chans=3,
                                num_classes=21841,
                                embed_dim=192, #96,
                                depths=[2,2,18,2],#[2,2,6,2], #[2,2,6,2],
                                num_heads=[6,12,24,48], #[3,6,12,24], #[3,6,12,24], #,
                                window_size=7,
                                mlp_ratio=4.,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.2,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=True)
    checkpoint = torch.load('/data0/liuyang/Video-QA/HRCN/configs/swin_large_patch4_window7_224_22k.pth')                   
    model.load_state_dict(checkpoint['model'], strict=True)
    model.cuda()
    model.eval()
    return model

def build_model3d():

    model = SwinTransformer3D(
                 pretrained= '/data0/liuyang/Video-QA/HRCN/configs/swin_base_patch244_window877_kinetics600_22k.pth',
                 pretrained2d=False,
                 patch_size=(2,4,4),
                 in_chans=3,
                 embed_dim=128, #96,
                 depths=[2,2,18,2], #[2, 2, 6, 2], #[2, 2, 6, 2],
                 num_heads=[4,8,16,32], #[3, 6, 12, 24], #[3, 6, 12, 24], #,
                 window_size=(8,7,7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=True)    
    model.cuda()
    model.eval()            
    return model

def build_volo():
    model = volo_d1()
    load_pretrained_weights(model, "/data0/liuyang/Video-QA/HRCN/configs/volo_d1_224_84.2.pth.tar", use_ema=False, 
                        strict=False, num_classes=1000)  
    return model