# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Tianxiao Zhang
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
from .vit import ViT
from .cait import CaiT


def build_model(config):
    model_type = config.MODEL.TYPE

    # accelerate layernorm
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm

    if model_type == 'swin':
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                norm_layer=layernorm,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                fused_window_process=config.FUSED_WINDOW_PROCESS)
    elif model_type == 'vit':
        model = ViT(image_size=config.DATA.IMG_SIZE,
                    patch_size=config.MODEL.ViT.PATCH_SIZE,
                    num_classes=config.MODEL.NUM_CLASSES,
                    dim=config.MODEL.ViT.DIM,
                    depth=config.MODEL.ViT.DEPTHS,
                    heads=config.MODEL.ViT.NUM_HEADS,
                    mlp_dim=config.MODEL.ViT.MLP_DIM,
                    dim_head=config.MODEL.ViT.DIM_HEAD,
                    dropout=config.MODEL.DROP_RATE,
                    emb_dropout=config.MODEL.DROP_RATE)
    elif model_type == 'vit_s':
        model = ViT(image_size=config.DATA.IMG_SIZE,
                    patch_size=config.MODEL.ViT_S.PATCH_SIZE,
                    num_classes=config.MODEL.NUM_CLASSES,
                    dim=config.MODEL.ViT_S.DIM,
                    depth=config.MODEL.ViT_S.DEPTHS,
                    heads=config.MODEL.ViT_S.NUM_HEADS,
                    mlp_dim=config.MODEL.ViT_S.MLP_DIM,
                    dim_head=config.MODEL.ViT_S.DIM_HEAD,
                    dropout=config.MODEL.DROP_RATE,
                    emb_dropout=config.MODEL.DROP_RATE)
    elif model_type == 'cait_xxs':
        model = CaiT(image_size=config.DATA.IMG_SIZE,
                     patch_size=config.MODEL.CaiT_XXS.PATCH_SIZE,
                     num_classes=config.MODEL.NUM_CLASSES,
                     dim=config.MODEL.CaiT_XXS.DIM,
                     depth=config.MODEL.CaiT_XXS.DEPTHS,
                     cls_depth=config.MODEL.CaiT_XXS.CLS_DEPTHS,
                     heads=config.MODEL.CaiT_XXS.NUM_HEADS,
                     mlp_dim=config.MODEL.CaiT_XXS.MLP_DIM,
                     dim_head=config.MODEL.CaiT_XXS.DIM_HEAD,
                     dropout=config.MODEL.DROP_RATE,
                     emb_dropout=config.MODEL.DROP_RATE,
                     layer_dropout=config.MODEL.DROP_RATE)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
