from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabV3ResNet50Binary(nn.Module):
    """COCO-pretrained DeepLabV3 with a one-channel binary output head.

    The penultimate classifier tensor is naturally 256 channels at output
    stride 8 (96x96 for a 768x768 input), so it can later support the same
    spatial pooling interface without fabricating a 256-D projection.
    """

    def __init__(self, pretrained: bool = True, freeze_batch_norm: bool = True):
        super().__init__()
        from torchvision.models.segmentation import (
            DeepLabV3_ResNet50_Weights,
            deeplabv3_resnet50,
        )

        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        weights_backbone = None if pretrained else None
        model = deeplabv3_resnet50(
            weights=weights,
            weights_backbone=weights_backbone,
            aux_loss=True if pretrained else False,
        )
        model.aux_classifier = None
        classifier_layers = list(model.classifier.children())
        in_channels = int(classifier_layers[-1].in_channels)
        self.backbone = model.backbone
        self.feature_head = nn.Sequential(*classifier_layers[:-1])
        self.output_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        nn.init.normal_(self.output_head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.output_head.bias)
        self.freeze_batch_norm = bool(freeze_batch_norm)
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_batch_norm:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def normalize_input(self, x):
        if x.shape[1] != 1:
            raise ValueError(f"Expected grayscale [B,1,H,W], got {tuple(x.shape)}")
        rgb = x.repeat(1, 3, 1, 1)
        return (rgb - self.image_mean.to(rgb.dtype)) / self.image_std.to(rgb.dtype)

    def encode_and_decode(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(self.normalize_input(x))["out"]
        fmap = self.feature_head(features)
        low_resolution_logits = self.output_head(fmap)
        logits = F.interpolate(
            low_resolution_logits,
            size=input_shape,
            mode="bilinear",
            align_corners=False,
        )
        return fmap, logits

    def forward(self, x):
        return self.encode_and_decode(x)[1]


def build_pilot_model(variant: str, cfg, build_segresnet):
    if variant in {"segresnet_geometry", "segresnet_geometry_pos3"}:
        return build_segresnet(cfg), {
            "family": "MONAI SegResNet",
            "pretrained": False,
            "feature_map_contract": [256, 96, 96],
        }
    if variant == "deeplabv3_resnet50_pretrained":
        model = DeepLabV3ResNet50Binary(pretrained=True, freeze_batch_norm=True)
        return model, {
            "family": "torchvision DeepLabV3-ResNet50",
            "pretrained": True,
            "pretraining": "COCO_WITH_VOC_LABELS_V1; binary head newly initialized",
            "batch_norm": "frozen running statistics during fine-tuning",
            "feature_map_contract": [256, 96, 96],
        }
    raise ValueError(variant)
