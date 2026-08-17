from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from model_interface import SpatialSegmentationModel


class ImageNetDeepLabV3Plus(SpatialSegmentationModel):
    family = "deeplabv3plus_resnet50_imagenet"

    def __init__(self, cfg: dict[str, Any], load_pretrained: bool = True):
        super().__init__()
        import segmentation_models_pytorch as smp

        settings = cfg["models"][self.family]
        encoder_weights = settings["encoder_weights"] if load_pretrained else None
        self.model = smp.DeepLabV3Plus(
            encoder_name=str(settings["encoder_name"]),
            encoder_weights=encoder_weights,
            in_channels=int(settings["in_channels"]),
            classes=int(settings["classes"]),
            activation=settings.get("activation"),
        )
        self.freeze_batch_norm_running_stats = bool(settings["freeze_batch_norm_running_stats"])
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
        if mode and self.freeze_batch_norm_running_stats:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected [B,1,H,W], received {tuple(x.shape)}")
        rgb = x.repeat(1, 3, 1, 1)
        return (rgb - self.image_mean.to(rgb.dtype)) / self.image_std.to(rgb.dtype)

    def encode_and_decode(self, x: torch.Tensor):
        features = self.model.encoder(self.normalize_input(x))
        decoder_feature_map = self.model.decoder(*features)
        logits = self.model.segmentation_head(decoder_feature_map)
        return decoder_feature_map, logits

    def encoder_conv1(self) -> torch.Tensor:
        return self.model.encoder.conv1.weight
