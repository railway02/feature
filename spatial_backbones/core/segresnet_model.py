from __future__ import annotations

from typing import Any

from model_interface import SpatialSegmentationModel


class CorrectedSegResNet(SpatialSegmentationModel):
    family = "segresnet"

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        from monai.networks.nets import SegResNet

        settings = cfg["models"][self.family]
        self.model = SegResNet(
            spatial_dims=int(settings["spatial_dims"]),
            in_channels=int(settings["in_channels"]),
            out_channels=int(settings["out_channels"]),
            init_filters=int(settings["init_filters"]),
            blocks_down=tuple(int(value) for value in settings["blocks_down"]),
            blocks_up=tuple(int(value) for value in settings["blocks_up"]),
            dropout_prob=settings.get("dropout_prob"),
        )

    def encode_and_decode(self, x):
        encoded = self.model.encode(x)
        if isinstance(encoded, tuple):
            feature_map, down = encoded
        else:
            feature_map, down = encoded, []
        logits = self.model.decode(feature_map, list(reversed(down))) if hasattr(self.model, "decode") else self.model(x)
        return feature_map, logits
