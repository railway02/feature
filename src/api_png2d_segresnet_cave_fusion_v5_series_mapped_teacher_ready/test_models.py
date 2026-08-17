import torch
from fusion_models import OutcomeModel, SpatialTemporalFusion

def main():
    b = 4
    spatial = torch.randn(b, 1024)
    temporal = torch.randn(b, 10240)

    for mode in ["cave_only","spatial_only","concat","interaction","gated_interaction"]:
        m = OutcomeModel(
            mode=mode,
            spatial_dim=1024,
            temporal_dim=10240,
            hidden_dim=256,
            fusion_mid_dim=512,
            dropout=0.2,
        )
        out = m(
            None if mode == "cave_only" else spatial,
            None if mode == "spatial_only" else temporal,
        )
        assert out["logit"].shape == (b,1)
        assert out["zmain"].shape == (b,256)
        assert torch.isfinite(out["logit"]).all()

        if mode == "gated_interaction":
            assert out["gate_2d"].shape == (b,256)
            assert out["gate_t"].shape == (b,256)
            assert out["hmain"].shape == (b,1024)

    print("V4_TEACHER_ALIGNED_FUSION_SMOKE_OK")

if __name__ == "__main__":
    main()
