import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class MultiHeadR3D18(nn.Module):
    """Shared R3D-18 backbone with two independent sigmoid heads:
    Violence-present and Weapon-present. Replaces a fragile hard-rule
    fusion cascade (tested and found to degrade accuracy in this project's
    own ablations -- see docs/METHODOLOGY.md)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
        backbone = r3d_18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.violence_head = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)
        )
        self.weapon_head = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        violence_logit = self.violence_head(features).squeeze(-1)
        weapon_logit = self.weapon_head(features).squeeze(-1)
        return violence_logit, weapon_logit
