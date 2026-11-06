from typing import Optional
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision


def conv_bn_relu(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


def make_head(in_ch: int, out_ch: int, mid_ch: int = 64) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(mid_ch, out_ch, kernel_size=1),
    )


class Neck(nn.Module):
    def __init__(self, in_channels: list[int], out_ch: int = 128):
        super().__init__()

        # 1*1 lateral convs for C2, C3, C4, C5
        self.lat = nn.ModuleList(
            [nn.Conv2d(c, out_ch, kernel_size=1) for c in in_channels]
        )

        # smoothing convs after adding skip connections
        self.smooth = nn.ModuleList([conv_bn_relu(out_ch, out_ch) for _ in range(3)])

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        # C5
        x = self.lat[3](feats[3])

        # Top-Down fusion for C4, C3, C2 (indices 2,1,0)
        for i in (2, 1, 0):
            x = F.interpolate(
                x,
                size=feats[i].shape[-2:],
                mode="nearest",
            ) + self.lat[i](feats[i])

            x = self.smooth[i](x)

        return x


CHANNELS = {
    "resnet18": [64, 128, 256, 512],
    "resnet34": [64, 128, 256, 512],
    "resnet50": [256, 512, 1024, 2048],
}


class Detector(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        neck_ch: int = 128,
    ):
        super().__init__()

        # Load torchvision ResNet
        weights = "DEFAULT" if (pretrained and not pretrained_path) else None
        m = getattr(torchvision.models, backbone)(weights=weights)
        if pretrained_path:
            m.load_state_dict(torch.load(pretrained_path, map_location="cpu"))

        # Slice resnet into stem and 4 residual stages
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        self.l1 = m.layer1
        self.l2 = m.layer2
        self.l3 = m.layer3
        self.l4 = m.layer4

        # neck
        self.neck = Neck(CHANNELS[backbone], out_ch=neck_ch)

        # attach 4 prediction heads
        self.hm = make_head(neck_ch, out_ch=1)  # Heatmap: 1 channel (person presence)
        self.wh = make_head(neck_ch, out_ch=2)  # Size: (w, h) in grid cells
        self.off = make_head(neck_ch, out_ch=2)  # Offset: (dx, dy) sub-pixel correction

        # CRITICAL TRICK: CenterNet Bias Initialization
        # Set final conv bias of heatmap head to -2.19
        # Reason: sigmoid(-2.19) ≈ 0.10.
        # Since 99% of grid cells are background (empty), initializing predictions
        # near 0 prevents massive focal loss values at epoch 0!
        final_conv = self.hm[-1]

        if isinstance(final_conv, nn.Conv2d) and final_conv.bias is not None:
            nn.init.constant_(final_conv.bias, -2.19)

    def forward(self, x: torch.Tensor):
        # backbone forward
        x = self.stem(x)
        c2 = self.l1(x)
        c3 = self.l2(c2)
        c4 = self.l3(c3)
        c5 = self.l4(c4)

        f = self.neck([c2, c3, c4, c5])  # (B, 128, 160, 160)

        # 3 Heads forward
        return {
            "hm": self.hm(f),  # (B, 1, 160, 160) - raw logits
            "wh": self.wh(f),  # (B, 2, 160, 160) - size in cells
            "off": self.off(f),  # (B, 2, 160, 160) - sub-pixel offset
        }
