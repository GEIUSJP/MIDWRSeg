import torch
import torch.nn as nn
from model.nn.criss_cross_attention import CrissCrossAttention

__all__ = ['RCCAModule']


class RCCAModule(nn.Module):
    def __init__(self, recurrence=2, in_channels=2048, num_classes=33):
        super(RCCAModule, self).__init__()
        self.recurrence = recurrence
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.inter_channels = in_channels // 4
        self.conv_in = nn.Sequential(
            nn.Conv2d(self.in_channels, self.inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.inter_channels)
        )
        self.CCA = CrissCrossAttention(self.inter_channels)
        self.conv_out = nn.Sequential(
            nn.Conv2d(self.inter_channels, self.inter_channels, 3, padding=1, bias=False)
        )
        self.cls_seg = nn.Sequential(
            nn.Conv2d(self.in_channels + self.inter_channels, self.inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.inter_channels),
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True),
            nn.Conv2d(self.inter_channels, self.num_classes, 1)
        )

    def forward(self, x):
        # reduce channels from C to C'   2048->512
        output = self.conv_in(x)

        for i in range(self.recurrence):
            output = self.CCA(output)

        output = self.conv_out(output)
        output = self.cls_seg(torch.cat([x, output], 1))
        return output


if __name__ == "__main__":
    model = RCCAModule(in_channels=2048)
    x = torch.randn(2, 2048, 28, 28)
    model.cuda()
    out = model(x.cuda())
    print(out.shape)
