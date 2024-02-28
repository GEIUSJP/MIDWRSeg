import torch
import torch.nn as nn
import torch._utils
from torchstat import stat
from model.mscan import MSCAN




class Stem(nn.Module):
    def __init__(self, out_planes):
        super(Stem, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, out_channels=out_planes // 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=out_planes // 2)
        )
        self.left1 = nn.Sequential(
            nn.Conv2d(out_planes // 2, out_planes // 4, kernel_size=1),
            nn.GroupNorm(num_groups=16, num_channels=out_planes // 4),
            nn.ELU(inplace=True)
        )
        self.left2 = nn.Conv2d(out_planes // 4, out_planes // 2, kernel_size=3, stride=2, padding=1)
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=out_planes),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        feat = self.conv1(x)
        left = self.left2(self.left1(feat))
        right = self.max_pooling(feat)
        fusion = self.fuse(torch.cat([left, right], dim=1))
        return fusion


class SIR(nn.Module):
    def __init__(self, out_planes):
        super(SIR, self).__init__()
        self.conv3x3 = nn.Conv2d(2 * out_planes, 6 * out_planes, kernel_size=3, padding=1)
        self.stage = nn.Sequential(
            nn.GroupNorm(num_groups=16, num_channels=6 * out_planes),
            nn.ELU(inplace=True)
        )
        self.conv1x1 = nn.Conv2d(6 * out_planes, out_planes * 2, kernel_size=1)

    def forward(self, x):
        residual = x
        feat = self.conv3x3(x)
        feat = self.stage(feat)
        feat = self.conv1x1(feat)
        return feat + residual


class IDWR_Branch3(nn.Module):
    def __init__(self, out_planes):
        super(IDWR_Branch3, self).__init__()
        self.conv3x3_bn_relu = nn.Sequential(
            nn.Conv2d(2 * out_planes, 3 * out_planes, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=3 * out_planes),
            nn.ELU(inplace=True)
        )
        self.depth_wise_1 = nn.Conv2d(out_planes, 2 * out_planes, kernel_size=3, padding=1, groups=out_planes)
        self.down_sample = nn.Conv2d(out_planes, 2 * out_planes, kernel_size=1)
        self.depth_wise_2 = nn.Conv2d(out_planes, out_planes, kernel_size=(3, 3), dilation=(3, 3), padding=(3, 3),
                                      groups=out_planes)
        self.depth_wise_3 = nn.Conv2d(out_planes, out_planes, kernel_size=(3, 3), dilation=(5, 5), padding=(5, 5),
                                      groups=out_planes)

        self.fuse = nn.Sequential(
            nn.GroupNorm(num_groups=16, num_channels=4 * out_planes),
            nn.Conv2d(4 * out_planes, 2 * out_planes, kernel_size=1)
        )

    def forward(self, x):
        out = self.conv3x3_bn_relu(x)
        b, c, h, w = out.size()
        out_1, out_2, out_3 = out[:, 0:c // 3, :, :], out[:, c // 3:2 * c // 3, :, :], out[:, 2 * c // 3:c, :, :]
        out_1 = nn.functional.elu_(self.depth_wise_1(out_1) + self.down_sample(out_1))
        out_2 = nn.functional.elu_(self.depth_wise_2(out_2) + out_2)
        out_3 = nn.functional.elu_(self.depth_wise_3(out_3) + out_3)
        feat = self.fuse(torch.cat([out_1, out_2, out_3], dim=1))
        return x + feat


class IDWR_Branch2(nn.Module):
    def __init__(self, out_planes):
        super(IDWR_Branch2, self).__init__()
        self.conv3x3_bn_relu = nn.Sequential(
            nn.Conv2d(2 * out_planes, 2 * out_planes, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=2 * out_planes),
            nn.ELU(inplace=True)
        )
        self.depth_wise_1 = nn.Conv2d(out_planes, 2 * out_planes, kernel_size=3, padding=1, groups=out_planes)
        self.down_sample = nn.Conv2d(out_planes, out_planes * 2, kernel_size=1)
        self.depth_wise_2 = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, kernel_size=(3, 3), dilation=(3, 3), padding=(3, 3), groups=out_planes),
        )

        self.fuse = nn.Sequential(
            nn.GroupNorm(num_groups=16, num_channels=3 * out_planes),
            nn.Conv2d(3 * out_planes, 2 * out_planes, kernel_size=1)
        )

    def forward(self, x):
        out = self.conv3x3_bn_relu(x)
        b, c, h, w = out.size()
        out_1, out_2 = out[:, 0:c // 2, :, :], out[:, c // 2:c, :, :]
        out_1 = nn.functional.elu_(self.depth_wise_1(out_1) + self.down_sample(out_1))
        out_2 = nn.functional.elu_(self.depth_wise_2(out_2) + out_2)
        feat = self.fuse(torch.cat([out_1, out_2], dim=1))
        return x + feat


class SegHead(nn.Module):
    def __init__(self, out_planes, num_classes):
        super(SegHead, self).__init__()
        self.conv3x3_bn_relu = nn.Sequential(
            nn.Conv2d(out_planes, 128, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=16, num_channels=128),
            nn.ELU(inplace=True),
        )
        self.proj = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv3x3_bn_relu(x)
        x = self.proj(x)

        return x


class MIDWRSeg(nn.Module):
    def __init__(self, classes=19):
        super(MIDWRSeg, self).__init__()
        self.stem_stage_1 = Stem(out_planes=64)
        self.SIR_stage_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            SIR(32),
            SIR(32),
            SIR(32),
            SIR(32),
            SIR(32),
            SIR(32),
            SIR(32),
            SIR(32),
        )
        self.mscan = MSCAN(in_chans=64, embed_dims=[64, 128], depths=[3, 5], num_stages=2, drop_path_rate=0.2,
                           mlp_ratios=[8, 4])
        '''
        self.DWR2_stage_3 = nn.Sequential(
            # nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            # MSCAAttention(128),
            # MSCAAttention(128),
            # MSCAAttention(128),
            # IDWR_Branch2(64),
            # IDWR_Branch2(64),
            IDWR_Branch2(64),
            IDWR_Branch2(64),
            IDWR_Branch2(64),
            IDWR_Branch2(64),
            IDWR_Branch2(64),
            IDWR_Branch2(64),
        )
        '''
        self.DWR3_satge_4 = nn.Sequential(
            # nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            IDWR_Branch3(64),
            IDWR_Branch3(64),
            IDWR_Branch3(64),

        )
        self.final_head = SegHead(out_planes=384, num_classes=classes)

    def forward(self, x):
        b, c, h, w = x.size()
        feat_1 = self.stem_stage_1(x)  # [3, 64, 63, 127]

        feat_2 = self.SIR_stage_2(feat_1)  # [3, 64, 32, 64]
        feat_3, feat_4 = self.mscan(feat_2)  # [3, 64, 16, 32],[3, 128, 8, 16]

        feat_5 = self.DWR3_satge_4(feat_4)
        '''
        print('feat_1.shape', feat_1.shape)
        print('feat_2.shape', feat_2.shape)
        print('feat_3.shape', feat_3.shape)
        print('feat_4.shape', feat_4.shape)
        print('feat_5.shape', feat_5.shape)
        '''
        b, c, h1, h2 = feat_2.size()
        up_2 = nn.functional.interpolate(feat_3, size=(h1, h2), mode='bilinear')
        up_3 = nn.functional.interpolate(feat_4, size=(h1, h2), mode='bilinear')
        up_4 = nn.functional.interpolate(feat_5, size=(h1, h2), mode='bilinear')
        final = torch.cat([feat_2, up_2, up_3, up_4], dim=1)
        out = self.final_head(final)
        out = nn.functional.interpolate(out, size=(h, w), mode='bilinear')
        return out


if __name__ == '__main__':
    net = MIDWRSeg(classes=19)
    x = torch.randn(3, 3, 512, 1024)
    print(net(x).shape)
    stat(net, (3, 512, 1024))
