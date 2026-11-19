import torch
import torch.nn as nn
import torch.nn.functional as F

# CBAM 注意力机制模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out

# 带 CBAM 的 REBNCONV
class REBNCONV_CBAM(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV_CBAM, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_ch)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))
        xout = self.cbam(xout)
        return xout

## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):
    src = F.upsample(src, size=tar.shape[2:], mode='bilinear')
    return src

### RSU-7 with CBAM ###
class RSU7_CBAM(nn.Module):
    def __init__(self, in_ch=3, mid_ch=32, out_ch=64):
        super(RSU7_CBAM, self).__init__()

        self.rebnconvin = REBNCONV_CBAM(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV_CBAM(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV_CBAM(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV_CBAM(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV_CBAM(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV_CBAM(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV_CBAM(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = REBNCONV_CBAM(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV_CBAM(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV_CBAM(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV_CBAM(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV_CBAM(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV_CBAM(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV_CBAM(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin

### RSU-6 with CBAM ###
class RSU6_CBAM(nn.Module):
    def __init__(self, in_ch=3, mid_ch=32, out_ch=64):
        super(RSU6_CBAM, self).__init__()

        self.rebnconvin = REBNCONV_CBAM(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV_CBAM(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV_CBAM(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV_CBAM(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV_CBAM(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV_CBAM(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = REBNCONV_CBAM(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV_CBAM(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV_CBAM(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV_CBAM(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV_CBAM(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV_CBAM(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin

### RSU-5 with CBAM ###
class RSU5_CBAM(nn.Module):
    def __init__(self, in_ch=3, mid_ch=32, out_ch=64):
        super(RSU5_CBAM, self).__init__()

        self.rebnconvin = REBNCONV_CBAM(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV_CBAM(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV_CBAM(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV_CBAM(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV_CBAM(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV_CBAM(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV_CBAM(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV_CBAM(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV_CBAM(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV_CBAM(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin

### RSU-4 with CBAM ###
class RSU4_CBAM(nn.Module):
    def __init__(self, in_ch=3, mid_ch=32, out_ch=64):
        super(RSU4_CBAM, self).__init__()

        self.rebnconvin = REBNCONV_CBAM(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV_CBAM(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV_CBAM(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV_CBAM(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV_CBAM(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV_CBAM(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV_CBAM(mid_ch*2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV_CBAM(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin

### RSU-4F with CBAM ###
class RSU4F_CBAM(nn.Module):
    def __init__(self, in_ch=3, mid_ch=32, out_ch=64):
        super(RSU4F_CBAM, self).__init__()

        self.rebnconvin = REBNCONV_CBAM(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV_CBAM(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV_CBAM(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV_CBAM(mid_ch, mid_ch, dirate=4)
        self.rebnconv4 = REBNCONV_CBAM(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV_CBAM(mid_ch*2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV_CBAM(mid_ch*2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV_CBAM(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)
        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin

### U2NET with CBAM ###
class U2NET_CBAM(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET_CBAM, self).__init__()

        # Encoder with mid_ch=32 for all stages
        self.stage1 = RSU7_CBAM(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6_CBAM(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5_CBAM(128, 32, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4_CBAM(256, 32, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F_CBAM(512, 32, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F_CBAM(512, 32, 512)

        # Decoder with mid_ch=32 for all stages
        self.stage5d = RSU4F_CBAM(1024, 32, 512)
        self.stage4d = RSU4_CBAM(1024, 32, 256)
        self.stage3d = RSU5_CBAM(512, 32, 128)
        self.stage2d = RSU6_CBAM(256, 32, 64)
        self.stage1d = RSU7_CBAM(128, 32, 64)

        self.stage1d_final = nn.Conv2d(64, out_ch, 3, padding=1)
        self.stage2d_final = nn.Conv2d(64, out_ch, 3, padding=1)
        self.stage3d_final = nn.Conv2d(128, out_ch, 3, padding=1)
        self.stage4d_final = nn.Conv2d(256, out_ch, 3, padding=1)
        self.stage5d_final = nn.Conv2d(512, out_ch, 3, padding=1)
        self.stage6d_final = nn.Conv2d(512, out_ch, 3, padding=1)

    def forward(self, x):
        hx = x

        # stage 1
        h1 = self.stage1(hx)
        hx = self.pool12(h1)

        # stage 2
        h2 = self.stage2(hx)
        hx = self.pool23(h2)

        # stage 3
        h3 = self.stage3(hx)
        hx = self.pool34(h3)

        # stage 4
        h4 = self.stage4(hx)
        hx = self.pool45(h4)

        # stage 5
        h5 = self.stage5(hx)
        hx = self.pool56(h5)

        # stage 6
        h6 = self.stage6(hx)
        h6up = _upsample_like(h6, h5)

        # decoder
        h5d = self.stage5d(torch.cat((h6up, h5), 1))
        h5dup = _upsample_like(h5d, h4)

        h4d = self.stage4d(torch.cat((h5dup, h4), 1))
        h4dup = _upsample_like(h4d, h3)

        h3d = self.stage3d(torch.cat((h4dup, h3), 1))
        h3dup = _upsample_like(h3d, h2)

        h2d = self.stage2d(torch.cat((h3dup, h2), 1))
        h2dup = _upsample_like(h2d, h1)

        h1d = self.stage1d(torch.cat((h2dup, h1), 1))

        # side output
        d1 = self.stage1d_final(h1d)

        d2 = self.stage2d_final(h2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.stage3d_final(h3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.stage4d_final(h4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.stage5d_final(h5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.stage6d_final(h6)
        d6 = _upsample_like(d6, d1)

        return F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6), F.sigmoid(d1)