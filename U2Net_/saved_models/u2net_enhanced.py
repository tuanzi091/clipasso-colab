import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return x * self.sigmoid(attention)

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class WeightedBiDirectionalPyramid(nn.Module):
    def __init__(self, channels_list, fusion_type='add'):
        super(WeightedBiDirectionalPyramid, self).__init__()
        self.fusion_type = fusion_type
        self.weights = nn.Parameter(torch.ones(len(channels_list)) / len(channels_list))
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        for i in range(len(channels_list) - 1):
            self.down_convs.append(
                nn.Sequential(
                    nn.Conv2d(channels_list[i], channels_list[i+1], 3, stride=2, padding=1),
                    nn.BatchNorm2d(channels_list[i+1]),
                    nn.ReLU(inplace=True)
                )
            )

        for i in range(len(channels_list) - 1):
            self.up_convs.append(
                nn.Sequential(
                    nn.Conv2d(channels_list[len(channels_list)-1-i], channels_list[len(channels_list)-2-i], 3, padding=1),
                    nn.BatchNorm2d(channels_list[len(channels_list)-2-i]),
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, features):
        normalized_weights = F.softmax(self.weights, dim=0)

        top_down_features = []
        current = features[0]
        for i, down_conv in enumerate(self.down_convs):
            current = down_conv(current)
            top_down_features.append(current)

        bottom_up_features = []
        current = features[-1]
        bottom_up_features.append(current)

        for i in range(len(self.up_convs)):
            target_idx = len(features) - i - 2
            interp_size = features[target_idx].shape[2:]
            interpolated = F.interpolate(current, size=interp_size, mode='bilinear', align_corners=True)

            bottom_up_idx = len(bottom_up_features) - 1
            actual_channels = bottom_up_features[bottom_up_idx].shape[1]
            expected_channels = interpolated.shape[1]

            if actual_channels != expected_channels:
                adapter = nn.Conv2d(actual_channels, expected_channels, 1).to(current.device)
                interpolated = adapter(interpolated)

            current = self.up_convs[i](interpolated)
            bottom_up_features.append(current)

        bottom_up_features = bottom_up_features[::-1]

        target_size = features[0].shape[2:]

        fused_features = []
        for i in range(len(features)):
            feat_i = features[i]
            target_size_i = feat_i.shape[2:]
            td_i = F.interpolate(top_down_features[i], size=target_size_i, mode='bilinear', align_corners=True) if i < len(top_down_features) else None
            bu_i = F.interpolate(bottom_up_features[i], size=target_size_i, mode='bilinear', align_corners=True) if i < len(bottom_up_features) else None

            if self.fusion_type == 'add':
                if td_i is not None and bu_i is not None:
                    weighted_sum = normalized_weights[i] * (feat_i + td_i + bu_i)
                elif td_i is not None:
                    weighted_sum = normalized_weights[i] * (feat_i + td_i)
                elif bu_i is not None:
                    weighted_sum = normalized_weights[i] * (feat_i + bu_i)
                else:
                    weighted_sum = normalized_weights[i] * feat_i
            elif self.fusion_type == 'mul':
                if td_i is not None and bu_i is not None:
                    weighted_sum = normalized_weights[i] * (feat_i * td_i * bu_i)
                elif td_i is not None:
                    weighted_sum = normalized_weights[i] * (feat_i * td_i)
                elif bu_i is not None:
                    weighted_sum = normalized_weights[i] * (feat_i * bu_i)
                else:
                    weighted_sum = normalized_weights[i] * feat_i
            else:
                tensors_to_cat = [feat_i]
                if td_i is not None:
                    tensors_to_cat.append(td_i)
                if bu_i is not None:
                    tensors_to_cat.append(bu_i)
                weighted_sum = normalized_weights[i] * torch.cat(tensors_to_cat, dim=1)
            fused_features.append(weighted_sum)

        return fused_features

class EnhancedREBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1, use_cbam=True):
        super(EnhancedREBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM(out_ch)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        if self.use_cbam:
            xout = self.cbam(xout)

        return xout

def _upsample_like(src, tar):
    src = F.upsample(src, size=tar.shape[2:], mode='bilinear')
    return src

class EnhancedRSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, use_cbam=True):
        super(EnhancedRSU7, self).__init__()

        self.rebnconvin = EnhancedREBNCONV(in_ch, out_ch, dirate=1, use_cbam=use_cbam)

        self.rebnconv1 = EnhancedREBNCONV(out_ch, mid_ch, dirate=1, use_cbam=use_cbam)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = EnhancedREBNCONV(mid_ch, mid_ch, dirate=1, use_cbam=use_cbam)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = EnhancedREBNCONV(mid_ch, mid_ch, dirate=1, use_cbam=use_cbam)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = EnhancedREBNCONV(mid_ch, mid_ch, dirate=1, use_cbam=use_cbam)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = EnhancedREBNCONV(mid_ch, mid_ch, dirate=1, use_cbam=use_cbam)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = EnhancedREBNCONV(mid_ch, mid_ch, dirate=1, use_cbam=use_cbam)

        self.rebnconv7 = EnhancedREBNCONV(mid_ch, mid_ch, dirate=2, use_cbam=use_cbam)

        self.rebnconv6d = EnhancedREBNCONV(mid_ch*2, mid_ch, dirate=1, use_cbam=use_cbam)
        self.rebnconv5d = EnhancedREBNCONV(mid_ch*2, mid_ch, dirate=1, use_cbam=use_cbam)
        self.rebnconv4d = EnhancedREBNCONV(mid_ch*2, mid_ch, dirate=1, use_cbam=use_cbam)
        self.rebnconv3d = EnhancedREBNCONV(mid_ch*2, mid_ch, dirate=1, use_cbam=use_cbam)
        self.rebnconv2d = EnhancedREBNCONV(mid_ch*2, mid_ch, dirate=1, use_cbam=use_cbam)
        self.rebnconv1d = EnhancedREBNCONV(mid_ch*2, out_ch, dirate=1, use_cbam=use_cbam)

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

class EnhancedRSU6(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, use_cbam=True):
        super(EnhancedRSU6, self).__init__()

        self.rebnconvin = EnhancedREBNCONV(in_ch, out_ch, dirate=1, use_cbam=use_cbam)

        self.rebnconv1 = EnhancedREBNCONV(out_ch, mid_ch, dirate=1, use_cbam=use_cbam)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = EnhancedREBNCONV(mid_ch, mid_ch, dirate=1, use_cbam=use_cbam)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = EnhancedREBNCONV(mid_ch, mid_ch, dirate=1, use_cbam=use_cbam)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = EnhancedREBNCONV(mid_ch, mid_ch, dirate=1, use_cbam=use_cbam)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = EnhancedREBNCONV(mid_ch, mid_ch, dirate=1, use_cbam=use_cbam)

        self.rebnconv6 = EnhancedREBNCONV(mid_ch, mid_ch, dirate=2, use_cbam=use_cbam)

        self.rebnconv5d = EnhancedREBNCONV(mid_ch*2, mid_ch, dirate=1, use_cbam=use_cbam)
        self.rebnconv4d = EnhancedREBNCONV(mid_ch*2, mid_ch, dirate=1, use_cbam=use_cbam)
        self.rebnconv3d = EnhancedREBNCONV(mid_ch*2, mid_ch, dirate=1, use_cbam=use_cbam)
        self.rebnconv2d = EnhancedREBNCONV(mid_ch*2, mid_ch, dirate=1, use_cbam=use_cbam)
        self.rebnconv1d = EnhancedREBNCONV(mid_ch*2, out_ch, dirate=1, use_cbam=use_cbam)

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

class EnhancedRSU5(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, use_cbam=True):
        super(EnhancedRSU5, self).__init__()

        self.rebnconvin = EnhancedREBNCONV(in_ch, out_ch, dirate=1, use_cbam=use_cbam)

        self.rebnconv1 = EnhancedREBNCONV(out_ch, mid_ch, dirate=1, use_cbam=use_cbam)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = EnhancedREBNCONV(mid_ch, mid_ch, dirate=1, use_cbam=use_cbam)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = EnhancedREBNCONV(mid_ch, mid_ch, dirate=1, use_cbam=use_cbam)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = EnhancedREBNCONV(mid_ch, mid_ch, dirate=1, use_cbam=use_cbam)

        self.rebnconv5 = EnhancedREBNCONV(mid_ch, mid_ch, dirate=2, use_cbam=use_cbam)

        self.rebnconv4d = EnhancedREBNCONV(mid_ch*2, mid_ch, dirate=1, use_cbam=use_cbam)
        self.rebnconv3d = EnhancedREBNCONV(mid_ch*2, mid_ch, dirate=1, use_cbam=use_cbam)
        self.rebnconv2d = EnhancedREBNCONV(mid_ch*2, mid_ch, dirate=1, use_cbam=use_cbam)
        self.rebnconv1d = EnhancedREBNCONV(mid_ch*2, out_ch, dirate=1, use_cbam=use_cbam)

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

class EnhancedRSU4(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, use_cbam=True):
        super(EnhancedRSU4, self).__init__()

        self.rebnconvin = EnhancedREBNCONV(in_ch, out_ch, dirate=1, use_cbam=use_cbam)

        self.rebnconv1 = EnhancedREBNCONV(out_ch, mid_ch, dirate=1, use_cbam=use_cbam)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = EnhancedREBNCONV(mid_ch, mid_ch, dirate=1, use_cbam=use_cbam)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = EnhancedREBNCONV(mid_ch, mid_ch, dirate=1, use_cbam=use_cbam)

        self.rebnconv4 = EnhancedREBNCONV(mid_ch, mid_ch, dirate=2, use_cbam=use_cbam)

        self.rebnconv3d = EnhancedREBNCONV(mid_ch*2, mid_ch, dirate=1, use_cbam=use_cbam)
        self.rebnconv2d = EnhancedREBNCONV(mid_ch*2, mid_ch, dirate=1, use_cbam=use_cbam)
        self.rebnconv1d = EnhancedREBNCONV(mid_ch*2, out_ch, dirate=1, use_cbam=use_cbam)

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

class U2NETEnhanced(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, use_cbam=True, use_pyramid=True):
        super(U2NETEnhanced, self).__init__()

        self.use_cbam = use_cbam
        self.use_pyramid = use_pyramid

        self.stage1 = EnhancedRSU7(in_ch, 32, 64, use_cbam=use_cbam)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = EnhancedRSU6(64, 32, 128, use_cbam=use_cbam)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = EnhancedRSU5(128, 32, 256, use_cbam=use_cbam)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = EnhancedRSU4(256, 32, 512, use_cbam=use_cbam)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = EnhancedRSU4(512, 32, 512, use_cbam=use_cbam)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = EnhancedRSU4(512, 32, 512, use_cbam=use_cbam)

        self.stage5d = EnhancedRSU4(1024, 32, 512, use_cbam=use_cbam)
        self.stage4d = EnhancedRSU4(1024, 32, 256, use_cbam=use_cbam)
        self.stage3d = EnhancedRSU5(512, 32, 128, use_cbam=use_cbam)
        self.stage2d = EnhancedRSU6(256, 32, 64, use_cbam=use_cbam)
        self.stage1d = EnhancedRSU7(128, 32, 64, use_cbam=use_cbam)

        self.stage1d_final = nn.Conv2d(64, out_ch, 3, padding=1)
        self.stage2d_final = nn.Conv2d(64, out_ch, 3, padding=1)
        self.stage3d_final = nn.Conv2d(128, out_ch, 3, padding=1)
        self.stage4d_final = nn.Conv2d(256, out_ch, 3, padding=1)
        self.stage5d_final = nn.Conv2d(512, out_ch, 3, padding=1)
        self.stage6d_final = nn.Conv2d(512, out_ch, 3, padding=1)

        if use_pyramid:
            self.pyramid = WeightedBiDirectionalPyramid([64, 128, 256, 512, 512], fusion_type='add')

    def forward(self, x):
        hx = x

        hx1 = self.stage1(hx)
        hx = self.pool1(hx1)

        hx2 = self.stage2(hx)
        hx = self.pool2(hx2)

        hx3 = self.stage3(hx)
        hx = self.pool3(hx3)

        hx4 = self.stage4(hx)
        hx = self.pool4(hx4)

        hx5 = self.stage5(hx)
        hx = self.pool5(hx5)

        hx6 = self.stage6(hx)

        if self.use_pyramid and hasattr(self, 'pyramid'):
            features = [hx1, hx2, hx3, hx4, hx5]
            fused_features = self.pyramid(features)
            hx1_f, hx2_f, hx3_f, hx4_f, hx5_f = fused_features
        else:
            hx1_f, hx2_f, hx3_f, hx4_f, hx5_f = hx1, hx2, hx3, hx4, hx5

        hx6up = _upsample_like(hx6, hx5)
        hx5d = self.stage5d(torch.cat((hx6up, hx5_f), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4_f), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3_f), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2_f), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1_f), 1))

        hx1d_final = self.stage1d_final(hx1d)
        hx2d_final = self.stage2d_final(hx2d)
        hx3d_final = self.stage3d_final(hx3d)
        hx4d_final = self.stage4d_final(hx4d)
        hx5d_final = self.stage5d_final(hx5d)
        hx6d_final = self.stage6d_final(hx6)

        return (torch.sigmoid(hx1d_final),
                torch.sigmoid(hx2d_final),
                torch.sigmoid(hx3d_final),
                torch.sigmoid(hx4d_final),
                torch.sigmoid(hx5d_final),
                torch.sigmoid(hx6d_final))

def U2NETPEnhanced():
    return U2NETEnhanced(in_ch=3, out_ch=1, use_cbam=True, use_pyramid=False)

def U2NETEnhancedLite():
    return U2NETEnhanced(in_ch=3, out_ch=1, use_cbam=True, use_pyramid=False)
