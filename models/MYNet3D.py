import torch
import torch.nn as nn
from models.resnet import resnet18, resnet34
import torch.nn.functional as F
import numpy as np
import math
from torch import nn, einsum
from thop import clever_format,profile
from einops import rearrange
from timm.models.layers import DropPath
from models._common import Attention, AttentionLePE, DWConv
from ops.torch.bra3 import BiLevelRoutingCrossAttention
from models._blocks import Conv1x1, Conv3x3, MaxPool2x2
import misc1


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        # print("++",x.size()[1],self.inp_dim,x.size()[1],self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SimpleResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv3x3(in_ch, out_ch, norm=True, act=True)
        self.conv2 = Conv3x3(out_ch, out_ch, norm=True)

    def forward(self, x):
        x = self.conv1(x)
        return F.relu(x + self.conv2(x))


class BasicConv3D(nn.Module):
    def __init__(
        self, in_ch, out_ch,
        kernel_size,
        bias='auto',
        bn=False, act=False,
        **kwargs
    ):
        super().__init__()
        seq = []
        if kernel_size >= 2:
            seq.append(nn.ConstantPad3d(kernel_size//2, 0.0))
        seq.append(
            nn.Conv3d(
                in_ch, out_ch, kernel_size,
                padding=0,
                bias=(False if bn else True) if bias=='auto' else bias,
                **kwargs
            )
        )
        if bn:
            seq.append(nn.BatchNorm3d(out_ch))
        if act:
            seq.append(nn.ReLU())
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)


class Conv3x3x3(BasicConv3D):
    def __init__(self, in_ch, out_ch, bias='auto', bn=False, act=False, **kwargs):
        super().__init__(in_ch, out_ch, 3, bias=bias, bn=bn, act=act, **kwargs)


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, itm_ch, stride=1, ds=None):
        super().__init__()
        self.conv1 = BasicConv3D(in_ch, itm_ch, 1, bn=True, act=True, stride=stride)
        self.conv2 = Conv3x3x3(itm_ch, itm_ch, bn=True, act=True)
        self.conv3 = BasicConv3D(itm_ch, out_ch, 1, bn=True, act=False)
        self.ds = ds

    def forward(self, x):
        res = x
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        if self.ds is not None:
            res = self.ds(res)
        y = F.relu(y + res)
        return y


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, itm_ch, stride=1, ds=None):
        super().__init__()
        self.conv1 = BasicConv2d(in_ch, itm_ch, stride=stride)
        # self.conv1 = Conv3x3(in_ch, itm_ch, norm=True, act=True)
        self.conv2 = Conv3x3(itm_ch, itm_ch, norm=True, act=True)
        # self.conv3 = Conv3x3(itm_ch, out_ch, norm=True)
        self.conv3 = BasicConv2d(itm_ch, out_ch)
        self.ds = ds

    def forward(self, x):
        res = x
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        if self.ds is not None:
            res = self.ds(res)
        y = F.relu(y + res)
        return y


class TMEncoder(nn.Module):
    def __init__(self, in_ch, enc_chs=(32, 64)):
        super().__init__()
        if in_ch != 3:
            raise NotImplementedError

        self.n_layers = 2
        self.expansion = 2
        self.tem_scales = (1.0, 0.5)

        self.stem = nn.Sequential(
            nn.Conv3d(3, enc_chs[0], kernel_size=(3, 9, 9), stride=(1, 4, 4), padding=(1, 4, 4), bias=False),
            # nn.Conv3d(enc_chs[0], enc_chs[0], 3, 1, 1, bias=True, groups=enc_chs[0]),
            nn.BatchNorm3d(enc_chs[0]),
            nn.ReLU()
        )
        exps = self.expansion
        self.layer1 = nn.Sequential(
            ResBlock3D(
                enc_chs[0],
                enc_chs[0] * exps,
                enc_chs[0],
                ds=BasicConv3D(enc_chs[0], enc_chs[0] * exps, 1, bn=True)
            ),
            ResBlock3D(enc_chs[0] * exps, enc_chs[0] * exps, enc_chs[0])
        )
        self.layer2 = nn.Sequential(
            ResBlock3D(
                enc_chs[0] * exps,
                enc_chs[1] * exps,
                enc_chs[1],
                stride=(2, 2, 2),
                ds=BasicConv3D(enc_chs[0] * exps, enc_chs[1] * exps, 1, stride=(2, 2, 2), bn=True)
            ),
            ResBlock3D(enc_chs[1] * exps, enc_chs[1] * exps, enc_chs[1])
        )

    def forward(self, x):
        feats = []

        x = self.stem(x)
        for i in range(self.n_layers):
            layer = getattr(self, f'layer{i + 1}')
            x = layer(x)
            feats.append(x)

        return feats


class DecBlock(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch):
        super().__init__()
        self.conv_fuse = SimpleResBlock(in_ch1 + in_ch2, out_ch)
        self.conv_out = Conv1x1(out_ch, 1)

    def forward(self, x1, x2):
        x2 = F.interpolate(x2, size=x1.shape[2:])
        x = torch.cat([x1, x2], dim=1)
        # print(x.shape)
        outs = self.conv_fuse(x)
        output = self.conv_out(outs)
        return outs, output
        # return outs


class SpatialEncoder(nn.Module):
    def __init__(self, in_ch, topk):
        super().__init__()
        self.n_layers = 3
        self.block = Block(dim=in_ch, topk=topk)

    def forward(self, t1, t2):    # (1,64,64,64) (1,128,32,32) (1,256,16,16)
        feat = self.block(t1, t2)
        return feat


class SimpleDecoder(nn.Module):
    def __init__(self, itm_ch, enc_chs, dec_chs):  # 256, 128, 64, 32
        super().__init__()

        enc_chs = enc_chs[::-1]  # 256,128,64,6
        self.conv_bottom = Conv3x3(itm_ch, itm_ch, norm=True, act=True)

        self.decoder1 = DecBlock(256, 256, 256)
        self.decoder2 = DecBlock(128, 256, 128)
        self.decoder3 = DecBlock(64, 128, 64)
        self.decoder4 = DecBlock(6, 64, 32)

        self.conv_out = Conv1x1(dec_chs[-1], 1)

    def forward(self, x, feats):
        feats = feats[::-1]  # 倒置
        # print(x.shape)
        # for key in feats:
        #     print(key.shape)

        x = self.conv_bottom(x)  # [1, 256, 16, 16]
        x, middle_out1 = self.decoder1(feats[0], x)  # [1, 256, 16, 16]
        x, middle_out2 = self.decoder2(feats[1], x)  # [1, 128, 32, 32]
        x, middle_out3 = self.decoder3(feats[2], x)  # [1, 64, 64, 64]
        x, output = self.decoder4(feats[3], x)       # [1, 32, 256, 256]

        return output, middle_out1, middle_out2, middle_out3


class RGBConcatConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(RGBConcatConv, self).__init__()
        self.conv = nn.Conv2d(in_channels*2, out_channels, kernel_size, stride, padding)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=3)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        return F.relu(self.norm(x))


class ConvReduce(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvReduce, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, drop_path=0.1, layer_scale_init_value=-1,
                 num_heads=8, n_win=8, qk_dim=None, qk_scale=None,
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='ada_avgpool',
                 topk=2, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False,
                 mlp_ratio=3, mlp_dwconv=True,
                 side_dwconv=5, before_attn_dwconv=3, pre_norm=True, auto_pad=False):
        super().__init__()     # drop_path=0., mlp_ratio=4, mlp_dwconv=False,
        qk_dim = qk_dim or dim

        # modules
        if before_attn_dwconv > 0:
            self.pos_embed = nn.Conv2d(dim, dim, kernel_size=before_attn_dwconv, padding=1, groups=dim)
        else:
            self.pos_embed = lambda x: 0
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)  # important to avoid attention collapsing
        if topk > 0:
            self.attn = BiLevelRoutingCrossAttention(dim=dim, num_heads=num_heads, n_win=n_win, qk_dim=qk_dim,
                                                qk_scale=qk_scale, kv_per_win=kv_per_win,
                                                kv_downsample_ratio=kv_downsample_ratio,
                                                kv_downsample_kernel=kv_downsample_kernel,
                                                kv_downsample_mode=kv_downsample_mode,
                                                topk=topk, param_attention=param_attention, param_routing=param_routing,
                                                diff_routing=diff_routing, soft_routing=soft_routing,
                                                side_dwconv=side_dwconv,
                                                auto_pad=auto_pad)
        self.pre_norm = pre_norm
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio * dim)),
                                 DWConv(int(mlp_ratio * dim)) if mlp_dwconv else nn.Identity(),
                                 nn.GELU(),
                                 nn.Linear(int(mlp_ratio * dim), dim)
                                 )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.concat_chs = RGBConcatConv(in_channels=dim, out_channels=dim)

    def forward(self, t1, t2):
        # conv pos embedding  3×3卷积，一个残差连接
        t1 = t1 + self.pos_embed(t1)
        t2 = t2 + self.pos_embed(t2)
        # permute to NHWC tensor for attention
        t1 = t1.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        t2 = t2.permute(0, 2, 3, 1)
        # x = self.concat_chs(t1, t2)

        # attention & mlp
        if self.pre_norm:
            x1, x2 = self.attn(self.norm1(t1), self.norm1(t2))
            t1 = t1 + self.drop_path(x1)
            t2 = t2 + self.drop_path(x2)
            # x = x + self.drop_path(self.attn(self.norm1(t1), self.norm1(t2)))  # (N, H, W, C)
            # x = x + self.drop_path(self.mlp(self.norm1(x)))  # (N, H, W, C)
            t1 = t1 + self.drop_path(self.mlp(self.norm1(t1)))  # (N, H, W, C)
            t2 = t2 + self.drop_path(self.mlp(self.norm1(t2)))  # (N, H, W, C)

        # permute back
        x = self.concat_chs(t1, t2)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        return x


class MYNet3D(nn.Module):
    def __init__(self, len=8, normal_init=True, in_ch=(64, 128, 256), dec_chs=(256, 128, 64, 32),
                 topks=(4, 16, 32), pretrained=False):  # (4,16,32)
        super(MYNet3D, self).__init__()

        self.resnet = resnet18()
        self.resnet.load_state_dict(torch.load('../pretrained/resnet18-5c106cde.pth'))
        self.resnet.layer4 = nn.Identity()

        self.crossAttention1 = SpatialEncoder(in_ch[0], topk=topks[0])
        self.crossAttention2 = SpatialEncoder(in_ch[1], topk=topks[1])
        self.crossAttention3 = SpatialEncoder(in_ch[2], topk=topks[2])

        self.sigmoid = nn.Sigmoid()

        self.len = len

        self.encoder_v = TMEncoder(3, enc_chs=(32, 64))  # (32,64)

        self.convs_video = nn.ModuleList(
            [
                Conv1x1(2 * ch, ch, norm=True, act=True)
                for ch in (64, 128)
            ]
        )

        self.conv_out_v = Conv1x1(128, 1)

        self.decoder = SimpleDecoder(in_ch[-1], (6,) + in_ch, dec_chs)

        self.ConvReduce1 = ConvReduce(in_channels=64+64, out_channels=64)
        self.ConvReduce2 = ConvReduce(in_channels=128+128, out_channels=128)

        self.concat_chs = RGBConcatConv(in_channels=3, out_channels=3)

        if normal_init:
            self.init_weights()

    def pair_to_video(self, im1, im2, rate_map=None):
        def _interpolate(im1, im2, rate_map, len):
            delta = 1.0/(len-1)
            delta_map = rate_map * delta  # 1/7  在时间维度上进行切割 (1,1,256,256)
            steps = torch.arange(len, dtype=torch.float, device=delta_map.device).view(1,-1,1,1,1)  # tensor[0., 1., 2., 3., 4., 5., 6., 7.]
            interped = im1.unsqueeze(1)+((im2-im1)*delta_map).unsqueeze(1)*steps
            return interped

        if rate_map is None:
            rate_map = torch.ones_like(im1[:,0:1])
        frames = _interpolate(im1, im2, rate_map, 8)
        frame = rearrange(frames, "n l c h w -> n c l h w")
        return frame

    def generate_transition_video_tensor(self, frame1, frame2, num_frames=8):
        transition_frames = []

        for t in torch.linspace(0, 1, num_frames):
            weighted_frame1 = frame1 * (1 - t)
            weighted_frame2 = frame2 * t
            blended_frame = weighted_frame1 + weighted_frame2
            transition_frames.append(blended_frame.unsqueeze(0))

        transition_video = torch.cat(transition_frames, dim=0)
        frame = rearrange(transition_video, "l n c h w -> n c l h w")
        return frame

    def tem_aggr(self, f):
        return torch.cat([torch.mean(f, dim=2), torch.max(f, dim=2)[0]], dim=1)

    def forward(self, imgs1, imgs2, labels=None, return_aux=True):
        # frames = self.pair_to_video(imgs1, imgs2)  # [1, 3, 8, 256, 256]
        frames = self.generate_transition_video_tensor(imgs1, imgs2)
        feats_v = self.encoder_v(frames)

        for i, feat in enumerate(feats_v):
            feats_v[i] = self.convs_video[i](self.tem_aggr(feat))

        for keys in feats_v:
            print(keys.shape)

        c1 = self.resnet.conv1(imgs1)
        c1 = self.resnet.bn1(c1)
        c1 = self.resnet.relu(c1)
        c1 = self.resnet.maxpool(c1)

        c1 = self.resnet.layer1(c1)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)

        c1_imgs2 = self.resnet.conv1(imgs2)
        c1_imgs2 = self.resnet.bn1(c1_imgs2)
        c1_imgs2 = self.resnet.relu(c1_imgs2)
        c1_imgs2 = self.resnet.maxpool(c1_imgs2)

        c1_imgs2 = self.resnet.layer1(c1_imgs2)  # [1, 64, 64, 64]
        c2_imgs2 = self.resnet.layer2(c1_imgs2)  # [1, 128, 32, 32]
        c3_imgs2 = self.resnet.layer3(c2_imgs2)  # [1, 256, 16, 16]

        att0 = torch.cat([imgs1, imgs2], dim=1)  # 1,6,256,256
        att1 = self.crossAttention1(c1, c1_imgs2)  # 1,64,64,64
        att2 = self.crossAttention2(c2, c2_imgs2)  # 1,128,32,32
        att3 = self.crossAttention3(c3, c3_imgs2)  # 1,256,16,16

        att1 = self.ConvReduce1(torch.cat([att1, feats_v[0]], dim=1))
        att2 = self.ConvReduce2(torch.cat([att2, feats_v[1]], dim=1))

        feats_p = [att0, att1, att2, att3]

        pred, middle_out1, middle_out2, middle_out3 = self.decoder(feats_p[-1], feats_p)  # (32,64)
        pred = self.sigmoid(pred)  # (1,1,256,256)

        if return_aux:
            pred_v = self.conv_out_v(feats_v[-1])
            pred_v = F.interpolate(pred_v, size=pred.shape[2:])  # (1,1,32,32)->(1,1,256,256)
            pred_v = self.sigmoid(pred_v)

            # middle_out1 = F.interpolate(middle_out1, size=middle_out3.shape[2:])
            # middle_out2 = F.interpolate(middle_out2, size=middle_out3.shape[2:])   # (1,1,64,64)
            middle_out2 = F.interpolate(middle_out2, size=pred.shape[2:])  # (1,1,256,256)
            # middle_out3 = F.interpolate(middle_out3, size=pred.shape[2:])   # (1,1,256,256)
            # middle_out1 = self.sigmoid(middle_out1)
            middle_out2 = self.sigmoid(middle_out2)
            # middle_out3 = self.sigmoid(middle_out3)
            return pred, pred_v, middle_out2
            # return pred, pred_v, middle_out2, middle_out3
        else:
            return pred

    def init_weights(self):
        pass


def init_conv(conv, glu=True):
    nn.init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


if __name__ == '__main__':

    input1 = torch.randn(1, 3, 256, 256).cuda(1)
    input2 = torch.randn(1, 3, 256, 256).cuda(1)

    model = MYNet3D(pretrained=True).cuda(1)
    misc1.print_module_summary(model, [input1, input2])
    flops, params = profile(model, inputs=(input1, input2))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops)
    print(params)
    out, out_v = model(input1, input2)
    print(out_v.shape)