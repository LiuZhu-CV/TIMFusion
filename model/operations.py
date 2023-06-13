import torch
import torch.nn as nn
import numpy as np
from .antialias import Downsample as downsamp
import torch.nn.functional as F

# Channels, stride, bn anf kernel_size

OPS = {
    'Denseblocks': lambda C, kernel, dialtion, affine: ResidualDenseBlock(C, kernel, dialtion),
    'Residualblocks': lambda C, kernel, dialtion, affine: ResidualModule(C, kernel, dialtion),
    'ECAattention': lambda C, kernel, dialtion, affine: ECABasicBlock(C, C, kernel, dialtion),
    'SPAattention': lambda C, kernel, dialtion, affine: Spatial_BasicBlock(C, C, kernel, dialtion),
    'NonLocalattention': lambda C, kernel, dialtion, affine: NLBasicBlock(C, C, kernel, dialtion),
    'DilConv': lambda C, kernel, dialtion, affine: DilConv(C, C, kernel, dialtion),
    'SepConv': lambda C, kernel, dialtion, affine: SepConv(C, C, kernel, dialtion, padding=1),
}


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, dilation=1, groups=1, relu=True, bn=False,
                 bias=False):
        super(BasicConv, self).__init__()
        # judge
        stride = 1
        padding = 0
        if kernel_size == 3 and dilation == 1:
            padding = 1
        if kernel_size == 3 and dilation == 2:
            padding = 2
        if kernel_size == 5 and dilation == 1:
            padding = 2
        if kernel_size == 5 and dilation == 2:
            padding = 4
        if kernel_size == 7 and dilation == 1:
            padding = 3
        if kernel_size == 7 and dilation == 2:
            padding = 6
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.PReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class NonLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels, bias=False):
        super(NonLocalBlock2D, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                           padding=0, bias=bias)

        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1,
                           padding=0, bias=bias)
        # for pytorch 0.3.1
        # nn.init.constant(self.W.weight, 0)
        # nn.init.constant(self.W.bias, 0)
        # for pytorch 0.4.0
        nn.init.constant_(self.W.weight, 0)
        # nn.init.constant_(self.W.bias, 0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                               padding=0, bias=bias)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                             padding=0, bias=bias)

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)

        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)

        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)

        f_div_C = F.softmax(f, dim=1)

        y = torch.matmul(f_div_C, g_x)

        y = y.permute(0, 2, 1).contiguous()

        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z


class NLBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel=3, dilation=1, stride=1, with_norm=False):
        super(NLBasicBlock, self).__init__()
        self.with_norm = with_norm

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv2 = BasicConv(inplanes, inplanes, kernel, relu=False)
        self.se = NonLocalBlock2D(inplanes, inplanes)
        self.relu = nn.PReLU()
        if self.with_norm:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = x = self.conv1(x)
        if self.with_norm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.se(out)
        out += x
        out = self.conv2(out)
        if self.with_norm:
            out = self.bn2(out)
        out = self.relu(out)
        return out


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class ChannelPool2(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class ChannelPool3(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=5):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, relu=False)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class spatial_attn_layer2(nn.Module):
    def __init__(self, kernel_size=1):
        super(spatial_attn_layer2, self).__init__()
        self.compress = ChannelPool2()
        self.spatial = BasicConv(3, 1, kernel_size, relu=False)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return scale


class Mask_based_WSM(nn.Module):
    def __init__(self):
        super(Mask_based_WSM, self).__init__()

    def getmask(self, image_mask):
        ## image_mask h*w
        out = torch.histc(image_mask, bins=256, max=255)
        mask_output = torch.zeros([256])
        for i in range(256):
            for j in range(256):
                mask_output[i] += np.abs(j - i) * out[j]
        mask = torch.zeros_like(image_mask)
        for i in range(256):
            mask[torch.where(image_mask == i)] = mask_output[i]
        if mask.max() == 0:
            return torch.zeros_like(image_mask)
        return image_mask

    def forward(self, image_irr, image_vis):
        image_irr = image_irr * 255
        image_vis = image_vis * 255
        b, c, h, w = image_irr.shape
        masks_bi = []
        masks_bi_vis = []
        for bi in range(b):
            print('handling batch' + str(bi))
            mi = image_irr[bi, 0, :, :]
            mask_ir_bi = self.getmask(mi)
            mask_ir_bi = mask_ir_bi / 255.
            mask_ir_bi = mask_ir_bi.unsqueeze(0)
            mask_ir_bi = mask_ir_bi.unsqueeze(0)
            masks_bi.append(mask_ir_bi)
            masks_bi_vis.append(1 - mask_ir_bi)
        msk_ir = torch.cat(masks_bi, dim=0)
        msk_vis = torch.cat(masks_bi_vis, dim=0)
        msk_ir = msk_ir.unsqueeze(0)
        msk_vis = msk_vis.unsqueeze(0)
        msk = torch.cat([msk_ir, msk_vis], dim=0)
        msks = F.softmax(msk, dim=0)
        return msks[0], msks[1]


class SpatialAttention2(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention2, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out, _ = torch.max(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        return x


class spatial_attn_layer3(nn.Module):
    def __init__(self, kernel_size=1):
        super(spatial_attn_layer3, self).__init__()
        self.compress = ChannelPool2()
        self.spatial = BasicConv(3, 1, kernel_size, relu=False)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return scale


class Spatial_BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel=3, dilation=1, stride=1, reduction=64, with_norm=False):
        super(Spatial_BasicBlock, self).__init__()
        # self.with_norm = with_norm
        #
        # self.conv1 = conv3x3(inplanes, planes, stride)
        # self.conv2 = BasicConv(inplanes,inplanes,kernel,relu=False)
        self.se = spatial_attn_layer(kernel)
        # self.relu = nn.PReLU()
        # if self.with_norm:
        #   self.bn1 = nn.BatchNorm2d(planes)
        #   self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        # /
        out = self.se(x)
        # out += x
        # out = self.relu(out)
        return out


class ResidualDownSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(ResidualDownSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
                                 nn.PReLU(),
                                 nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=bias),
                                 nn.PReLU(),
                                 downsamp(channels=in_channels, filt_size=3, stride=2),
                                 nn.Conv2d(in_channels, in_channels * 2, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(downsamp(channels=in_channels, filt_size=3, stride=2),
                                 nn.Conv2d(in_channels, in_channels * 2, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top + bot
        return out


class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, stride=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(ResidualDownSample(in_channels))
            in_channels = int(in_channels * stride)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


class ResidualUpSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(ResidualUpSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=bias),
                                 nn.PReLU(),
                                 nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1,
                                                    bias=bias),
                                 nn.PReLU(),
                                 nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias),
                                 nn.Conv2d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top + bot
        return out


class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, stride=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(ResidualUpSample(in_channels))
            in_channels = int(in_channels // stride)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


# Attention Modules SEBlocks, ECABlocks

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def get_tv(tensor):
    N, C, H, W = tensor.size()
    f = tensor[:, :, :-1, :-1]
    g = tensor[:, :, :-1, 1:]
    h = tensor[:, :, 1:, :-1]
    tv_ = (f - g) ** 2. + (f - h) ** 2.
    return tv_


class SE_TVLayer(nn.Module):
    def __init__(self, channel, stride=1, affine=False, reduction=64):
        super(SE_TVLayer, self).__init__()
        self.avg_pool_1 = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, channel),
            nn.Sigmoid()
        )
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=1, stride=stride, padding=0, bias=False, )
        self.bn = nn.BatchNorm2d(channel, affine=affine)

    def forward(self, x):
        tvs = get_tv(x)
        b, c, _, _ = tvs.size()
        y = self.avg_pool_1(tvs).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = x * y
        return y


class TVBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, reduction=64, with_norm=False):
        super(TVBasicBlock, self).__init__()
        self.with_norm = with_norm

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv2 = conv3x3(planes, planes, 1)
        self.se = SE_TVLayer(planes, reduction)
        self.relu = nn.PReLU()
        if self.with_norm:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = x = self.conv1(x)
        if self.with_norm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.with_norm:
            out = self.bn2(out)
        out = self.se(out)
        out += x
        out = self.relu(out)
        return out


class eca_layer(nn.Module):
    """Constructs a ECA module.
  Args:
      channel: Number of channels of the input feature map
      k_size: Adaptive selection of kernel size
  """

    def __init__(self, channel, c_out, stride, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        y = x * y.expand_as(x)

        return y


class ECABasicBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel=3, dilation=1, stride=1, reduction=64, with_norm=False):
        super(ECABasicBlock, self).__init__()
        self.with_norm = with_norm

        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.conv2 = BasicConv(inplanes,inplanes,kernel,relu=False)
        self.se = eca_layer(planes, planes, stride, k_size=kernel)
        self.relu = nn.PReLU()

    def forward(self, x):
        out = self.se(x)
        out = self.conv1(out)
        out = self.relu(out)
        return out


# Fusion Modules
class SKFF(nn.Module):
    def __init__(self, in_channels, height=3, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dialtions=1, bias=False):
        super(ResidualDenseBlock, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = BasicConv(in_channels, in_channels, kernel_size, dilation=dialtions, relu=False,
                               groups=in_channels)
        self.conv2 = BasicConv(in_channels * 2, in_channels, kernel_size, dilation=dialtions, relu=False,
                               groups=in_channels)
        self.conv3 = BasicConv(in_channels * 3, in_channels, kernel_size, dilation=dialtions, relu=False,
                               groups=in_channels)

        self.lrelu = nn.PReLU()

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        return x3 * 0.333 + x


class ResidualModule(nn.Module):
    def __init__(self, in_channels, kernel_size, dialtions=1, bias=False):
        super(ResidualModule, self).__init__()
        self.op = nn.Sequential(
            BasicConv(in_channels, in_channels, kernel_size, dilation=dialtions, relu=False, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=2, dilation=2, groups=in_channels,
                      bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
        )

    def forward(self, x):
        res = self.op(x)
        return x + res


class EnhanceResidualModule(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(EnhanceResidualModule, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=1, padding=4, dilation=2, groups=in_channels,
                      bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=2, dilation=2, groups=in_channels,
                      bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=2, dilation=2, groups=in_channels,
                      bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
        )

    def forward(self, x):
        res = self.op(x)
        return x + res


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            BasicConv(C_in, C_out, kernel_size, dilation=dilation, relu=False, groups=C_in),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x) + x


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x




