import torch
from torch import nn
from einops import rearrange
import numbers
import math
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    # 修正：删除冗余的 c1, c2 参数，仅保留 no_spatial
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not self.no_spatial:
            self.hw = AttentionGate()

    # forward 函数完全不变（注意力逻辑本身适配任意输入通道）
    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')
def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# class Restormer_Decoder(nn.Module):
#     def __init__(self, dim=128, out_channels=1):
#         super().__init__()
#         # 压缩层：处理两种场景的通道拼接
#         self.reduce_ir = nn.Conv2d(192, dim, kernel_size=1)  # 3个64通道特征→192→128
#         self.reduce_all = nn.Conv2d(384, dim, kernel_size=1)  # 6个64通道特征→384→128
#
#         # 增加Transformer Block层数至2层，增强特征融合能力
#         self.encoder_level2 = nn.Sequential(
#             # 第一层Transformer Block
#             TransformerBlock(dim=dim, num_heads=8, ffn_expansion_factor=2,
#                            bias=False, LayerNorm_type='WithBias'),
#             # 新增第二层Transformer Block
#             TransformerBlock(dim=dim, num_heads=8, ffn_expansion_factor=2,
#                            bias=False, LayerNorm_type='WithBias'),
#         )
#
#         # 输出层（确保输出1通道）
#         self.output = nn.Sequential(
#             Conv(dim, dim // 2, k=3),
#             Conv(dim // 2, out_channels, k=3, act=False)  # 输出1通道
#         )
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, enc1, feat1, feat2, enc2=None, feat3=None, feat4=None):
#         """
#         兼容两种输入场景：
#         1. 仅处理单模态特征（如红外）：enc1=ir_encoder, feat1=ir_B, feat2=ir_D（enc2/feat3/feat4为None）
#         2. 处理双模态融合：enc1=ir_encoder, feat1=ir_B, feat2=ir_D, enc2=vi_encoder, feat3=vi_B, feat4=vi_D
#         """
#         if enc2 is None and feat3 is None and feat4 is None:
#             # 场景1：仅单模态特征（3个特征，64*3=192通道）
#             x = torch.cat([enc1, feat1, feat2], dim=1)  # [B, 192, H, W]
#             x = self.reduce_ir(x)  # 压缩至128通道
#         else:
#             # 场景2：双模态融合（6个特征，64*6=384通道）
#             x = torch.cat([enc1, feat1, feat2, enc2, feat3, feat4], dim=1)  # [B, 384, H, W]
#             x = self.reduce_all(x)  # 压缩至128通道
#
#         # 经过多层Transformer Block处理
#         x = self.encoder_level2(x)
#         x = self.output(x)  # 输出1通道
#         return self.sigmoid(x)

class Restormer_Decoder(nn.Module):
    def __init__(self, dim=128, out_channels=1):
        super().__init__()
        # 压缩层：新增"双分支4特征"的通道处理（保持原有压缩层，新增适配4特征的层）
        self.reduce_ir = nn.Conv2d(192, dim, kernel_size=1)  # 单模态3特征（64*3=192）→128
        self.reduce_all = nn.Conv2d(384, dim, kernel_size=1)  # 双模态6特征（64*6=384）→128
        self.reduce_4feat = nn.Conv2d(256, dim, kernel_size=1)  # 双分支4特征（64*4=256）→128（新增）

        # 原有Transformer Block结构保持不变
        self.encoder_level2 = nn.Sequential(
            TransformerBlock(dim=dim, num_heads=8, ffn_expansion_factor=2,
                           bias=False, LayerNorm_type='WithBias'),
            TransformerBlock(dim=dim, num_heads=8, ffn_expansion_factor=2,
                           bias=False, LayerNorm_type='WithBias'),
        )

        # 输出层保持不变
        self.output = nn.Sequential(
            Conv(dim, dim // 2, k=3),
            Conv(dim // 2, out_channels, k=3, act=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, x3, x4=None, x5=None, x6=None):
        """
        兼容三种输入场景（通过参数数量区分）：
        1. 单模态3特征：x1=enc1, x2=feat1, x3=feat2（x4/x5/x6=None）→ 对应原场景1
        2. 双模态6特征：x1=enc1, x2=feat1, x3=feat2, x4=enc2, x5=feat3, x6=feat4 → 对应原场景2
        3. 双分支4特征：x1=ir_B, x2=vi_B, x3=ir_D, x4=vi_D（x5/x6=None）→ 新增场景
        """
        if x4 is None and x5 is None and x6 is None:
            # 场景1：单模态3特征（原逻辑不变）
            x = torch.cat([x1, x2, x3], dim=1)  # [B, 192, H, W]
            x = self.reduce_ir(x)
        elif x5 is None and x6 is None:
            # 场景3：双分支4特征（新增逻辑）
            # x1=ir_B, x2=vi_B, x3=ir_D, x4=vi_D（假设每个特征64通道）
            x = torch.cat([x1, x2, x3, x4], dim=1)  # [B, 64*4=256, H, W]
            x = self.reduce_4feat(x)  # 压缩至128通道
        else:
            # 场景2：双模态6特征（原逻辑不变）
            x = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)  # [B, 384, H, W]
            x = self.reduce_all(x)

        # 后续处理保持不变
        x = self.encoder_level2(x)
        x = self.output(x)
        return self.sigmoid(x)

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class Encoder(nn.Module):
    """原始Encoder模块，保留用于对比实验"""
    def __init__(self):
        super().__init__()
        self.con = Conv(1, 64)
        self.c3 = C3(64, 64)

    def forward(self, x):
        return self.c3(self.con(x))


############## ConvNext 模块（适配单通道输入）##############
class LayerNorm_s(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNextBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # 深度卷积，适配输入dim通道
        self.norm = LayerNorm_s(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class CNeB(nn.Module):
    """修正CNeB：支持自定义输入通道数，适配单通道原始图像"""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # 隐藏层通道数
        self.cv1 = Conv(c1, c_, 1, 1)  # 输入通道c1可自定义（1或64）
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(ConvNextBlock(c_) for _ in range(n)))  # ConvNextBlock适配c_通道

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
############## ConvNext 模块结束 ##############


class Base(nn.Module):
    """修正Base：新增in_channels参数，支持1通道（原始图像）或64通道（Encoder输出）输入"""
    def __init__(self, in_channels=64):  # 默认64通道，适配原始代码；消融实验传1通道
        super().__init__()
        # 调用修正后的CNeB，输入通道为in_channels，输出64通道（与双分支特征维度一致）
        self.convNext = CNeB(c1=in_channels, c2=64)

    def forward(self, x):
        return self.convNext(x)


class Detail(nn.Module):
    """修正Detail：TripletAttention无冗余参数，适配任意输入通道"""
    def __init__(self):
        super().__init__()
        # 修正：删除 c1=None, c2=None，仅传 no_spatial=False（与TripletAttention新定义匹配）
        self.TAM = TripletAttention(no_spatial=False)

    def forward(self, x):
        # 注意力逻辑动态适配输入通道（1通道/64通道均支持）
        return self.TAM(x)


# 测试代码（验证模块兼容性）
if __name__ == '__main__':
    # 测试：1通道原始图像输入Detail
    import torch

    detail = Detail()
    ir_img = torch.randn(1, 1, 128, 128)  # 消融实验的1通道红外图像
    ir_D = detail(ir_img)
    print("Detail输出形状:", ir_D.shape)  # 预期：torch.Size([1, 1, 128, 128])（通道数不变）