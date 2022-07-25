import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


class GCHW_conv(nn.Module):
    """
    global convolution
    kernel is split into H and W two directions
    """
    def __init__(self, dim, global_kernel_size):
        super().__init__()
        self.dim = dim // 2
        self.gks = global_kernel_size
        self.gch_conv = nn.Conv2d(self.dim, self.dim, kernel_size=(self.gks, 1), padding=(self.gks // 2, 0), groups=self.dim)
        self.gcw_conv = nn.Conv2d(self.dim, self.dim, kernel_size=(1, self.gks), padding=(0, self.gks // 2), groups=self.dim)

    def forward(self, x):
        x_H, x_W = torch.chunk(x, 2, dim=1)
        x_H, x_W = self.gch_conv(x_H), self.gcw_conv(x_W)
        x = torch.cat((x_H, x_W), dim=1)

        return x


class ConvNext_GC_Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, global_kernel_size=14):
        super().__init__()
        self.gc_conv = GCHW_conv(dim, global_kernel_size)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.gc_conv(x)
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



class ConvNext_Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
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


class ConvNeXt_gc(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 gcc_bs_indices=[3, 3, 6, 2], stages_rs=[56, 28, 13, 7] # gcc block start indices, input resolutions of four stages
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i, [depth, dim, gcc_bs_index, input_size] in enumerate(zip(depths, dims, gcc_bs_indices, stages_rs)):
            blocks = []
            for j in range(depth):
                if j<gcc_bs_index:
                    blocks.append(ConvNext_Block(dim=dim, drop_path=dp_rates[cur + j],
                                                 layer_scale_init_value=layer_scale_init_value))
                else:
                    blocks.append(ConvNext_GC_Block(dim=dim, drop_path=dp_rates[cur + j],
                                                     layer_scale_init_value=layer_scale_init_value,
                                                     global_kernel_size=input_size))
                stage = nn.Sequential(*blocks)
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def get_model_size(self):
        return sum([p.numel() for p in self.parameters()])


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

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


@register_model
def convnext_gc_xt(pretrained=False, in_22k=False, **kwargs):
    """
    Constructs a mixed model, where 7*7 depth wise conv operations in the last 1/3 blocks of the last two stages are
    replaced with position aware global conv (GC) operations.
    Args:
        gcc_bs_indices [s1, s2, s3, s4]: gcc_bs_indices[i] indicates that blocks from gcc_bs_indices[i] to end are
        replaced in stage i.
    """

    model = ConvNeXt_gc(depths=[3, 3, 9, 3], dims=[48, 96, 192, 384], gcc_bs_indices=[3, 3, 6, 2], stages_rs=[56, 28, 13, 7], **kwargs)
    if pretrained:
        raise NotImplementedError("no pretrained model")
    # test
    input = torch.randn(2, 3, 224, 224)
    out = model(input)
    return model

