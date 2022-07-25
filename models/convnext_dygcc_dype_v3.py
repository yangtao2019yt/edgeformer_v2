import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class gen(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        # Gen block
        self.gen = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, kernel_size=3, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(channel//reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//reduction, channel, kernel_size=3, padding=1, groups=1),
        )

    def forward(self, x):
        return self.gen(x)

class dygcc_dype_v3_Block(nn.Module):
    def __init__(self,
        dim,
        drop_path=0.,
        layer_scale_init_value=1e-6,
        meta_kernel_size=16,
        instance_kernel_method=None,
        use_pe=True
    ):
        reduction = 16

        super().__init__()
        self.use_pe = use_pe
        self.dim = dim
        self.pe_gen = gen(dim, reduction=reduction) if use_pe else None
        self.kernel_gen = gen(dim, reduction=reduction)
        self.bias = nn.Parameter(torch.zeros(dim)*1.)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        B, C, H, W = x.shape
        
        # generate pe and weights
        if self.use_pe:
            HW_pe = self.pe_gen(x)
            x = x + HW_pe
        H_weight, W_weight = torch.chunk(self.kernel_gen(x), 2, dim=1)
        H_weight, W_weight = H_weight.mean(dim=3), W_weight.mean(dim=2)
    
        # token mixer
        x_1, x_2 = torch.chunk(x, 2, dim=1)
        # branch 1 - GCC-H
        x_1, H_weight = x_1.reshape(1, B*C//2, H, W), H_weight.view(B*C//2, 1, H, 1)
        x_1 = torch.cat((x_1, x_1[:, :, :-1, :]), dim=2)
        x_1 = F.conv2d(x_1, weight=H_weight, bias=None, padding=0, groups=B*C//2).view(B, C//2, H, W)
        # branch 2 - GCC-W
        x_2, W_weight = x_2.reshape(1, B*C//2, H, W), W_weight.view(B*C//2, 1, 1, W)
        x_2 = torch.cat((x_2, x_2[:, :, :, :-1]), dim=3)
        x_2 = F.conv2d(x_2, weight=W_weight, bias=None, padding=0, groups=B*C//2).view(B, C//2, H, W)
        # fusion
        x = torch.cat((x_1, x_2), dim=1)
        x = x + self.bias.view(1, C, 1, 1)

        # channel mixer
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        # super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvNeXt_dygcc_dype_v3(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        stages_fs = [56, 28, 14, 7]
        for i in range(4):
            if i < 2:   # for stage 0 and 1, no gcc
                stage = nn.Sequential(*[
                    Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) \
                    for j in range(depths[i])
                ])
            else:       # for stage 2 and 3, gcc modules is used
                # e.g. in stage3, j+1=7 > lo=2*9//3=6, so block 678 is gcc_block, while block 0-5 is normal
                stage = nn.Sequential(*[
                    dygcc_dype_v3_Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value,
                        meta_kernel_size=stages_fs[i], instance_kernel_method=None, \
                        use_pe=(((i==2)&(j==7))|((i==3)&(j==2)))
                    )\
                    if 2*depths[i]//3 < j+1 else \
                    Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) \
                    for j in range(depths[i])
                ])
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

@register_model
def convnext_dygcc_dype_v3_tt(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt_dygcc_dype_v3(depths=[3, 3, 9, 3], dims=[48, 96, 192, 384], **kwargs)
    if pretrained or in_22k:
        raise NotImplementedError("no pretrained model")
    return model