from multiprocessing.sharedctypes import Value
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

class Reshape(nn.Module):
    def __init__(self, shape, keep_batch=True):
        super().__init__()
        self.keep_batch = keep_batch
        self.shape = shape
    
    def forward(self, x):
        new_shape = (x.shape[0], *self.shape) if self.keep_batch else self.shape
        return x.view(new_shape)


class kernel_gen(nn.Module):
    def __init__(self, channel, kernel_size, type='H', gen_type="FC", act="HardSigmoid", reduction=4):
        super().__init__()
        # kernel shape
        self.type = type
        kernel_size_2 = (kernel_size, 1) if type=='H' else (1, kernel_size)
        gen_kernel_size_2 = (3, 1) if type=='H' else (1, 3)
        gen_kernel_padding_2 = [(i-1)//2 for i in gen_kernel_size_2]
        # activation
        if act == "Hardsigmoid":
            activation = nn.Hardsigmoid()
        elif act == "Hardswish":
            activation = nn.Hardswish()
        # Gen block
        self.gen_type = gen_type
        if gen_type == "FC":
            self.gen = nn.Sequential(
                # B, C, H, 1 -> B, C*H
                Reshape(shape=(channel*kernel_size), keep_batch=True),
                # Main Module
                nn.Linear(kernel_size*channel, kernel_size*channel//reduction, bias=False),
                nn.BatchNorm1d(kernel_size*channel//reduction),
                activation,
                nn.Linear(kernel_size*channel//reduction, kernel_size*channel, bias=False),
                # B, C*H -> B, C, H, 1
                Reshape(shape=(channel, *kernel_size_2), keep_batch=True)
            )
        elif gen_type == "CONV":
            self.gen = nn.Sequential(
                nn.Conv2d(channel, channel//reduction, kernel_size=gen_kernel_size_2, padding=gen_kernel_padding_2, bias=False, groups=1),
                nn.BatchNorm2d(channel//reduction),
                activation,
                nn.Conv2d(channel//reduction, channel, kernel_size=gen_kernel_size_2, padding=gen_kernel_padding_2, bias=False, groups=1),
            )
        elif gen_type == "DW_CONV":
            self.gen = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=gen_kernel_size_2, padding=gen_kernel_padding_2, bias=False, groups=channel),
                nn.BatchNorm2d(channel),
                activation,
                nn.Conv2d(channel, channel, kernel_size=gen_kernel_size_2, padding=gen_kernel_padding_2, bias=False, groups=channel),
            )
        # GAP
        self.gap = nn.AdaptiveAvgPool2d(1)

    # Notice: ConvNext Outer Model will init these weights since they are normal FC nad Conv
    # def gcc_init(self):
    #     for name, module in self.gen.named_modules:
    #         if "weight" in name and len(module.weight.shape) == 4:
    #             trunc_normal_(module.weight, std=.02)

    def forward(self, x):
        glob_info = self.gap(x)
        if self.type=='H':
            H_info = torch.mean(x, dim=3, keepdim=True)
            x = H_info + glob_info.expand_as(H_info)
        elif self.type=='W':
            W_info = torch.mean(x, dim=2, keepdim=True)
            x = W_info + glob_info.expand_as(W_info)
        kernel_weight = self.gen(x)
        return kernel_weight
        

# Convnext like Blocks (trunc_normal weight init)
class dygcc_Conv2d(nn.Module):
    def __init__(self, dim, type, meta_kernel_size, instance_kernel_method=None, bias=True, use_pe=True):
        super().__init__()
        # super(gcc_Conv2d, self).__init__()
        self.type = type    # H or W
        self.dim = dim
        self.instance_kernel_method = instance_kernel_method
        self.meta_kernel_size_2 = (meta_kernel_size, 1) if self.type=='H' else (1, meta_kernel_size)
        self.weight_gen = kernel_gen(dim, meta_kernel_size, type=type, gen_type='DW_CONV', act="Hardswish", reduction=1)
        # self.weight  = nn.Conv2d(dim, dim, kernel_size=self.meta_kernel_size_2, groups=dim).weight
        self.bias    = nn.Parameter(torch.randn(dim)) if bias else None
        self.meta_pe = nn.Parameter(torch.randn(1, dim, *self.meta_kernel_size_2)) if use_pe else None

    def gcc_init(self):
        # trunc_normal_(self.weight, std=.02)
        # self.weight_gen.gcc_init()
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if self.meta_pe is not None:
            trunc_normal_(self.meta_pe, std=.02)

    def get_instance_pe(self, instance_kernel_size_2):
        # if no use of dynamic resolution, keep a static kernel
        if self.instance_kernel_method is None:
            return  self.meta_pe
        elif self.instance_kernel_method == 'interpolation_bilinear':
            return  F.interpolate(self.meta_pe, instance_kernel_size_2, mode='bilinear', align_corners=True)\
                        .expand(1, self.dim, *instance_kernel_size_2)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.meta_pe is not None:
            x = x + self.get_instance_pe((H, W))
        # weight = self.get_instance_kernel((H, W))
        weight = self.weight_gen(x).view(B*C, 1, H, 1) if self.type=='H' else self.weight_gen(x).view(B*C, 1, 1, W)

        x = x.view(1, B*C, H, W)
        x_cat = torch.cat((x, x[:, :, :-1, :]), dim=2) if self.type=='H' else torch.cat((x, x[:, :, :, :-1]), dim=3)
        x = F.conv2d(x_cat, weight=weight, bias=None, padding=0, groups=self.dim*B).view(B, C, H, W)
        if self.bias is not None:
            x = x + self.bias.view(1, C, 1, 1)
        return x

class dygcc_cvx_Block(nn.Module):
    def __init__(self,
        dim,
        drop_path=0.,
        layer_scale_init_value=1e-6,
        meta_kernel_size=16,
        instance_kernel_method=None,
        use_pe=True
    ):
        super().__init__()
        # super(gcc_cvx_Block, self).__init__()
        # branch1
        self.gcc_conv_1H = dygcc_Conv2d(dim//2, type='H', meta_kernel_size=meta_kernel_size,
            instance_kernel_method=instance_kernel_method, use_pe=use_pe) 
        # branch2
        self.gcc_conv_2W = dygcc_Conv2d(dim//2, type='W', meta_kernel_size=meta_kernel_size,
            instance_kernel_method=instance_kernel_method, use_pe=use_pe)
        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x_1, x_2 = torch.chunk(x, 2, dim=1)
        x_1, x_2 = self.gcc_conv_1H(x_1), self.gcc_conv_2W(x_2)
        x = torch.cat((x_1, x_2), dim=1)
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
        # super(Block, self).__init__()
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