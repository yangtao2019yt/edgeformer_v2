import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

# Convnext like Blocks (trunc_normal weight init)
class gcc_Conv2d(nn.Module):
    def __init__(self, dim, type, meta_kernel_size, instance_kernel_method=None, bias=True, use_pe=True):
        super().__init__()
        self.type = type    # H or W
        self.dim = dim
        self.instance_kernel_method = instance_kernel_method
        self.meta_kernel_size_2 = (meta_kernel_size, 1) if self.type=='H' else (1, meta_kernel_size)
        self.weight  = nn.Conv2d(dim, dim, kernel_size=self.meta_kernel_size_2, groups=dim).weight
        self.bias    = nn.Parameter(torch.randn(dim)) if bias else None
        self.meta_pe = nn.Parameter(torch.randn(1, dim, *self.meta_kernel_size_2)) if use_pe else None

    def gcc_init(self):
        trunc_normal_(self.weight, std=.02)
        nn.init.constant_(self.bias, 0)
        if self.use_pe:
            trunc_normal_(self.meta_pe, std=.02)

    def get_instance_kernel(self, instance_kernel_size_2):
        # if no use of dynamic resolution, keep a static kernel
        if self.instance_kernel_method is None:
            return  self.weight
        elif self.instance_kernel_method == 'interpolation_bilinear':
            instance_kernel_size_2 =  (instance_kernel_size_2[0], 1) if self.type=='H' else (1, instance_kernel_size_2[1])
            return  F.interpolate(self.weight, instance_kernel_size_2, mode='bilinear', align_corners=True)
        
    def get_instance_pe(self, instance_kernel_size_2):
        # if no use of dynamic resolution, keep a static kernel
        if self.instance_kernel_method is None:
            return  self.meta_pe
        elif self.instance_kernel_method == 'interpolation_bilinear':
            return  F.interpolate(self.meta_pe, instance_kernel_size_2, mode='bilinear', align_corners=True)\
                        .expand(1, self.dim, *instance_kernel_size_2)

    def forward(self, x):
        _, _, H, W = x.shape
        if self.meta_pe is not None:
            x = x + self.get_instance_pe((H, W))
        weight = self.get_instance_kernel((H, W))
        x_cat = torch.cat((x, x[:, :, :-1, :]), dim=2) if self.type=='H' else torch.cat((x, x[:, :, :, :-1]), dim=3)
        x = F.conv2d(x_cat, weight=weight, bias=self.bias, padding=0, groups=self.dim)
        return x

class gcc_cvx_lg_Block(nn.Module):
    def __init__(self,
        dim,
        drop_path=0.,
        layer_scale_init_value=1e-6,
        meta_kernel_size=16,
        instance_kernel_method=None,
        use_pe=True
    ):
        super().__init__()
        # local part
        self.dwconv = nn.Conv2d(dim//2, dim//2, kernel_size=7, padding=3, groups=dim//2) # depthwise conv
        # global part
        # branch1
        self.gcc_conv_1H = gcc_Conv2d(dim//4, type='H', meta_kernel_size=meta_kernel_size,
            instance_kernel_method=instance_kernel_method, use_pe=use_pe) 
        # branch2
        self.gcc_conv_2W = gcc_Conv2d(dim//4, type='W', meta_kernel_size=meta_kernel_size,
            instance_kernel_method=instance_kernel_method, use_pe=use_pe)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x_global, x_local = torch.chunk(x, 2, 1)
        # local part
        x_local = self.dwconv(x_local)
        # global part
        x_1, x_2 = torch.chunk(x_global, 2, 1)
        x_1, x_2 = self.gcc_conv_1H(x_1), self.gcc_conv_2W(x_2)
        x_global = torch.cat((x_1, x_2), dim=1)
        # global & local fusion
        x = torch.cat((x_global, x_local), dim=1)
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

class gcc_cvx_lg_Block_2stage(nn.Module):
    def __init__(self,
        dim,
        drop_path=0.,
        layer_scale_init_value=1e-6,
        meta_kernel_size=16,
        local_kernel_size=7,
        instance_kernel_method=None,
        use_pe=True
    ):
        super().__init__()
        # local part
        self.dwconv = nn.Conv2d(dim//2, dim//2, kernel_size=local_kernel_size, padding=3, groups=dim//2) # depthwise conv
        # global part
        # branch1   ..->H->W->..
        self.gcc_conv_1H = gcc_Conv2d(dim//4, type='H', meta_kernel_size=meta_kernel_size,
            instance_kernel_method=instance_kernel_method, use_pe=use_pe) 
        self.gcc_conv_1W = gcc_Conv2d(dim//4, type='W', meta_kernel_size=meta_kernel_size,
            instance_kernel_method=instance_kernel_method, use_pe=use_pe)
        # branch2   ..->W->H->..
        self.gcc_conv_2W = gcc_Conv2d(dim//4, type='W', meta_kernel_size=meta_kernel_size,
            instance_kernel_method=instance_kernel_method, use_pe=use_pe)
        self.gcc_conv_2H = gcc_Conv2d(dim//4, type='H', meta_kernel_size=meta_kernel_size,
            instance_kernel_method=instance_kernel_method, use_pe=use_pe) 
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x_global, x_local = torch.chunk(x, 2, dim=1)
        # local part
        x_local = self.dwconv(x_local)
        # global part
        x_1, x_2 = torch.chunk(x_global, 2, dim=1)
        # branch1   ..->H->W->..
        x_1 = self.gcc_conv_1W(self.gcc_conv_1H(x_1))
        # branch2   ..->W->H->..
        x_2 = self.gcc_conv_2H(self.gcc_conv_2W(x_2))
        # global branch12 fusion
        x_global = torch.cat((x_1, x_2), dim=1)
        # global & local fusion
        x = torch.cat((x_global, x_local), dim=1)
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

# v1, top1-accu 77.8
class gcc_cvx_lg_Block_2stage_res_v1(nn.Module):
    def __init__(self,
        dim,
        drop_path=0.,
        layer_scale_init_value=1e-6,
        meta_kernel_size=16,
        local_kernel_size=7,
        instance_kernel_method=None,
        use_pe=True
    ):
        super().__init__()

        # Token Mixer
        self.norm_t = LayerNorm(dim, eps=1e-6)
        # local part
        self.dwconv = nn.Conv2d(dim//2, dim//2, kernel_size=local_kernel_size, padding=3, groups=dim//2) # depthwise conv
        # global part
        # branch1   ..->H->W->..
        self.gcc_conv_1H = gcc_Conv2d(dim//2, type='H', meta_kernel_size=meta_kernel_size,
            instance_kernel_method=instance_kernel_method, use_pe=use_pe) 
        self.gcc_conv_1W = gcc_Conv2d(dim//2, type='W', meta_kernel_size=meta_kernel_size,
            instance_kernel_method=instance_kernel_method, use_pe=use_pe)

        # Channel Mixer
        self.norm_c = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # Others
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path_t = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_c = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # Token Mixer
        input = x
        x = self.norm_t(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous() # pre-norm
        x_global, x_local = torch.chunk(x, 2, dim=1)
        # local part
        x_local = self.dwconv(x_local)
        # global part, 1 branch ..->H->W->..
        x_global = self.gcc_conv_1W(self.gcc_conv_1H(x_global))
        # global & local fusion
        x = torch.cat((x_global, x_local), dim=1)
        x = input + self.drop_path_t(x)
        
        # Channel Mixer
        input = x
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm_c(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path_c(x)
        return x

# v2, a single PE before all modules
class gcc_cvx_lg_Block_2stage_res_v2(nn.Module):
    def __init__(self,
        dim,
        drop_path=0.,
        layer_scale_init_value=1e-6,
        meta_kernel_size=16,
        local_kernel_size=7,
        instance_kernel_method=None,
        use_pe=True
    ):
        super().__init__()

        # in v2, we use a single PE before all modules
        self.dim = dim
        self.use_pe = use_pe
        self.instance_kernel_method = instance_kernel_method
        self.meta_pe = nn.Parameter(torch.randn(1, dim, meta_kernel_size, meta_kernel_size)) if use_pe else None

        # Token Mixer
        self.norm_t = LayerNorm(dim, eps=1e-6)
        # local part
        self.dwconv = nn.Conv2d(dim//2, dim//2, kernel_size=local_kernel_size, padding=3, groups=dim//2) # depthwise conv
        # global part
        # branch1   ..->H->W->..
        self.gcc_conv_1H = gcc_Conv2d(dim//2, type='H', meta_kernel_size=meta_kernel_size,
            instance_kernel_method=instance_kernel_method, use_pe=False) # no pe within gcc
        self.gcc_conv_1W = gcc_Conv2d(dim//2, type='W', meta_kernel_size=meta_kernel_size,
            instance_kernel_method=instance_kernel_method, use_pe=False) # no pe within gcc
        self.gamma_t = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        # Channel Mixer
        self.norm_c = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma_c = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        # Others
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def get_instance_pe(self, instance_kernel_size): # get_instance_pe method buildt for the whole block, differ from PE-within-GCC
        # if no use of dynamic resolution, keep a static kernel
        if self.instance_kernel_method is None:
            return  self.meta_pe
        elif self.instance_kernel_method == 'interpolation_bilinear':
            instance_kernel_size_2 =  (instance_kernel_size, instance_kernel_size)
            return  F.interpolate(self.meta_pe, instance_kernel_size_2, mode='bilinear', align_corners=True)\
                        .expand(1, self.dim, instance_kernel_size, instance_kernel_size)

    def forward(self, x):
        # in v2, we use a single PE before all modules
        if self.use_pe:
            _, _, fs, _ = x.shape
            x = x + self.get_instance_pe(fs)
        
        # Token Mixer
        input = x
        x = self.norm_t(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous() # pre-norm
        x_global, x_local = torch.chunk(x, 2, dim=1)
        # local part
        x_local = self.dwconv(x_local)
        # global part, 1 branch ..->H->W->..
        x_global = self.gcc_conv_1W(self.gcc_conv_1H(x_global))
        # global & local fusion
        x = torch.cat((x_global, x_local), dim=1)
        if self.gamma_t is not None:
            x = self.gamma_t.unsqueeze(-1).unsqueeze(-1) * x  # (N, C, H, W) * (C, 1, 1)
        x = input + self.drop_path(x)
        
        # Channel Mixer
        input = x
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm_c(x) # pre-norm
        # ffn
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma_c is not None:
            x = self.gamma_c * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

# v3, use CAT for fusion instead of ADD
class gcc_cvx_lg_Block_2stage_res_v3(nn.Module):
    def __init__(self,
        dim,
        drop_path=0.,
        layer_scale_init_value=1e-6,
        meta_kernel_size=16,
        instance_kernel_method=None,
        use_pe=True
    ):
        super().__init__()
        # local part
        self.dwconv = nn.Conv2d(dim//2, dim//2, kernel_size=7, padding=3, groups=dim//2) # depthwise conv
        # global part
        # branch1
        self.gcc_conv_1H = gcc_Conv2d(dim//2, type='H', meta_kernel_size=meta_kernel_size,
            instance_kernel_method=instance_kernel_method, use_pe=use_pe) 
        self.gcc_conv_1W = gcc_Conv2d(dim//2, type='W', meta_kernel_size=meta_kernel_size,
            instance_kernel_method=instance_kernel_method, use_pe=use_pe)
        self.norm = LayerNorm(2 * dim, eps=1e-6)   # in v3, 2*dim -> 4*dim -> dim
        self.pwconv1 = nn.Linear(2 * dim, 4 * dim) # in v3, 2*dim -> 4*dim -> dim
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        # Token Mixer
        x_global, x_local = torch.chunk(x, 2, 1)
        # local part
        x_local = self.dwconv(x_local)
        # global part
        x_global = self.gcc_conv_1W(self.gcc_conv_1H(x_global))
        # global & local fusion
        x = torch.cat((x_global, x_local), dim=1)
        # v3, instead of ADD, a CAT operation is used for this skip connection
        x = torch.cat((x, input), dim=1)

        # Channel Mixer
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

# v4
class gcc_cvx_lg_Block_2stage_res_v4(nn.Module):
    def __init__(self,
        dim,
        drop_path=0.,
        layer_scale_init_value=1e-6, # not used in v4
        meta_kernel_size=16,
        instance_kernel_method=None,
        use_pe=True
    ):
        super().__init__()

        # Token Mixer
        # local part, in v4 this part takes up 1/3 of channels
        self.dwconv = nn.Conv2d(dim//3, dim//3, kernel_size=7, padding=3, groups=dim//3) # depthwise conv
        # global part, in v4 this part takes up 2/3 of channels
        # branch1   ..->H->W->..
        self.gcc_conv_1H = gcc_Conv2d(dim//3, type='H', meta_kernel_size=meta_kernel_size,
            instance_kernel_method=instance_kernel_method, use_pe=use_pe) 
        self.gcc_conv_1W = gcc_Conv2d(dim//3, type='W', meta_kernel_size=meta_kernel_size,
            instance_kernel_method=instance_kernel_method, use_pe=use_pe)
        # branch2   ..->W->H->..
        self.gcc_conv_2W = gcc_Conv2d(dim//3, type='W', meta_kernel_size=meta_kernel_size,
            instance_kernel_method=instance_kernel_method, use_pe=use_pe)
        self.gcc_conv_2H = gcc_Conv2d(dim//3, type='H', meta_kernel_size=meta_kernel_size,
            instance_kernel_method=instance_kernel_method, use_pe=use_pe)
        self.norm_t = nn.BatchNorm2d(dim)      # in v4 we use BN instead of LN
        
        # Channel Mixer
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.norm_c = nn.BatchNorm2d(dim)      # in v4 we use BN instead of LN
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # Token Mixer
        input = x
        x_global1, x_global2, x_local = torch.chunk(x, 3, dim=1)
        # local part
        x_local = self.dwconv(x_local)
        # global part
        # branch1   ..->H->W->..
        x_global1 = self.gcc_conv_1W(self.gcc_conv_1H(x_global1))
        # branch2   ..->W->H->..
        x_global2 = self.gcc_conv_2H(self.gcc_conv_2W(x_global2))
        # global & local fusion
        x = torch.cat((x_global1, x_global2, x_local), dim=1)
        x = self.norm_t(x)
        x = input + self.drop_path(x)

        # Channel Mixer
        input = x
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # keep N x C x H x W format before BN layer
        x = self.norm_c(x)
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