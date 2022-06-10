from multiprocessing.sharedctypes import Value
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

# Convnext like Blocks (trunc_normal weight init)
class gcc_Conv2d(nn.Module):
    def __init__(self, dim, type, meta_kernel_size, instance_kernel_method=None, bias=True, use_pe=True):
        super().__init__()
        # super(gcc_Conv2d, self).__init__()
        self.type = type    # H or W
        self.dim = dim
        self.instance_kernel_method = instance_kernel_method
        self.meta_kernel_size_2 = (meta_kernel_size, 1) if self.type=='H' else (1, meta_kernel_size)
        self.weight  = nn.Conv2d(dim, dim, kernel_size=self.meta_kernel_size_2, groups=dim).weight
        self.bias    = nn.Parameter(torch.randn(dim)) if bias else None
        self.meta_pe = nn.Parameter(torch.randn(1, dim, *self.meta_kernel_size_2)) if use_pe else None

    def gcc_init(self):
        trunc_normal_(self.weight, std=.02)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if self.meta_pe is not None:
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

class gcc_cvx_Block(nn.Module):
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
        self.gcc_conv_1H = gcc_Conv2d(dim//2, type='H', meta_kernel_size=meta_kernel_size,
            instance_kernel_method=instance_kernel_method, use_pe=use_pe) 
        # branch2
        self.gcc_conv_2W = gcc_Conv2d(dim//2, type='W', meta_kernel_size=meta_kernel_size,
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

class gcc_cvx_Block_2stage(nn.Module):
    def __init__(
        self,
        dim,
        drop_path=0.,
        layer_scale_init_value=1e-6,
        meta_kernel_size=16,
        instance_kernel_method=None,
        use_pe=True
    ):
        super().__init__()
        # super(gcc_cvx_Block_2stage, self).__init__()
        # branch1   ..->H->W->..
        self.gcc_conv_1H = gcc_Conv2d(dim//2, type='H', meta_kernel_size=meta_kernel_size,
            instance_kernel_method=instance_kernel_method, use_pe=use_pe) 
        self.gcc_conv_1W = gcc_Conv2d(dim//2, type='W', meta_kernel_size=meta_kernel_size,
            instance_kernel_method=instance_kernel_method, use_pe=use_pe)
        # branch2   ..->W->H->..
        self.gcc_conv_2W = gcc_Conv2d(dim//2, type='W', meta_kernel_size=meta_kernel_size,
            instance_kernel_method=instance_kernel_method, use_pe=use_pe)
        self.gcc_conv_2H = gcc_Conv2d(dim//2, type='H', meta_kernel_size=meta_kernel_size,
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
        # branch1   ..->H->W->..
        x_1 = self.gcc_conv_1W(self.gcc_conv_1H(x_1))
        # branch2   ..->W->H->..
        x_2 = self.gcc_conv_2H(self.gcc_conv_2W(x_2))
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

# v1 is MetaFormer test
class gcc_cvx_test_Block_v1(nn.Module):
    def __init__(self,
        dim,
        drop_path=0.,
        layer_scale_init_value=1e-6,
        meta_kernel_size=16,
        instance_kernel_method=None,
        use_pe=True
    ):
        super().__init__()

        # Token Mixer
        self.norm_t = LayerNorm(dim, eps=1e-6)
        # global part, branch1 GCC-H, branch2 GCC-W
        self.gcc_conv_1H = gcc_Conv2d(dim//2, type='H', meta_kernel_size=meta_kernel_size,
            instance_kernel_method=instance_kernel_method, use_pe=use_pe) 
        self.gcc_conv_2W = gcc_Conv2d(dim//2, type='W', meta_kernel_size=meta_kernel_size,
            instance_kernel_method=instance_kernel_method, use_pe=use_pe)
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

    def forward(self, x):
        # Token Mixer
        input = x
        x = self.norm_t(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous() # pre-norm
        x_1, x_2 = torch.chunk(x, 2, dim=1)
        # branch1 GCC-H, branch2 GCC-W
        x_1, x_2 = self.gcc_conv_1H(x_1), self.gcc_conv_2W(x_2)
        x = torch.cat((x_1, x_2), dim=1)
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

# v2 is local & global test, group-norm
class gcc_cvx_test_Block_v2(nn.Module):
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

        # local part, takes 1/3 in v2
        self.dwconv = nn.Conv2d(dim//3, dim//3, kernel_size=local_kernel_size, padding=3, groups=dim//3) # depthwise conv
        self.norm_local = LayerNorm(dim//3, eps=1e-6) # in v2, use grouped Norm
        # global part, branch1 GCC-H, branch2 GCC-W, takes 2/3 in v2
        self.gcc_conv_1H = gcc_Conv2d(dim//3, type='H', meta_kernel_size=meta_kernel_size,
            instance_kernel_method=instance_kernel_method, use_pe=use_pe) 
        self.gcc_conv_2W = gcc_Conv2d(dim//3, type='W', meta_kernel_size=meta_kernel_size,
            instance_kernel_method=instance_kernel_method, use_pe=use_pe)
        self.norm_global = LayerNorm(2*dim//3, eps=1e-6) # in v2, use grouped Norm
        
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                            requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        x_global1, x_global2, x_local = torch.chunk(x, 3, dim=1)
        
        # local part
        x_local = self.dwconv(x_local)
        x_local = self.norm_local(x_local.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # local-norm
        
        # global part, branch1 GCC-H, branch2 GCC-W, takes 2/3 in v2
        x_global1, x_global2 = self.gcc_conv_1H(x_global1), self.gcc_conv_2W(x_global2)
        # global fusion
        x_global = torch.cat((x_global1, x_global2), dim=1)
        x_global = self.norm_global(x_global.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # global-norm

        # local & global fusion
        x = torch.cat((x_global, x_local), dim=1)

        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

# v3 is local & global test, group-norm + group-fc
class gcc_cvx_test_Block_v3(nn.Module):
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

        # local part, takes 1/3 in v3
        self.dwconv = nn.Conv2d(dim//3, dim//3, kernel_size=local_kernel_size, padding=3, groups=dim//3) # depthwise conv
        self.norm_local = LayerNorm(dim//3, eps=1e-6) # in v3, use grouped Norm
        self.pwconv1_local = nn.Linear(dim//3, 4 * dim//3) # in v3, use grouped FC

        # global part, branch1 GCC-H, branch2 GCC-W, takes 2/3 in v3
        self.gcc_conv_1H = gcc_Conv2d(dim//3, type='H', meta_kernel_size=meta_kernel_size,
            instance_kernel_method=instance_kernel_method, use_pe=use_pe) 
        self.gcc_conv_2W = gcc_Conv2d(dim//3, type='W', meta_kernel_size=meta_kernel_size,
            instance_kernel_method=instance_kernel_method, use_pe=use_pe)
        self.norm_global = LayerNorm(2*dim//3, eps=1e-6) # in v3, use grouped Norm
        self.pwconv1_global = nn.Linear(2*dim//3, 4 * 2*dim//3)  # in v3, use grouped FC
        
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                            requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x

        x_global1, x_global2, x_local = torch.chunk(x, 3, dim=1)

        # local part
        x_local = self.dwconv(x_local)
        x_local = self.norm_local(x_local.permute(0, 2, 3, 1))  # local-norm
        x_local = self.pwconv1_local(x_local).permute(0, 3, 1, 2)
        
        # global part, branch1 GCC-H, branch2 GCC-W, takes 2/3 in v2
        x_global1, x_global2 = self.gcc_conv_1H(x_global1), self.gcc_conv_2W(x_global2)
        # global fusion
        x_global = torch.cat((x_global1, x_global2), dim=1)
        x_global = self.norm_global(x_global.permute(0, 2, 3, 1)) # global-norm
        x_global = self.pwconv1_global(x_global).permute(0, 3, 1, 2)

        # local & global fusion
        x = torch.cat((x_global, x_local), dim=1)

        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

# v4 used preparameter skill for GCC
class rep_gcc_Conv2d(nn.Module):
    def __init__(self, dim, type, meta_kernel_size, rep_kernel_size, instance_kernel_method=None, bias=True, use_pe=True):
        super().__init__()
        self.type = type    # H or W
        self.dim = dim
        self.instance_kernel_method = instance_kernel_method
        assert instance_kernel_method is None # Rep-GCC cannot support DY resolution for now!
        self.meta_kernel_size_2 = (meta_kernel_size, 1) if self.type=='H' else (1, meta_kernel_size)
        self.meta_pe = nn.Parameter(torch.randn(1, dim, *self.meta_kernel_size_2)) if use_pe else None
        # GCC Parameters
        self.weight  = nn.Conv2d(dim, dim, kernel_size=self.meta_kernel_size_2, groups=dim).weight
        self.bias    = nn.Parameter(torch.randn(dim)) if bias else None
        # Rep Kernel Parametrs
        self.rep_kernel_size = rep_kernel_size
        self.rep_kernel_size_2 = (rep_kernel_size, 1) if self.type=='H' else (1, rep_kernel_size)
        self.rep_weight  = nn.Conv2d(dim, dim, kernel_size=self.rep_kernel_size_2, groups=dim).weight

    def gcc_init(self): # notice should also init rep-kernel
        trunc_normal_(self.weight, std=.02)
        trunc_normal_(self.rep_weight, std=.02)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if self.meta_pe is not None:
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
        # Rep Kernel branch
        x_cat = torch.cat((x, x[:, :, :self.rep_kernel_size-1, :]), dim=2) if self.type=='H'\
            else torch.cat((x, x[:, :, :, :self.rep_kernel_size-1]), dim=3)
        x_rep = F.conv2d(x_cat, weight=self.rep_weight, bias=None, padding=0, groups=self.dim) # bias only used in GCC brach!
        # GCC branch
        weight = self.get_instance_kernel((H, W))
        x_cat = torch.cat((x, x[:, :, :-1, :]), dim=2) if self.type=='H'\
            else torch.cat((x, x[:, :, :, :-1]), dim=3)
        x_gcc = F.conv2d(x_cat, weight=weight, bias=self.bias, padding=0, groups=self.dim)
        # branch fusion
        x = x_rep + x_gcc
        return x

# convert a RepGCC's state_dict to a normal GCC's state_dict
def convert_state_dict(old_state_dict):
    def zero_padding_as(input, target):
        input_shape, target_shape = input.shape, target.shape
        pad_2, pad = [], (torch.tensor(target_shape)-torch.tensor(input_shape)).tolist()
        for padi in reversed(pad):  # dim used in 'F.pad' is reversed
            pad_2 += [0, padi]      # we pad only on the RIGHT side
        return F.pad(input, pad=pad_2, mode='constant', value=0)
    new_state_dict = {}
    rep_dict, weight_dict = {}, {}
    for name in old_state_dict:
        if "gcc_conv_" in name and "weight" in name:     # save GCC weight and rep_weight for reprocess
            tmp_dict = rep_dict if "rep" in name else weight_dict
            tmp_dict[name] = old_state_dict[name]
        else:   # directly add other weight
            new_state_dict[name] = old_state_dict[name]
    for name in weight_dict:
        rep_name = name.replace("weight", "rep_weight")
        old_weight, rep_weight = weight_dict[name], rep_dict[rep_name]
        new_state_dict[name] = old_weight + zero_padding_as(rep_weight, target=old_weight)
    return new_state_dict

# v4 is Reparameter Test
class gcc_cvx_test_Block_v4(nn.Module):
    def __init__(self,
        dim,
        drop_path=0.,
        layer_scale_init_value=1e-6,
        meta_kernel_size=16,
        rep_kernel_size=3,
        instance_kernel_method=None,
        use_pe=True
    ):
        super().__init__()
        # super(gcc_cvx_Block, self).__init__()
        # branch1
        self.gcc_conv_1H = rep_gcc_Conv2d(dim//2, type='H', meta_kernel_size=meta_kernel_size,
            rep_kernel_size=rep_kernel_size, # Rep Kernel, default use 3
            instance_kernel_method=instance_kernel_method, use_pe=use_pe) 
        # branch2
        self.gcc_conv_2W = rep_gcc_Conv2d(dim//2, type='W', meta_kernel_size=meta_kernel_size,
            rep_kernel_size=rep_kernel_size, # Rep Kernel, default use 3
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