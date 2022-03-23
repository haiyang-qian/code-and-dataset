from Visualizer.visualizer import get_local
# get_local.activate()
import torch
import torch.nn as nn
from functools import partial
import math
import Visualizer
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'tnt_s_patch16_224': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'tnt_b_patch16_224': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'myvt_224_h8_edim64':_cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Tokenizer(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_conv_layers=1,
                 n_input_channels=3,
                 n_output_channels=64,
                 in_planes=64,
                 activation=None,
                 max_pool=True,
                 conv_bias=False):
        super(Tokenizer, self).__init__()

        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [n_output_channels]

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(n_filter_list[i], n_filter_list[i + 1],
                          kernel_size=(kernel_size, kernel_size),
                          stride=(stride, stride),
                          padding=(padding, padding), bias=conv_bias),
                nn.Identity() if activation is None else activation(),
                nn.MaxPool2d(kernel_size=pooling_kernel_size,
                             stride=pooling_stride,
                             padding=pooling_padding) if max_pool else nn.Identity()
            )
                for i in range(n_conv_layers)
            ])

        self.flattener = nn.Flatten(2, 3)
        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        return self.flattener(self.conv_layers(x)).transpose(-2, -1)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SE(nn.Module):
    def __init__(self, dim, hidden_ratio=None):
        super().__init__()
        hidden_ratio = hidden_ratio or 1
        self.dim = dim
        hidden_dim = int(dim * hidden_ratio)
        self.fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim, dim),
            nn.Tanh()
        )

    def forward(self, x):
        a = x.mean(dim=1, keepdim=True)  # B, 1, C
        a = self.fc(a)
        x = a * x
        return x


class Attention(nn.Module):
    def __init__(self, dim, hidden_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qk = nn.Linear(dim, hidden_dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=False)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=False)
    @get_local("attn")
    def forward(self, x):
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        q, k = qk[0], qk[1]  # make torchscript happy (cannot use tensor as tuple)
        v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Block(nn.Module):

    def __init__(self, embed_dim,embed_num_heads, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, se=0):
        super().__init__()


        #embed_patch_tokens
        self.embed_norm1 = norm_layer(embed_dim)
        self.embed_attn = Attention(
            embed_dim,embed_dim,num_heads=embed_num_heads,qkv_bias=qkv_bias,
            qk_scale=qk_scale,attn_drop=attn_drop,proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.embed_norm2 = norm_layer(embed_dim)
        self.embed_mlp = Mlp(embed_dim,hidden_features=int(embed_dim * mlp_ratio),
                             out_features=embed_dim,act_layer=act_layer,drop=drop)


    def forward(self,feature_tokens):

        feature_tokens = feature_tokens + self.drop_path(self.embed_attn(self.embed_norm1(feature_tokens)))
        feature_tokens = feature_tokens + self.drop_path(self.embed_mlp(self.embed_norm2(feature_tokens)))
        return feature_tokens



class MYVT(nn.Module):

    def __init__(self,embed_dim,embed_num_heads,kernel_size,stride,padding,pooling_kernel_size=3,pooling_stride=2,pooling_padding=1,
                 n_conv_layers = 1,n_input_channels=3,n_output_channels=64,in_planes=64,activation=None,
                 max_pool=True,conv_bias=False,
                 mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,

                 num_classes=4,
                 depth=12,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,  se=0):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim

        self.feature_embed = Tokenizer(
            kernel_size=kernel_size,stride=stride,padding=padding,pooling_kernel_size=pooling_kernel_size,
            pooling_stride=pooling_stride,pooling_padding=pooling_padding,n_conv_layers=n_conv_layers,
            n_input_channels=n_input_channels,n_output_channels=n_output_channels,in_planes=in_planes,
            activation=activation,max_pool=max_pool,conv_bias=conv_bias)
        self.num_tokens = self.feature_embed.sequence_length()


        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.tokens_pos = nn.Parameter(torch.zeros(1, self.num_tokens + 1, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        vanilla_idxs = []
        blocks = []
        for i in range(depth):
            blocks.append(Block(
                embed_dim, embed_num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], act_layer=act_layer,
                norm_layer=norm_layer, se=se))


        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.tokens_pos, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'outer_pos', 'inner_pos', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.outer_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]

        feature_tokens = self.feature_embed(x)
        feature_tokens = torch.cat((self.cls_token.expand(B, -1, -1), feature_tokens), dim=1)
        feature_tokens = feature_tokens + self.tokens_pos
        feature_tokens = self.pos_drop(feature_tokens)

        for blk in self.blocks:
            feature_tokens = blk(feature_tokens)

        feature_tokens = self.norm(feature_tokens)
        return feature_tokens[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

@register_model
def myvt_224_h8_edim64(pretrained=False,**kwargs):
    embed_dim = 64
    embed_num_heads = 4
    kernel_size = 7
    stride = 4
    padding = 1
    model = MYVT(embed_dim=embed_dim,embed_num_heads=embed_num_heads,kernel_size=kernel_size,
                 stride=stride,padding=padding,**kwargs)
    model.default_cfg = default_cfgs['myvt_224_h8_edim64']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model
