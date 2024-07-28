from random import randrange
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers


def exists(val):
    return val is not None


def dropout_layers(layers, dropout):
    if dropout == 0:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) < dropout

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

# classes


class LayerScale(nn.Module):
    def __init__(self, dim, fn, depth):
        super().__init__()
        if depth <= 18:  # epsilon detailed in section 2 of paper
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # talking heads matrix, can be removed for small dataset
        self.mix_heads_pre_attn = nn.Parameter(torch.randn(heads, heads))
        self.mix_heads_post_attn = nn.Parameter(torch.randn(heads, heads))

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        b, n, _, h = *x.shape, self.heads

        context = x if not exists(context) else torch.cat((x, context), dim=1)

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # talking heads, pre-softmax, can be removed for small dataset
        attn = torch.matmul(attn.permute(0, 2, 3, 1), self.mix_heads_pre_attn).permute(0, 3, 1, 2)

        attn = self.attend(attn)
        attn = self.dropout(attn)

        # talking heads, post-softmax, can be removed for small dataset
        attn = torch.matmul(attn.permute(0, 2, 3, 1), self.mix_heads_post_attn).permute(0, 3, 1, 2)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., layer_dropout=0., height=None, width=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layer_dropout = layer_dropout
        self.height = height
        self.width = width

        if self.height:
            self.gelu = nn.GELU()
            self.convs = nn.ModuleList([])
            self.batchnorms = nn.ModuleList([])
            for ind in range(depth):
                self.convs.append(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, groups=dim))
                self.batchnorms.append(nn.BatchNorm2d(dim))
        for ind in range(depth):
            # with layerscale
            self.layers.append(nn.ModuleList([
                LayerScale(dim, PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)), depth=ind + 1),
                LayerScale(dim, PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)), depth=ind + 1)
            ]))

            # without layerscale for small dataset
            # self.layers.append(nn.ModuleList([
            #     PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
            #     PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            # ]))

    def forward(self, x, context=None):
        layers = dropout_layers(self.layers, dropout=self.layer_dropout)

        if self.height:
            for i, [attn, ff] in enumerate(layers):
                b, n, d = x.shape
                shortcut = x[:]
                shortcut = shortcut.view(b, self.height, self.width, d).permute(0, 3, 1, 2).contiguous()
                shortcut = self.gelu(shortcut)
                shortcut = self.batchnorms[i](shortcut)
                shortcut = self.convs[i](shortcut)
                shortcut = shortcut.permute(0, 2, 3, 1).view(b, self.height * self.width, d).contiguous()
                x = attn(x, context=context) + x
                x = ff(x) + x
                x = shortcut + x
        else:
            for attn, ff in layers:
                x = attn(x, context=context) + x
                x = ff(x) + x

        return x


class CaiT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        cls_depth,
        heads,
        mlp_dim,
        dim_head=64,
        dropout=0.,
        emb_dropout=0.,
        layer_dropout=0.
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2
        height = image_size // patch_size
        width = image_size // patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.patch_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, layer_dropout, height, width)
        self.cls_transformer = Transformer(dim, cls_depth, heads, dim_head, mlp_dim, dropout, layer_dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.patch_transformer(x)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = self.cls_transformer(cls_tokens, context=x)

        return self.mlp_head(x[:, 0])
