import torch
import numpy as np
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class Vis_Embedder(nn.Module):
    def __init__(self, in_dim,  hidden_size):
        super().__init__()
        self.emb = nn.Linear(in_dim, hidden_size)
        self.mlp = nn.Sequential(
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t):
        t_freq = self.emb(t)
        t_emb = self.mlp(t_freq)
        return t_emb


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers_x = nn.ModuleList([])
        self.layers_c = nn.ModuleList([])
        for _ in range(depth):
            self.layers_x.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout),
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(dim//4, 6 * dim // 4, bias=True)
                ),
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(dim//4, 6 * dim * 3//4, bias=True)
                ),
                nn.LayerNorm(dim),
                nn.LayerNorm(dim),
            ]))

        for i in range(depth):
            nn.init.zeros_(self.layers_x[i][2][1].weight)
            nn.init.zeros_(self.layers_x[i][2][1].bias)
            nn.init.zeros_(self.layers_x[i][3][1].weight)
            nn.init.zeros_(self.layers_x[i][3][1].bias)
                  
    def forward(self, x, mu, f):
        for attn, ff, adaLN_modulation_mu, adaLN_modulation_f, norm1, norm2 in self.layers_x:
            shift_msa_mu, scale_msa_mu, gate_msa_mu, shift_mlp_mu, scale_mlp_mu, gate_mlp_mu = adaLN_modulation_mu(mu).chunk(6, dim=-1)
            shift_msa_f, scale_msa_f, gate_msa_f, shift_mlp_f, scale_mlp_f, gate_mlp_f = adaLN_modulation_f(f).chunk(6, dim=-1)
            x_attn = attn(modulate(norm1(x), torch.cat([shift_msa_mu, shift_msa_f],dim=-1), torch.cat([scale_msa_mu, scale_msa_f],dim=-1)))
            x = x + torch.cat([gate_msa_mu, gate_msa_f],dim=-1) * x_attn
            x = x + torch.cat([gate_mlp_mu, gate_mlp_f],dim=-1) * ff(modulate(norm2(x), torch.cat([shift_mlp_mu, shift_mlp_f],dim=-1), torch.cat([scale_mlp_mu, scale_mlp_f],dim=-1)))
        return x
    
class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation_mu = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size//4, 2 * hidden_size//4, bias=True)
        )
        self.adaLN_modulation_f = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size//4, 2 * hidden_size *3//4, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation_mu[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation_mu[-1].bias, 0)
        nn.init.constant_(self.adaLN_modulation_f[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation_f[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x, mu, f):
        shift_mu, scale_mu = self.adaLN_modulation_mu(mu).chunk(2, dim=-1)
        shift_f, scale_f = self.adaLN_modulation_f(f).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), torch.cat([shift_mu, shift_f],dim=-1), torch.cat([scale_mu, scale_f],dim=-1))
        x = self.linear(x)
        return x

resolution = 64
class Model(nn.Module):
    def __init__(self, image_size=resolution, patch_size=4, dim=256, depth=8, heads=8, mlp_dim=256, pool = 'cls', in_channels = 10, out_channels=1, dim_head = 32, dropout = 0., emb_dropout = 0.,args=None):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.out_channels = out_channels
        self.patch_num_height = image_height // patch_height
        self.patch_num_width = image_width // patch_width

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        in_channels = args.in_dim
        self.ref = 4
        self.pos = self.get_grid([resolution, resolution], 'cuda')
        patch_dim = (in_channels + self.ref*self.ref) * self.patch_height * self.patch_width

        self.mu_embedder = Vis_Embedder(1, dim // 4)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        patch_dim_force = (1) * self.patch_height * self.patch_width
        
        self.to_patch_embedding_force = nn.Sequential(
            Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim_force, dim // 4)
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.mlp_head = FinalLayer(mlp_dim, patch_size=patch_height, out_channels=out_channels)

    def forward(self, x, mu, f):
        B, H, W, C = x.shape
        mu = mu[:, None] * 3200 
        mu = self.mu_embedder(mu)[:, None,:]
        
        f = f[..., None]
        f = self.to_patch_embedding_force(f)
        mu = mu.repeat(1, f.shape[1], 1)

        grid = self.pos.repeat(B,1,1,1)

        x = torch.cat((x, grid), dim=-1)
        x = x.permute(0, 3, 1, 2)  

        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        x = self.dropout(x)

        x = self.transformer(x, mu, f)

        b,l,ch = x.shape
        x = self.mlp_head(x, mu, f).reshape(b, self.patch_num_height, self.patch_num_width, self.patch_height, self.patch_width, self.out_channels).permute(0,1,3,2,4,5).contiguous()
        x = x.reshape(b, self.patch_num_height * self.patch_height, self.patch_num_width * self.patch_width, self.out_channels)
        return x


    def get_grid(self, shape, device):
        size_x, size_y = shape[0], shape[1]
        batchsize = 1
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).to(device)  # B H W 2


        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1).to(device)  # B H W 8 8 2

        pos = torch.sqrt(torch.sum((grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, size_x, size_y, self.ref * self.ref).contiguous()
        return pos

