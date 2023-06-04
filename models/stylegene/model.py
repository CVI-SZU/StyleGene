import torch
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange

pair = lambda x: x if isinstance(x, tuple) else (x, x)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0., dense=nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )


class MappingSub2W(nn.Module):
    def __init__(self, N=8, dim=512, depth=6, expansion_factor=4., expansion_factor_token=0.5, dropout=0.1):
        super(MappingSub2W, self).__init__()
        num_patches = N * 34

        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        self.layer = nn.Sequential(
            Rearrange('b c h w -> b (c h) w'),
            *[nn.Sequential(
                PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
                PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
            ) for _ in range(depth)],
            nn.LayerNorm(dim),
            Rearrange('b c h -> b h c'),
            nn.Linear(34 * N, 34 * N),
            nn.LayerNorm(34 * N),
            nn.GELU(),
            nn.Linear(34 * N, N),
            Rearrange('b h c -> b c h')
        )

    def forward(self, x):
        return self.layer(x)


class MappingW2Sub(nn.Module):
    def __init__(self, N=8, dim=512, depth=8, expansion_factor=4., expansion_factor_token=0.5, dropout=0.1):
        super(MappingW2Sub, self).__init__()
        self.N = N
        num_patches = N * 34
        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

        self.layer = nn.Sequential(
            Rearrange('b c h -> b h c'),
            nn.Linear(N, num_patches),
            Rearrange('b h c -> b c h'),
            *[nn.Sequential(
                PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
                PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
            ) for _ in range(depth)],
            nn.LayerNorm(dim)
        )
        self.mu_fc = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
                PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
            ) for _ in range(2)],
            nn.LayerNorm(dim),
            nn.Tanh(),
            Rearrange('b c h -> b (c h)')
        )
        self.var_fc = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
                PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
            ) for _ in range(2)],
            nn.LayerNorm(dim),
            nn.Tanh(),
            Rearrange('b c h -> b (c h)')
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps * std + mu

    def forward(self, x):
        f = self.layer(x)
        mu = self.mu_fc(f)
        var = self.var_fc(f)

        z = self.reparameterize(mu, var)
        z = rearrange(z, 'a (b c d) -> a b c d', b=self.N, c=34)
        return rearrange(mu, 'a (b c d) -> a b c d', b=self.N, c=34), rearrange(var, 'a (b c d) -> a b c d',
                                                                                b=self.N, c=34), z


class HeadEncoder(nn.Module):
    def __init__(self, N=8, dim=512, depth=2, expansion_factor=4., expansion_factor_token=0.5, dropout=0.1):
        super(HeadEncoder, self).__init__()
        channels = [32, 64, 64, 64]
        self.N = N
        num_patches = N
        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

        self.s1 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(),
            nn.Conv2d(channels[1], channels[2], kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(channels[2]),
            nn.LeakyReLU(),
            nn.Conv2d(channels[2], channels[3], kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(channels[3]),
            nn.LeakyReLU())
        self.mlp1 = nn.Linear(channels[3] * 8 * 8, 512)

        self.up_N = nn.Linear(1, N)

        self.mu_fc = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
                PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
            ) for _ in range(depth)],
            nn.LayerNorm(dim),
            nn.Tanh()
        )
        self.var_fc = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
                PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
            ) for _ in range(depth)],
            nn.LayerNorm(dim),
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        feature = self.s1(x)
        s2 = torch.flatten(feature, start_dim=1)
        s2 = self.mlp1(s2).unsqueeze(2)
        s2 = self.up_N(s2)
        s2 = rearrange(s2, 'b h c -> b c h')
        mu = self.mu_fc(s2)
        var = self.var_fc(s2)
        z = self.reparameterize(mu, var)
        return mu, var, z


class RegionEncoder(nn.Module):
    def __init__(self, N=8):
        super(RegionEncoder, self).__init__()
        channels = [8, 16, 32, 32, 64, 64]
        self.s1 = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, stride=2)
        self.s2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(channels[1]),
            nn.LeakyReLU(),
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(channels[2]),
            nn.LeakyReLU()
        )
        self.heads = nn.ModuleList()
        for i in range(34):
            self.heads.append(HeadEncoder(N=N))

    def forward(self, x, all_mask=None):
        s1 = self.s1(x)
        s2 = self.s2(s1)
        result = []
        mus = []
        log_vars = []
        for i, head in enumerate(self.heads):
            m = all_mask[:, i, :].unsqueeze(1)
            mu, var, z = head(s2 * m)
            result.append(z.unsqueeze(2))
            mus.append(mu.unsqueeze(2))
            log_vars.append(var.unsqueeze(2))

        return torch.cat(mus, dim=2), torch.cat(log_vars, dim=2), torch.cat(result, dim=2)
