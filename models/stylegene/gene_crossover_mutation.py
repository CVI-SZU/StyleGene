import torch
from .data_util import face_class, face_shape
import random


def reparameterize(mu, logvar):
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


def mix(w18_F, w18_M, w18_syn):
    for k in [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]:
        w18_syn[:, k, :] = w18_F[:, k, :] * 0.5 + w18_M[:, k, :] * 0.5
    return w18_syn


def fuse_latent(w2sub34, sub2w, w18_F, w18_M, random_fakes, fixed_gamma=0.47, fixed_eta=0.4):
    device = w18_F.device

    mu_F, var_F, sub34_F = w2sub34(w18_F)
    mu_M, var_M, sub34_M = w2sub34(w18_M)
    new_sub34 = torch.zeros_like(sub34_F, dtype=torch.float, device=device)

    if len(random_fakes) == 0:  # EXCEPTION HANDLER (No matching gene pool)
        random_fakes = [(mu_F.cpu(), var_F.cpu())] + [(mu_M.cpu(), var_M.cpu())]

    # region genetic variation weights
    weights = {}
    for i in face_class:
        weights[i] = (random.uniform(0, 1 - float(fixed_gamma)), float(fixed_gamma))

    # select genetic regions
    cur_class = random.sample(face_class, int(len(face_class) * (1 - float(fixed_eta))))

    for i, classname in enumerate(face_class):
        if classname == 'background':
            new_sub34[:, :, i, :] = reparameterize(mu_F[:, :, i, :], var_F[:, :, i, :])
            continue

        if classname in cur_class:  # # corresponding to t = 0 in Eq.10
            fake_mu, fake_var = random.choice(random_fakes)
            w_i, b_i = weights[classname]
            new_sub34[:, :, i, :] = reparameterize(
                mu_F[:, :, i, :] * w_i + fake_mu[:, :, i, :].to(device) * b_i + mu_M[:, :, i, :] * (1 - w_i - b_i),
                var_F[:, :, i, :] * w_i + fake_var[:, :, i, :].to(device) * b_i + var_M[:, :, i, :] * (1 - w_i - b_i))
        else:  # corresponding to t = 1 in Eq.10
            fake_mu, fake_var = random.choice(random_fakes)
            fake_latent = reparameterize(fake_mu[:, :, i, :], fake_var[:, :, i, :]).to(device)
            var = fake_latent
            new_sub34[:, :, i, :] = new_sub34[:, :, i, :] + var
    w18_syn = sub2w(new_sub34)

    w18_syn = mix(w18_F, w18_M, w18_syn)

    return w18_syn
