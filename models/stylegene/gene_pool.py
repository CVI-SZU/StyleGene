import os
import random
import pandas as pd
import torch.nn.functional as F

from .util import load_img
from configs import path_csv_ffhq_attritube


class GenePoolFactory(object):
    def __init__(self, root_ffhq, device, mean_latent, max_sample=100):
        self.device = device
        self.mean_latent = mean_latent
        self.root_ffhq = root_ffhq
        self.max_sample = max_sample

        self.pools = {}
        path_ffhq_attributes = path_csv_ffhq_attritube
        self.df = pd.read_csv(path_ffhq_attributes)
        self.df.replace('Male', 'male', inplace=True)
        self.df.replace('Female', 'female', inplace=True)

    def __call__(self, encoder, w2sub34, age, gender, race):
        keyname = f'{age}-{gender}-{race}'
        if keyname in self.pools.keys():
            return self.pools[keyname]
        elif self.root_ffhq is not None:
            result = self.df.query(f'gender == "{gender}" and age == "{age}" and race == "{race}"')
            result = result[['file_id']].values
            tmp = []
            random.shuffle(result)
            for fid in result[:self.max_sample]:
                filename = format(int(fid[0]), '05d') + ".png"
                img = load_img(os.path.join(self.root_ffhq, filename))
                img = img.to(self.device)
                w18_1 = encoder(F.interpolate(img, size=(256, 256))) + self.mean_latent
                mu, var, sub34_1 = w2sub34(w18_1)
                tmp.append((mu.cpu(), var.cpu()))
            self.pools[keyname] = tmp
            return self.pools[keyname]
        else:
            return []
