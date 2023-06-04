import numpy as np
from PIL import Image
from torchvision import transforms


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


def load_img(path_img, img_size=(256, 256)):
    transform = transforms.Compose(
        [transforms.Resize(img_size),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5),
                              (0.5, 0.5, 0.5))])
    if type(path_img) is np.ndarray:
        img = Image.fromarray(path_img)
    else:
        img = Image.open(path_img).convert('RGB')
    img = transform(img)
    img.unsqueeze_(0)
    return img
