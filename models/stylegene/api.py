import torch
import numpy as np
import torch.nn.functional as F
from models.stylegan2.model import Generator
from models.encoders.psp_encoders import Encoder4Editing
from models.stylegene.model import MappingSub2W, MappingW2Sub
from models.stylegene.util import get_keys, requires_grad, load_img
from models.stylegene.gene_pool import GenePoolFactory
from models.stylegene.gene_crossover_mutation import fuse_latent
from models.stylegene.fair_face_model import init_fair_model, predict_race
from configs import path_ckpt_e4e, path_ckpt_stylegan2, path_ckpt_stylegene, path_ckpt_genepool, path_dataset_ffhq
from preprocess.align_images import align_face
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def init_model(image_size=1024, latent_dim=512):
    ckp = torch.load(path_ckpt_e4e, map_location='cpu')
    encoder = Encoder4Editing(50, 'ir_se', image_size).eval()
    encoder.load_state_dict(get_keys(ckp, 'encoder'), strict=True)
    mean_latent = ckp['latent_avg'].to('cpu')
    mean_latent.unsqueeze_(0)

    generator = Generator(image_size, latent_dim, 8)
    checkpoint = torch.load(path_ckpt_stylegan2, map_location='cpu')
    generator.load_state_dict(checkpoint["g_ema"], strict=False)
    generator.eval()
    sub2w = MappingSub2W(N=18).eval()
    w2sub34 = MappingW2Sub(N=18).eval()
    ckp = torch.load(path_ckpt_stylegene, map_location='cpu')
    w2sub34.load_state_dict(get_keys(ckp, 'w2sub34'))
    sub2w.load_state_dict(get_keys(ckp, 'sub2w'))

    requires_grad(sub2w, False)
    requires_grad(w2sub34, False)
    requires_grad(encoder, False)
    requires_grad(generator, False)
    return encoder, generator, sub2w, w2sub34, mean_latent


# init model
encoder, generator, sub2w, w2sub34, mean_latent = init_model()
encoder, generator, sub2w, w2sub34, mean_latent = encoder.to(device), generator.to(device), sub2w.to(
    device), w2sub34.to(device), mean_latent.to(device)
model_fair_7 = init_fair_model(device)  # init FairFace model

# load a GenePool
geneFactor = GenePoolFactory(root_ffhq=path_dataset_ffhq, device=device, mean_latent=mean_latent, max_sample=300)
geneFactor.pools = torch.load(path_ckpt_genepool)
print("gene pool loaded!")


def tensor2rgb(tensor):
    tensor = (tensor * 0.5 + 0.5) * 255
    tensor = torch.clip(tensor, 0, 255).squeeze(0)
    tensor = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    tensor = tensor.astype(np.uint8)
    return tensor


def generate_child(w18_F, w18_M, random_fakes, gamma=0.46, eta=0.4):
    w18_syn = fuse_latent(w2sub34, sub2w, w18_F=w18_F, w18_M=w18_M,
                          random_fakes=random_fakes, fixed_gamma=gamma, fixed_eta=eta)

    img_C, _ = generator([w18_syn], return_latents=True, input_is_latent=True)
    return img_C, w18_syn


def synthesize_descendant(pF, pM, attributes=None):
    gender_all = ['male', 'female']
    ages_all = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
    if attributes is None:
        attributes = {'age': ages_all[0], 'gender': gender_all[0], 'gamma': 0.47, 'eta': 0.4}
    imgF = align_face(pF)
    imgM = align_face(pM)
    imgF = load_img(imgF)
    imgM = load_img(imgM)
    imgF, imgM = imgF.to(device), imgM.to(device)

    father_race, _, _, _ = predict_race(model_fair_7, imgF.clone(), imgF.device)
    mother_race, _, _, _ = predict_race(model_fair_7, imgM.clone(), imgM.device)

    w18_1 = encoder(F.interpolate(imgF, size=(256, 256))) + mean_latent
    w18_2 = encoder(F.interpolate(imgM, size=(256, 256))) + mean_latent

    random_fakes = []
    for r in list({father_race, mother_race}):  # search RFGs from Gene Pool
        random_fakes = random_fakes + geneFactor(encoder, w2sub34, attributes['age'], attributes['gender'], r)
    img_C, w18_syn = generate_child(w18_1.clone(), w18_2.clone(), random_fakes,
                                    gamma=attributes['gamma'], eta=attributes['eta'])
    img_C = tensor2rgb(img_C)
    img_F = tensor2rgb(imgF)
    img_M = tensor2rgb(imgM)

    return img_F, img_M, img_C
