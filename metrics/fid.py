"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import argparse

import paddorch as porch
import paddorch.nn as nn
import numpy as np
from paddorch.vision.models.inception import InceptionV3
from scipy import linalg
from core.data_loader import get_eval_loader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x



def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu -mu2)**2) + np.trace(cov + cov2 - 2*cc)
    return np.real(dist)


@porch.no_grad()
def calculate_fid_given_paths(paths, img_size=256, batch_size=50):
    print('Calculating FID given paths %s and %s...' % (paths[0], paths[1]))
    device = porch.device('cuda' if porch.cuda.is_available() else 'cpu')
    inception =   InceptionV3("./metrics/inception_v3_pretrained.pdparams")
    inception.eval()
    loaders = [get_eval_loader(path, img_size, batch_size) for path in paths]

    mu, cov = [], []
    for loader in loaders:
        actvs = []
        for x in tqdm(loader, total=len(loader)):
            x=porch.varbase_to_tensor(x[0])
            actv = inception(x )
            actvs.append(actv)
        actvs = porch.cat(actvs, dim=0).numpy()
        mu.append(np.mean(actvs, axis=0))
        cov.append(np.cov(actvs, rowvar=False))
    fid_value = frechet_distance(mu[0], cov[0], mu[1], cov[1])
    return fid_value.astype(float)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', type=str, nargs=2, help='paths to real and fake images')
    parser.add_argument('--img_size', type=int, default=256, help='image resolution')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size to use')
    args = parser.parse_args()
    from paddle import fluid
    place = fluid.CUDAPlace(0)
   
    with fluid.dygraph.guard(place=place):
        fid_value = calculate_fid_given_paths(args.paths, args.img_size, args.batch_size)
        print('FID: ', fid_value)

# python -m metrics.fid --paths PATH_REAL PATH_FAKE