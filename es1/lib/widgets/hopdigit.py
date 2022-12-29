from ipywidgets.widgets import Output, VBox, HBox, Label
import numpy as np
from PIL import Image
from scipy.io.matlab import loadmat
import torch
from tqdm.auto import tqdm

from ..models.hopfield import Hopfield
from ..utils.image_tools import show_ims


def ptrn2im(arr):
    arr = (arr + 1)/2
    return Image.fromarray(
        np.array(arr.reshape(16, 15) * 255)
        .astype('uint8')
    ).convert('RGB')


def get_digit_patterns():
    X = loadmat('digits.mat')['X']
    index_dig = list(range(0, 181, 20))
    return (torch.tensor(X[index_dig]).float() * 2) - 1


def show_digit_patterns(ptrns, title=None):
    ims = [
        ptrn2im(img_arr)
        for img_arr in ptrns
    ]
    show_ims(ims, columns=10, figsize=(5, 1.5),
             title=title)


def hopdigit(noise_level, num_steps):
    out_orig = Output()
    out_noise = Output()
    out_recon = Output()
    vbox = VBox([
        out_orig,
        out_noise,
        out_recon,
    ])

    T_digit = get_digit_patterns()

    with out_recon:
        show_digit_patterns(T_digit, title="Original")

    noise_level = noise_level*T_digit.max()
    P = T_digit + noise_level*torch.randn_like(T_digit)
    
    with out_noise:
        show_digit_patterns(P, title=f"Noisy (level {noise_level:.3f})")

    net_digit = Hopfield(T_digit)

    for _ in tqdm(range(num_steps)):
        P = net_digit(P)

    with out_recon:
        show_digit_patterns(P, title=f"After {num_steps} iters")

    return vbox