import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def show_ims(ims, columns=5, figsize=(20, 10), title=None, fontsize=None):
    """
    Show the given PIL images using matplotlib.

    Args:
        ims: A list of the PIL images
        columns: The number of columns
        figsize: The size of the matplotlib figure
        title: A list with the same length as `imgs`, giving the
            title to display above each respective image. If a single string is
            given, it is used as main title.
    """
    if title is not None and isinstance(title, list):
        assert len(title) == len(ims)

    fig = plt.figure(figsize=figsize)
    for i, im in enumerate(ims):
        plt.subplot(len(ims) // columns + 1, columns, i + 1)
        plt.imshow(np.asarray(im))
        plt.grid(b=False)
        plt.axis('off')

        if title is not None and isinstance(title, list):
            plt.title(title[i])
    if isinstance(title, str):
        fig.suptitle(title, fontsize=fontsize)


def show_imgs(imgs, columns=5, figsize=(20, 10), title=None):
    """
    Show the given images using matplotlib.

    Args:
        imgs: A list of the image paths
        columns: The number of columns
        figsize: The size of the matplotlib figure
        title: A list with the same length as `imgs`, giving the
            title to display above each respective image.
    """
    return show_ims([Image.open(img) for img in imgs],
                    columns=columns, figsize=figsize,
                    title=titles)


def scale_im(im, scale=0.3):
    """
    Return a rescaled version of the given PIL Image.
    """
    im = im.copy()
    im.thumbnail([int(s*scale) for s in im.size])
    return im
