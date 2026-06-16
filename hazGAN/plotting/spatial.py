# %%
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from ..statistics import pairwise_extremal_coeffs

from .base import CMAP
from .base import makegrid


def plot(fake, train, func, field=0, figsize=1., cmap=CMAP, vmin=None, vmax=None,
         title="Untitled", cbar_label="", alpha:float=1e-4, **func_kws):
    train = train[..., field].copy()
    fake  = fake[..., field].copy()
    
    train_res = func(train)
    fake_res  = func(fake)

    if alpha is not None:
        vmin = vmin or np.nanquantile(np.concatenate([fake_res, train_res]), alpha)
        vmax = vmax or np.nanquantile(np.concatenate([fake_res, train_res]), 1-alpha)
    else:
        vmin = vmin or train_res.min()
        vmax = vmax or train_res.max()
    
    cmap = getattr(plt.cm, cmap)
    cmap.set_under(cmap(0))
    cmap.set_over(cmap(.99))

    fig, axs, cax = makegrid(1, 2, figsize=figsize, projection=None)

    im = axs[0].imshow(train_res, cmap=cmap, vmin=vmin, vmax=vmax)
    _  = axs[-1].imshow(fake_res, cmap=cmap, vmin=vmin, vmax=vmax)

    for ax in axs:
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

    axs[0].set_title("ERA5", y=-0.15)
    axs[-1].set_title("HazGAN", y=-0.15)
    fig.suptitle(title, y=1.05)

    fig.colorbar(im, cax=cax, label=cbar_label)

    corr = np.corrcoef(train_res.flatten(), fake_res.flatten())[0, 1]
    print(f"Pearson correlation: {corr:.4f}")

    mae = np.mean(np.abs(train_res - fake_res))
    print(f"Mean Absolute Error: {mae:.4f}")

    return fig, {"mae": mae, "pearson": corr}


def smith1990(array):
    array = array.astype(np.float16)
    return pairwise_extremal_coeffs(array)


def pearson(array):
    n, h, w = array.shape
    array = array.reshape(n, h * w)
    corrs = np.corrcoef(array.T)
    print(corrs.shape)
    return corrs


def tail_dependence(array, thresholds=np.arange(0.8, 0.99, 0.01), dtype=np.float32):
    """
    Fast upper-tail dependence matrix for one spatial field.

    Parameters
    ----------
    array : np.ndarray
        Array of shape (n, h, w), assumed to have uniform marginals.
        n = number of samples/images.
        h, w = spatial dimensions.

    thresholds : array-like
        Thresholds over which to average the finite-threshold tail dependence.

    dtype : np.dtype
        Numeric dtype used for matrix multiplication and output.

    Returns
    -------
    chi : np.ndarray
        Array of shape (h*w, h*w), where

            chi[i, j] ≈ mean_t P(pixel_j > t | pixel_i > t)

        averaged over the supplied thresholds.
    """
    print("Computing tail dependence matrix...")
    
    n, h, w = array.shape
    p = h * w

    x = array.reshape(n, p)

    chi_sum = np.zeros((p, p), dtype=dtype)
    chi_count = np.zeros((p, p), dtype=dtype)

    for t in tqdm(thresholds, desc="Tail thresholds"):
        # Boolean exceedance matrix: shape (n, p)
        # exc[s, i] = True if sample s, pixel i exceeds threshold t
        exc = x > t

        # denom[i] = number of samples where pixel i exceeds threshold t
        denom = exc.sum(axis=0).astype(dtype)

        # Convert to numeric so matrix multiplication gives joint exceedance counts.
        exc_float = exc.astype(dtype)

        # numer[i, j] = number of samples where both pixel i and pixel j exceed t
        numer = exc_float.T @ exc_float

        # vals[i, j] = P(pixel_j > t | pixel_i > t)
        vals = np.divide(
            numer,
            denom[:, None],
            out=np.zeros_like(numer, dtype=dtype),
            where=denom[:, None] > 0,
        )

        valid = denom > 0

        chi_sum += vals
        chi_count += valid[:, None]

    chi = np.divide(
        chi_sum,
        chi_count,
        out=np.zeros_like(chi_sum),
        where=chi_count > 0,
    )

    # Optional but usually sensible:
    # P(pixel_i > t | pixel_i > t) = 1 when there are exceedances.
    np.fill_diagonal(chi, 1.0)

    return chi

def _tail_dependence_coeff(u, v):
    """
    Classical tail dependence coefficient λ for upper tail dependence.

    Args:
        u, v: 1D arrays of uniform marginals
        tail: 'upper' or 'lower'
    Returns:
        λ: tail dependence coefficient

    Refs:
        Joe, H. (1997). Multivariate Models and Dependence Concepts. Chapman & Hall.
        Nelsen, R. B. (2006). An Introduction to Copulas. Springer.
    """
    thresholds = np.arange(0.8, 0.99, 0.01)  # Multiple thresholds
    lambdas = []
    
    for t in thresholds:
        u_exceed = u > t
        both_exceed = (u > t) & (v > t)
        
        if np.sum(u_exceed) > 0:
            lambda_t = np.sum(both_exceed) / np.sum(u_exceed)
            lambdas.append(lambda_t)

    return np.mean(lambdas) if lambdas else 0
