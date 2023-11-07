"""
This module provides code for calculating spatial en metrics from n dimensional data structures

Author: Jeffrey L. Werbin
"""
import numpy as np
from scipy.fft import fft, fft2, fftn
from scipy.stats import norm
from typing import Tuple


fft_method = {1: fft, 2: fft2}


def fft_calc(data: np.array, **kwargs) -> np.array:
    """A generic fft method that picks the appropriate dimensional transform method"""
    return fft_method.get(len(data.shape), fftn)(data, **kwargs)


def calculate_kspace_probability_array(data: np.array, step: float = 0.005, **kwargs) -> np.array:
    """
    Calculates the frequency (K-space) probability in accordance with the paper Heinz et. al. 2009.
    It uses the following mathematical properties
        1. The fourier coeffiecents of random images are guassianly distributed.
        2. Standard Deviation of image values in real space is the same as the standard deviation of fourier coeffiecents
           This comes directly from [Parseval's Theorem](https://en.wikipedia.org/wiki/Parseval%27s_theorem)

    Taking these two together we can calculate how likely a particular fourier coefficent could have come from a random image. 
    
    
    Heinz, W.F., Werbin, J.L., Lattman, E.E., & Hoh, J.H. (2009). Computing Spatial Information from Fourier Coefficient Distributions. The Journal of Membrane Biology, 241, 59-68.

    inputs:

        data,               An N dimensional array of real numbers
        step,               Number of sigma to integrate over from x - step*sigma to x + step*sigma

    outputs:
        probabilities,      An N dimensional array the same size as data

    """
    centered = data - np.mean(data)
    std = np.std(centered)

    # Calculate the exponent of the guassian
    def guassian_prob(a: np.array, std, step):
        return norm.cdf(a + step*std, loc=0, scale=std) - norm.cdf(a - step*std, loc=0, scale=std)

    transformed = fft_calc(centered)
    Iks_real = guassian_prob(np.real(transformed), std)
    Iks_imag = guassian_prob(np.imag(transformed), std)

    return Iks_real, Iks_imag


def calculate_spatial_information(data: np.array, step: float = 0.005, **kwargs) -> np.array:
    """
    Calculates the spatial en of each fourier coefficent of the array.
    Using the Shannon definition of informatio -log(p) and calculating p of each coefficent
    that would be expected if the array was randomly generated.

    inputs:

        data,               An N dimensional array of real numbers
        step,               Number of sigma to integrate over from x - step*sigma to x + step*sigma

    outputs:

        Iks_real,      An N dimensional array the same size as data. Containing the information of revalue coeffs
        Iks_imag,      An N dimensional array the same size as data. Containing the information of revalue coeffs
    """
    p_r, p_i = calculate_kspace_probability_array(data, step=step, **kwargs)
    return -1 * np.log(p_r), -1 * np.log(p_i)


def calculate_image_entropy(data: np.array, bin_size, bin_range):
    """"""
    centered = data - np.mean(data)
    std = np.std(centered)
    step = bin_size * std
    b = np.arange(bin_range[0] *std, bin_range[1] * std + step, step)
    b = np.append(b, bin_range[1] * std)

    p = norm.cdf(b[1:], loc=0, scale=std) - norm.cdf(b[:-1], loc=0, scale=std)
    num= data.size
    hks = -2 * num * np.sum( np.multiply(p, np.log(p)))
    return hks


def kspace_information(data: np.array, bin_size:float = 0.01, bin_range: Tuple[float, float] = (-10, 10), **kwargs) -> float:
    """
    Calculates K Space en of an array in accordance with
    Heinz, W.F., Werbin, J.L., Lattman, E.E., & Hoh, J.H. (2009). Computing Spatial Information from Fourier Coefficient Distributions. The Journal of Membrane Biology, 241, 59-68.

    kSI = Hks - (sum(Iks_real) + sum(Iks_imag)

    where Hks represents the maxium entropy of the image


    inputs:

        data,                   An N dimensional array of real numbers
        bin_size,               Bin size for the histogram to calculate Hks in units of sigma. Default = 0.01
        bin_range,              Inclusive size of bins. Default [-10, 10]              

    outputs:

        spatial_information,    A measure of spatial information
    """

    h_ks = calculate_image_entropy(data, bin_size, bin_range)
    Iks_r, Iks_im = calculate_spatial_information(data, step=bin_size/2)
    return h_ks - np.nansum(Iks_r) - np.nansum(Iks_im) 
