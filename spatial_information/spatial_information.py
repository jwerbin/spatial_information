"""
This module provides code for calculating spatial en metrics from n dimensional data structures

Author: Jeffrey L. Werbin
"""
import numpy as np
from scipy.fft import fft, fft2, fftn


fft_method = {1: fft, 2: fft2}


def fft_cal(data: np.array, **kwargs) -> np.array:
    """A generic fft method that picks the appropriate dimensional transform method"""
    return fft_method.get(len(data.shape), fftn)(data, **kwargs)


def calculate_kspace_probability_array(data: np.array, **kwargs) -> np.array:
    """
    Calculates the frequency (K-space) probability in accordance with the paper Heinz et. al. 2009.
    It uses the following mathematical properties
        1. The fourier coeffiecents of random images are guassianly distributed.
        2. Standard Deviation of image values in real space is the same as the standard deviation of fourier coeffiecents

    Taking these two together we can calculate how likely a particular fourier coefficent could have come from a random image. 
    
    
    Heinz, W.F., Werbin, J.L., Lattman, E.E., & Hoh, J.H. (2009). Computing Spatial Information from Fourier Coefficient Distributions. The Journal of Membrane Biology, 241, 59-68.

    inputs:

        data,               An N dimensional array of real numbers

    outputs:
        probabilities,      An N dimensional array the same size as data

    """
    centered = data - np.mean(data)
    std = np.std(centered)

    # Calculate the exponent of the guassian
    exp = -1 * np.power(np.abs(centered), 2) / (2 * std ** 2)
    guassian = np.exp(exp) / (std * np.sqrt(2*np.pi))

    # The guassian transform reflects the point estimate of the probability 
    return guassian


def calculate_spatial_entropy(data: np.array, **kwargs) -> np.array:
    """
    Calculates the spatial en of each fourier coefficent of the array.
    Using the Shannon definition of entropy -p * log(p) and calculating p of each coefficent
    that would be expected if the array was randomly generated.

    inputs:

        data,               An N dimensional array of real numbers

    outputs:

        entropy,      An N dimensional array the same size as data
    """
    p = calculate_kspace_probability_array(data, **kwargs)
    return np.multiply(-p, np.log(p))


def kspace_information(data: np.array, **kwargs) -> float:
    """
    Calculates K Space en of an array in accordance with
    Heinz, W.F., Werbin, J.L., Lattman, E.E., & Hoh, J.H. (2009). Computing Spatial Information from Fourier Coefficient Distributions. The Journal of Membrane Biology, 241, 59-68.

    inputs:

        data,                   An N dimensional array of real numbers

    outputs:

        spatial_information,    A measure of spatial information
    """
    return np.sum(calculate_spatial_entropy(data, **kwargs))
