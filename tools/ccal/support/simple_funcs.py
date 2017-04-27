
"""
Computational Cancer Analysis Library

Authors:
    Huwate (Kwat) Yeerna (Medetgul-Ernar)
        kwat.medetgul.ernar@gmail.com
        Computational Cancer Analysis Laboratory, UCSD Cancer Center

    Pablo Tamayo
        ptamayo@ucsd.edu
        Computational Cancer Analysis Laboratory, UCSD Cancer Center
"""

import gzip
from os import mkdir, listdir, environ
from os.path import abspath, split, isdir, isfile, islink, join
from sys import platform

from Bio import bgzf
from pandas import Series, DataFrame, read_csv, concat

from .str_ import split_ignoring_inside_quotes, remove_nested_quotes


# ======================================================================================================================
# General functions
# ======================================================================================================================

def extract_top_bottom_features(matrix, n_up_features, n_dn_features):
    """
    Extract top and bottom features from a dataframe
    :param matrix: DataFrame
    :param n_up_features: number of up features
    :param n_dn_features: number of down features
    """
    N = matrix.shape[0]
    up_features = list(range(0, n_up_features))
    dn_features = list(range(N - n_dn_features, N))
    all_features = up_features + dn_features

    return matrix.ix[all_features, :].index
