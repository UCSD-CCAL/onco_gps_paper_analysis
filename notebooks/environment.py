import sys
sys.path.insert(0, '../tools/')

from os import listdir

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import zipfile

import ccal
from helper.file import unzip

blue = '#4292C6'
green = '#41AB5D'
red = '#EF3B2C'
purple = '#807DBA'
black = '#000000'
white = '#FFFFFF' 
light_blue = '#0099CC'
