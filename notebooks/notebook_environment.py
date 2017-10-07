import copy
import itertools
import os
import pickle
import re
import sys
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns

sys.path.insert(0, '../tools/')
print('Added \'../tools/\' to the path.')
from ccal import ccal

BLACK = '#000000'
WHITE = '#FFFFFF' 
PURPLE = '#807DBA'
BLUE = '#4292C6'
LIGHT_BLUE = '#0099CC'
GREEN = '#41AB5D'
RED = '#EF3B2C'