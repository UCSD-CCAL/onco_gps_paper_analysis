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

from repositories.helper.helper import df
from repositories.helper.helper import file
from repositories.file.file import gct
from repositories.plot.plot import plot
from repositories.plot.plot import plot_nmf
from repositories.match.match import match_panel, comparison_panel
from repositories.oncogps.oncogps import oncogps

BLACK = '#000000'
WHITE = '#FFFFFF' 
PURPLE = '#807DBA'
BLUE = '#4292C6'
LIGHT_BLUE = '#0099CC'
GREEN = '#41AB5D'
RED = '#EF3B2C'