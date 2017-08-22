import os
import sys
sys.path.insert(0, '../tools/')

import numpy as np
import pandas as pd


from repositories.helper.helper import df
from repositories.helper.helper import file

from repositories.file.file import gct

from repositories.plot.plot import plot
from repositories.plot.plot import plot_nmf

from repositories.match.match import match_panel
from repositories.match.match import comparison_panel

from repositories.oncogps.oncogps import oncogps

BLACK = '#000000'
WHITE = '#FFFFFF' 
PURPLE = '#807DBA'
BLUE = '#4292C6'
LIGHT_BLUE = '#0099CC'
GREEN = '#41AB5D'
RED = '#EF3B2C'