import sys
sys.path.insert(0, '../tools/')

import numpy as np
import pandas as pd

from file.file.gct import read_gct, write_gct
from match.match.match_panel import make_match_panel, make_summary_match_panel
from helper.helper.df import get_top_and_bottom_indices
from plot.plot.plot import plot_heatmap
from plot.plot.plot_nmf import plot_nmf
from oncogps.oncogps.oncogps import define_components
from match.match.comparison_panel import make_comparison_panel

blue = '#4292C6'
green = '#41AB5D'
red = '#EF3B2C'
purple = '#807DBA'
black = '#000000'
white = '#FFFFFF' 
light_blue = '#0099CC'