import copy
import gzip
import itertools
import os
import pickle
import re
import sys
from collections import OrderedDict
from pprint import pprint

import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, '../tools')
print('Added \'../tools\' to the path.')
from ccal import ccal
