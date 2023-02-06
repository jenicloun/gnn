import itertools
from tqdm import tqdm
from glob import glob
import math
import json
from operator import mod
import os
import sys
import pickle
import sys
from dataclasses import dataclass
from copy import copy

import numpy as np
from sklearn.linear_model import LinearRegression
import torch
#import matplotlib
from torch.utils.data.sampler import Sampler
from torch_geometric.data import Data
from tqdm import tqdm


FILEPATH, _ = os.path.split(os.path.realpath(__file__))

def stream_id(stream_dict):
    return f'{stream_dict["name"]}{tuple(stream_dict["input_objects"])}->{tuple(stream_dict["output_objects"])}'



print(FILEPATH, _)








