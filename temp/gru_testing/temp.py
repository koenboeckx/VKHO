"""
From https://blog.floydhub.com/gru-with-pytorch/
"""

import os
import time

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm_notebook
from sklearn.preprocessing import MinMaxScaler

# Define data root directory
data_dir = "./data"