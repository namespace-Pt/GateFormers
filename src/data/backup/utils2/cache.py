import os
import logging
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool
from random import sample, choice
from transformers import AutoTokenizer
from torch.utils.data import Dataset, IterableDataset
from utils.util import load_pickle


cache_dir =




cache_dir = 