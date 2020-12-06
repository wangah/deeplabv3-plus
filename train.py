from tqdm import tqdm
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import data_loader.cityscapes as cityscapes
import model.deeplabv3plus as deeplabv3plus

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")