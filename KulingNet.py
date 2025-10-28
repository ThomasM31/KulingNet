import numpy as np, pandas as pd
import os 
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

# Where we are right now, absolute pathing
base_dir = os.path.dirname(__file__)
images_file_path = os.path.join(base_dir, "CCSN_Image_set")
# example path for cirrus clouds
ci_file_path = os.path.join(images_file_path, "Ci")


class KulingNet(nn.Module):
    pass
