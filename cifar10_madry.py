# %%capture output
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import json
from models import cifar10CNN
from helpers import adv_train, test_eval
from advertorch.attacks import LinfPGDAttack
import os
    
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

with open('config.json') as config_file:
    config = json.load(config_file)
    
modeln = "madryCNN"
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

clean = Model()
adv = Model()

clean_path = os.path.join(config['model_dir'], 'mnist_madryCNN_clntrained.pt')
adv_path = os.path.join(config['model_dir'], 'mnist_madryCNN_advtrained.pt')

# Try to load the models from the file, otherwise train the model and save it to the path
try:
    clean.load_state_dict(torch.load(clean_path))
except:
    pass

try:
    adv.load_state_dict(torch.load(adv_path))
except:
    pass

clean.to(device)
adv.to(device)
clean.eval()
adv.eval()