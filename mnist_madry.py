import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
from models import mnistCNN, LargeCNN
from helpers import adv_train, test_eval
from advertorch.attacks import LinfPGDAttack, GradientSignAttack, SpatialTransformAttack
import os

################
#    Methods   #
################

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    
################
#     Setup    #
################

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

datasetn="mnist"
config_path='config.json'
modeln = "madryCNN"

with open(config_path) as config_file:
    config = json.load(config_file)
    
# Set random seed
torch.manual_seed(config['random_seed'])

# Create Models
clean = mnistCNN()
adv = mnistCNN()

# clean = LargeCNN()
# adv = LargeCNN()


# Send to device for fast computation if GPU available
clean.to(device)
adv.to(device)

# Create Optimizer: Adam
clean_opt =  optim.Adam(clean.parameters(), lr=1e-4)
adv_opt =  optim.Adam(adv.parameters(), lr=1e-4)

# Create path name for loading the models. Follows the same convention
## as when creating path name for saving the models on training
clean_path = os.path.join(config['model_dir'], datasetn + '_' + modeln + '_clntrained.pt')
adv_path = os.path.join(config['model_dir'], datasetn + '_' + modeln + '_advtrained.pt')

################
#   Training   #
################

# Try to load the models from the file, otherwise train the model and save it to the path
print("Clean model")
try:
    load_model(clean, clean_path)
    print("Loaded")
except:
    # Create the file
    adv_train(clean, modeln, datasetn, clean_opt, device, config, False)

# Get adversarially trained model
print("Adversarially trained model")
try:
    load_model(adv, adv_path)
    print("Loaded")
except:
    train_adv = LinfPGDAttack(
                adv, loss_fn=nn.CrossEntropyLoss(reduction="sum"), 
                eps=config['epsilon'], nb_iter=config['num_steps'],
                eps_iter=config['step_size'],
                rand_init=config['random_start'], 
                clip_min=0.0, clip_max=1.0, targeted=False)
    adv_train(adv, modeln, datasetn, adv_opt, device, config, True, train_adv)

    
################
#    Testing   #
################

# Create one for each clean and adv trained network
# Adversary used to test the network (the difference is the k_eval)

test_adv_clean= LinfPGDAttack(
                clean, loss_fn=nn.CrossEntropyLoss(reduction="sum"), 
                eps=config['epsilon'], nb_iter=config['num_steps_eval'],
                eps_iter=config['step_size'],
                rand_init=config['random_start'], 
                clip_min=0.0, clip_max=1.0, targeted=False)
test_adv_adv= LinfPGDAttack(
                adv, loss_fn=nn.CrossEntropyLoss(reduction="sum"), 
                eps=config['epsilon'], nb_iter=config['num_steps_eval'],
                eps_iter=config['step_size'],
                rand_init=config['random_start'], 
                clip_min=0.0, clip_max=1.0, targeted=False)

print("%s against _" % modeln)
print("Clean")
test_eval(clean, datasetn, device, config['eval_batch_size'], True, test_adv_clean)
print("Adversarial")
test_eval(adv, datasetn, device, config['eval_batch_size'], True, test_adv_adv)

test_adv_clean= GradientSignAttack(
                clean, loss_fn=nn.CrossEntropyLoss(reduction="sum"), 
                eps=config['epsilon'], 
                clip_min=0.0, clip_max=1.0, targeted=False)
test_adv_adv= GradientSignAttack(
                adv, loss_fn=nn.CrossEntropyLoss(reduction="sum"), 
                eps=config['epsilon'], 
                clip_min=0.0, clip_max=1.0, targeted=False)


print("%s against _" % modeln)
print("Clean")
test_eval(clean, datasetn, device, config['eval_batch_size'], True, test_adv_clean)
print("Adversarial")
test_eval(adv, datasetn, device, config['eval_batch_size'], True, test_adv_adv)