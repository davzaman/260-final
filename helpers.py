import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from torchvision import datasets, transforms
from advertorch.context import ctx_noparamgrad_and_eval

data_dir = './data/'

# Modified from: from advertorch_examples.utils 
def get_mnist_train_loader(batch_size, shuffle=True):
    return torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=True, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ])), batch_size=batch_size, shuffle=shuffle)

# Modified from: from advertorch_examples.utils 
def get_mnist_test_loader(batch_size, shuffle=False):
    return torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])), batch_size=batch_size, shuffle=shuffle)

def adv_train(model, modeln, optimizer, device, config, flag_advtrain, train_adversary=None):
    # Set random seed
    torch.manual_seed(config['random_seed'])
    
    # Get training and testing
    train_loader = get_mnist_train_loader(
        batch_size=config['training_batch_size'], shuffle=True)
    test_loader = get_mnist_test_loader(
        batch_size=config['eval_batch_size'], shuffle=False)

    # Set the saved model filename and set the # epochs
    if flag_advtrain:
        nb_epoch = config['num_advtr_epoch']
        model_filename = "mnist_" + modeln + "_advtrained.pt"
        if train_adversary:
            adversary = train_adversary 
    else:
        nb_epoch = config['num_cln_epoch']
        model_filename = "mnist_" + modeln + "_clntrained.pt"

    # Start training
    for epoch in range(nb_epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            if flag_advtrain:
                # when performing attack, the model needs to be in eval mode
                # also the parameters should be accumulating gradients
                with ctx_noparamgrad_and_eval(model):
                    data = adversary.perturb(data, target)

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target, reduction='mean')
            loss.backward()
            optimizer.step()
            if batch_idx % config['log_interval'] == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        model.eval()
#         adv_eval(model, device, adversary, config['eval_batch_size'], flag_advtrain)
        test_clnloss = 0
        clncorrect = 0

        if flag_advtrain:
            test_advloss = 0
            advcorrect = 0

        for clndata, target in test_loader:
            clndata, target = clndata.to(device), target.to(device)
            with torch.no_grad():
                output = model(clndata)
            test_clnloss += F.cross_entropy(
                output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            clncorrect += pred.eq(target.view_as(pred)).sum().item()

            if flag_advtrain:
                advdata = adversary.perturb(clndata, target)
                with torch.no_grad():
                    output = model(advdata)
                test_advloss += F.cross_entropy(
                    output, target, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]
                advcorrect += pred.eq(target.view_as(pred)).sum().item()

        test_clnloss /= len(test_loader.dataset)
        print('\nTest set: avg cln loss: {:.4f},'
              ' cln acc: {}/{} ({:.0f}%)\n'.format(
                  test_clnloss, clncorrect, len(test_loader.dataset),
                  100. * clncorrect / len(test_loader.dataset)))
        if flag_advtrain:
            test_advloss /= len(test_loader.dataset)
            print('Test set: avg adv loss: {:.4f},'
                  ' adv acc: {}/{} ({:.0f}%)\n'.format(
                      test_advloss, advcorrect, len(test_loader.dataset),
                      100. * advcorrect / len(test_loader.dataset)))

    # Python 3.2+: recursively creates the directory and does not raise exception if exists already
    os.makedirs(config['model_dir'], exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join(config['model_dir'], model_filename))

def adv_eval(model, device, adversary, eval_batch_size, flag_adv):
    test_loader = get_mnist_test_loader(
        batch_size=eval_batch_size, shuffle=False)

    test_clnloss = 0
    clncorrect = 0

    if flag_adv:
        test_advloss = 0
        advcorrect = 0

    for clndata, target in test_loader:
        clndata, target = clndata.to(device), target.to(device)
        with torch.no_grad():
            output = model(clndata)
        test_clnloss += F.cross_entropy(
            output, target, reduction='sum').item()
        pred = output.max(1, keepdim=True)[1]
        clncorrect += pred.eq(target.view_as(pred)).sum().item()

        if flag_adv:
            advdata = adversary.perturb(clndata, target)
            with torch.no_grad():
                output = model(advdata)
            test_advloss += F.cross_entropy(
                output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            advcorrect += pred.eq(target.view_as(pred)).sum().item()
        
    test_clnloss /= len(test_loader.dataset)
    print('\nTest set: avg cln loss: {:.4f},'
          ' cln acc: {}/{} ({:.0f}%)\n'.format(
              test_clnloss, clncorrect, len(test_loader.dataset),
              100. * clncorrect / len(test_loader.dataset)))
    if flag_adv:
        test_advloss /= len(test_loader.dataset)
        print('Test set: avg adv loss: {:.4f},'
              ' adv acc: {}/{} ({:.0f}%)\n'.format(
                  test_advloss, advcorrect, len(test_loader.dataset),
                  100. * advcorrect / len(test_loader.dataset)))
