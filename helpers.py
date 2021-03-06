import torch
import torch.nn.functional as F
import os
from data_loaders import get_train_loader, get_test_loader
from advertorch.context import ctx_noparamgrad_and_eval
import time

def adv_train(model, modeln, datasetn, optimizer, device, config, flag_advtrain, train_adversary=None, scheduler=None):
    # Get training and testing
    train_loader = get_train_loader(datasetn,
                                batch_size=config['training_batch_size'])
#     test_loader = get_test_loader(datasetn,
#         batch_size=config['eval_batch_size'])

    # Set the saved model filename and set the # epochs
    if flag_advtrain:
        nb_epoch = config['num_advtr_epoch']
        model_filename = datasetn + "_" + modeln + "_advtrained.pt"
        if train_adversary:
            adversary = train_adversary 
    else:
        nb_epoch = config['num_cln_epoch']
        model_filename = datasetn + "_" + modeln + "_clntrained.pt"

    # Start training
    for epoch in range(nb_epoch):
        model.train()
        if scheduler:
            scheduler.step()
            
        start = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            if flag_advtrain:
                # when attacking, the model needs to be in eval mode
                # also the parameters should be accumulating gradients
                with ctx_noparamgrad_and_eval(model):
                    data = adversary.perturb(data, target)

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target, reduction='mean')
            loss.backward()
            optimizer.step()
            
            
            if batch_idx % config['log_interval'] == 0:
                print("\tTime elapsed: %s" % (time.time() - start))
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

        """
        model.eval()
        if flag_advtrain:
            test_eval(model, datasetn, device, config['eval_batch_size'],
                     flag_advtrain, adversary)
        else:
            test_eval(model, datasetn, device, 
                     config['eval_batch_size'], flag_advtrain)
        """
        
#         test_clnloss = 0
#         clncorrect = 0

#         if flag_advtrain:
#             test_advloss = 0
#             advcorrect = 0

#         for clndata, target in test_loader:
#             clndata, target = clndata.to(device), target.to(device)
#             with torch.no_grad():
#                 output = model(clndata)
#             test_clnloss += F.cross_entropy(
#                 output, target, reduction='sum').item()
#             pred = output.max(1, keepdim=True)[1]
#             clncorrect += pred.eq(target.view_as(pred)).sum().item()

#             if flag_advtrain:
#                 advdata = adversary.perturb(clndata, target)
#                 with torch.no_grad():
#                     output = model(advdata)
#                 test_advloss += F.cross_entropy(
#                     output, target, reduction='sum').item()
#                 pred = output.max(1, keepdim=True)[1]
#                 advcorrect += pred.eq(target.view_as(pred)).sum().item()

#         test_clnloss /= len(test_loader.dataset)
#         print('\nTest set: avg cln loss: {:.4f},'
#               ' cln acc: {}/{} ({:.0f}%)\n'.format(
#                   test_clnloss, clncorrect, len(test_loader.dataset),
#                   100. * clncorrect / len(test_loader.dataset)))
#         if flag_advtrain:
#             test_advloss /= len(test_loader.dataset)
#             print('Test set: avg adv loss: {:.4f},'
#                   ' adv acc: {}/{} ({:.0f}%)\n'.format(
#                       test_advloss, advcorrect, len(test_loader.dataset),
#                       100. * advcorrect / len(test_loader.dataset)))

    # Python 3.2+: recursively creates the directory and does not raise exception if exists already
    os.makedirs(config['model_dir'], exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join(config['model_dir'], model_filename))

def test_eval(model, datasetn, device, eval_batch_size, flag_adv, adversary=None):
    test_loader = get_test_loader(datasetn,
        batch_size=eval_batch_size)

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
