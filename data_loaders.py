import torch
from torchvision import datasets, transforms

data_dir = './data/'

def get_train_loader(datasetn, batch_size):
    if datasetn is "mnist":
        return get_mnist_train_loader(batch_size)
    elif datasetn is "cifar10":
        return get_cifar10_train_loader(batch_size)

def get_test_loader(datasetn, batch_size):
    if datasetn is "mnist":
        return get_mnist_test_loader(batch_size)
    elif datasetn is "cifar10":
        return get_cifar10_test_loader(batch_size)

# from advertorch_examples.utils 
def get_mnist_train_loader(batch_size, shuffle=True):
    return torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=True, download=True,
               transform=transforms.ToTensor()), 
        batch_size=batch_size, shuffle=shuffle)

def get_mnist_test_loader(batch_size, shuffle=False):
    return torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=False, download=True,
                   transform=transforms.ToTensor()),
                   batch_size=batch_size,
                   shuffle=shuffle)

#### CIFAR 10 ####
def get_cifar10_train_loader(batch_size, shuffle=True):
#     transform = transforms.Compose([ transforms.ToTensor(), 
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.225, 0.225, 0.225])
#     ])
    transform = transforms.ToTensor()
    return torch.utils.data.DataLoader(
        datasets.CIFAR10(data_dir, train=True, download=True,
               transform=transform), 
        batch_size=batch_size, shuffle=shuffle)

def get_cifar10_test_loader(batch_size, shuffle=False):
    return torch.utils.data.DataLoader(
        datasets.CIFAR10(data_dir, train=False, download=True,
                   transform=transforms.ToTensor()), 
        batch_size=batch_size, shuffle=shuffle)
