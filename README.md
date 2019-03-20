# Adversarial Training
Requires advertorch and pytorch.

- `cifar10_madry.py` - The main driver for models on CIFAR10 dataset. 
- `mnist_madry.py`- Main driver for models on MNIST dataset.
- `models.py` - Defines different models. Browse to explore different model options to use in the main driver python models. Includes various VGG structures, Wide ResNet, and two other CNNs.
- `data_loaders.py` - Loads either MNIST or CIFAR10 data using pytorch.
- `helpers.py` - Contains logic for the training and testing/evaluation of the models.
- `cifar_config.json` - Contains the parameters for the Madry paper on the CIFAR10 dataset.
- `config.json` - Contains the parameters for the Madry paper on the MNIST dataset.
- `largeccn-config.json` - Contains the same parameters as Madry but model directory is changed so the  model is saved elsewhere, as it is  for a different architecture.

## Running Code
`python cifar10_madry.py` will run as is, using `cifar_config.json`. Similarly, `python mnist_madry.py` will run using `config.json`. 

If you want to try a different model you will need to go into the respective python files and change the config file and actual model created in the code before running. The  config file will contain the path to the saved model if it exists, if it doesn't exist it will immediately begin training a model. It is important to keep this in mind, since if you change the model, but not the config file, it will give you an error since the weights are different.  For example in `mnist_madry.py`, if you switch from `clean = mnistCNN()` to `clean = LargeCNN()` (and similarly for `adv`) then you must also change `config_path='largecnn-config.json'`. 

If you would like to run the same model but under different parameters, just change the  config  file. However, if the  model is already  trained and you want to change the parameters that it was trained on, you must either remove the models saved to `/models/modeln` specified in the config, or move it somewhere else.