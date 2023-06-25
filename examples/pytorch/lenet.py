
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os
import sys 

CLASSES_MODULE_PATH = "../../"
WEIGHT_FILE_PATH = "./"
MODELS_FOLDER = CLASSES_MODULE_PATH + 'models'

# appending a path
sys.path.append(CLASSES_MODULE_PATH) #CHANGE THIS LINE

from src.error_simulator_pytorch import Simulator

"""
This is a simple example on how to use the error simulator with PyTorch. The simulator has been developed as a custom
layer that can be added to any model. From a standard description of a LeNet5 model defined in class LeNet5 we 
modify the model by inserting our simulator as seen in class LeNet5Simulator at line 77. 
The three parameters that we pass to the simulator are 
1. operator_type: an element of the OperatorType enum that defines what kind of layer we are targeting
2. output shape of the layer as a string of format (None, channels, width, height)
3. models_folder: the path to the models folder, it can be changed to use different error models 

At lines 141 - 143 we manually copy the weights from the pre trained model to the new modified model and then we can
simply execute the inference obtaining the corrupted results of our simulation campaign

"""

RANDOM_SEED = 42
LEARNING_RATE = 0.001
BATCH_SIZE = 32
N_EPOCHS = 2

IMG_SIZE = 32
N_CLASSES = 10


class LeNet5(nn.Module):
    """
    A standard LeNet5 model
    """

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.tanh1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.tanh2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        self.tanh3 = nn.Tanh()
        self.linear1 = nn.Linear(in_features=120, out_features=84)
        self.tanh4 = nn.Tanh()
        self.linear2 = nn.Linear(in_features=84, out_features=n_classes)

    def forward(self, x):
        conv1 = self.conv1(x)
        x = self.tanh1(conv1)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.tanh2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.tanh3(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.tanh4(x)
        logits = self.linear2(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs, conv1


class LeNet5Simulator(nn.Module):
    """
    A standard LeNet5 model with the possibility of selecting a layer where to inject errors using CLASSES
    """

    def __init__(self, n_classes, selected_layer = None, conv_strategy = 'conv_gemm', fixed_spatial_class = None, fixed_domain_class = None):
        super(LeNet5Simulator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.tanh1 = nn.Tanh()
        self.conv_strategy = conv_strategy
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.tanh2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        self.tanh3 = nn.Tanh()
        self.linear1 = nn.Linear(in_features=120, out_features=84)
        self.tanh4 = nn.Tanh()
        self.linear2 = nn.Linear(in_features=84, out_features=n_classes)
        self.selected_layer = selected_layer
        self.fixed_spatial_class = fixed_spatial_class
        self.fixed_domain_class = fixed_domain_class
        self.simulator_output = None

    def __inject_error_if_requested(self, layer_num, operator_type, fixed_spatial_class = None, fixed_domain_class = None, verbose = False):
        def internal(layer_output):
            if self.selected_layer == layer_num:
                layer_output_shape = list(layer_output.shape)
                if len(layer_output_shape) < 4:
                    layer_output_shape = [1] * (4 - len(layer_output_shape)) + layer_output_shape
                layer_output_shape = str(tuple(layer_output_shape))
                if verbose:
                    print(f'Injecting after layer {self.selected_layer} ({operator_type}) with shape {layer_output_shape}')
                corrupted_layer_output = Simulator(operator_type, layer_output_shape, MODELS_FOLDER, fixed_spatial_class or self.fixed_spatial_class, fixed_domain_class or self.fixed_domain_class)(layer_output)
                self.simulator_output = corrupted_layer_output
                return corrupted_layer_output
            else:
                return layer_output

        return internal
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.__inject_error_if_requested(0, self.conv_strategy)(x)
        x = self.tanh1(x)
        x = self.__inject_error_if_requested(1, 'tanh')(x)
        x = self.pool1(x)
        x = self.__inject_error_if_requested(2, 'avgpool')(x)
        x = self.conv2(x)
        x = self.__inject_error_if_requested(3, self.conv_strategy)(x)
        x = self.tanh2(x)
        x = self.__inject_error_if_requested(4, 'tanh')(x)
        x = self.pool2(x)
        x = self.__inject_error_if_requested(5, 'avgpool')(x)
        x = self.conv3(x)
        # Force the spatial class to be bullet wake since the feature maps have size 1x1 at this layer
        x = self.__inject_error_if_requested(6, self.conv_strategy, fixed_spatial_class='bullet_wake')(x)
        x = self.tanh3(x)
        x = self.__inject_error_if_requested(7, 'tanh', fixed_spatial_class='bullet_wake')(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.tanh4(x)
        logits = self.linear2(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs, self.simulator_output


def load_datasets():
    '''
    Here we load and prepare the data, just a simple resize should
    be enough
    '''
    transf = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    # download and create datasets
    train_dataset = datasets.MNIST(root='mnist_data',
                                   train=True,
                                   transform=transf,
                                   download=True)

    valid_dataset = datasets.MNIST(root='mnist_data',
                                   train=False,
                                   transform=transf)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False)

    return train_dataset, valid_dataset, train_loader, valid_loader

def main():
    model = LeNet5(N_CLASSES).to('cpu')
    model.load_state_dict(torch.load(os.path.join(WEIGHT_FILE_PATH,'lenet.pth')))
    model_simulator = LeNet5Simulator(N_CLASSES, 4, fixed_domain_class={            
        "out_of_range": [
                100.0,
                100.0
                ]}).to('cpu')

    model.eval()
    model_simulator.eval()

    transf = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    test_dataset = datasets.MNIST('data', train=False, transform=transf, download=True)


    correct = 0
    fault_robust_runs = 0

    # Select a single image from the test dataset
    N_IMAGES = 1
    RUN_PER_IMAGE = 1
    for image_idx in range(N_IMAGES):
        img, label = test_dataset[image_idx]
        img = img.unsqueeze(0)
        output_vanilla = model(img)[0]
        pred_vanilla = output_vanilla.argmax(dim=1).item()
        for run in range(RUN_PER_IMAGE):
            output_corr = model_simulator(img)[0]
            print(f'The shape is {img.shape}')
            pred = output_corr.argmax(dim=1).item()


            print(f" Pred vs Label => ({pred},{label})")
            if pred == label:
                correct += 1
            
            if pred == pred_vanilla:
                fault_robust_runs+=1

    print(f"----------------------------------")
    print(f"Correctly predicted images: {correct} of {N_IMAGES * RUN_PER_IMAGE}")
    print(f"Predctions not changed w.r.t Vanilla model: {fault_robust_runs} of {N_IMAGES * RUN_PER_IMAGE}")


if __name__ == '__main__':
    main()