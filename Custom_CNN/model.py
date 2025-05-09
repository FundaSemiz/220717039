import torch
import torch.nn as nn #the library used for neural networks
import config

#creating cutom model on pytorch
class SimpleCNN(nn.Module):

    def __init__(self, num_classes=config.NUM_CLASSES, img_size=128):
        #calls the constructor of its parent class(nn.Module)
        super(SimpleCNN, self).__init__()
        #in case of image size not being correct
        if img_size != 128:
             print(f"Warning: SimpleCNN expects img_size=128 for default flatten calculation, received {img_size}. Adjust FC layer if needed.")
        #applying convolution-relu-pooling
        #in_channels is 3 since we have an rgb image
        #out_channels is number of filters it learns
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # img_size // 2

        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # img_size // 4

        # Layer 3 (Middle Layer)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # img_size // 8

        # Layer 4
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # img_size // 16

        # Layer 5 (Deepest Conv Layer)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2) # img_size // 32

        # calculation of the size it will have after flattening
        final_map_size = img_size // 32
        flattened_size = 512 * final_map_size * final_map_size

        # fully connected layer
        self.fc1 = nn.Linear(flattened_size, num_classes)

        #dictionary for names of layers for visualization
        self.conv_layers_for_viz = {
            'conv1': self.conv1, # 1'st layer
            'conv3': self.conv3, # Middle (3'rd) layer
            'conv5': self.conv5  # Deepest (5'th) layer
        }
    #
    def forward(self, x):
        #calling layers one by one
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        x = self.pool5(self.relu5(self.conv5(x)))
        #flattens to get single dimensional output
        x = torch.flatten(x, 1)
        #passes the flat vector through the last layer
        x = self.fc1(x)
        return x