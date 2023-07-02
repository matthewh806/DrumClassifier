import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioClassifier(nn.Module):
    
    def __init__(self):
        super().__init__()
        conv_layers = []

        # 1st convolution block
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5,5), stride=(2,2), padding=(2,2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        nn.init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # 2nd convolution block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        nn.init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # 3rd convolution block
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        nn.init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # 4th convolution block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        nn.init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=10)

        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x = self.conv(x)

        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        x = self.lin(x)

        return x
