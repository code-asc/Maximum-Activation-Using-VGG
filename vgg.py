import torch.nn as nn
import torch.nn.functional as F
import torch

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.conv_layers_ = self.get_vgg_model(cfg[vgg_name])
        self.fc_layer_ = nn.Linear(512, 10)

    def forward(self, x):
        out = self.conv_layers_(x)
        out = out.view(out.size(0), -1)
        out = self.fc_layer_(out)
        return out

    def get_vgg_model(self, pattern):

        layers = []
        channel_size = 3

        for x in pattern:
            if x == 'M':
                layers +=[nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(channel_size, x, kernel_size=3, padding=1),
                            nn.ReLU()]
                channel_size = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
