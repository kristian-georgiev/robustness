'''Simple conv model
'''
import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=3,
                              kernel_size=5, stride=2)
        self.linear = nn.Linear(432, num_classes)

    def forward(self, x, with_latent=False, fake_relu=None, no_relu=None):
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        final = self.linear(x)
        if with_latent:
            return final, x
        return final

convnet = ConvNet

def test():
    net = ConvNet()
    y = net(torch.randn(5, 1, 28, 28))
    print(y.size())
