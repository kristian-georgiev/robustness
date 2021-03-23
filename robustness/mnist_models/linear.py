'''Simple linear model
'''
import torch
import torch.nn as nn

class LinearNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(28 ** 2, num_classes)

    def forward(self, x, with_latent=False, fake_relu=None, no_relu=None):
        final = self.linear(x)
        if with_latent:
            return final, final  # no intermediate latent repr here
        return final

linearnet = LinearNet

def test():
    net = LinearNet()
    y = net(torch.randn(5, 1, 28, 28))
    print(y.size())
