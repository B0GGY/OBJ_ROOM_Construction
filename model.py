import torch
import torch.nn as nn

class Adaptive_selector(nn.Module):
    def __init__(self, inchannel=144, outchannel=1):
        super(Adaptive_selector, self).__init__()
        # self.adaptive_layer1 = nn.Linear(inchannel, inchannel, bias=False)
        # self.adaptive_layerpre = nn.Linear(144, 144, bias=False)
        # self.relu = nn.ReLU(inplace=True)
        self.adaptive_layer = nn.Linear(inchannel, outchannel, bias=False)

        # self.relu = nn.LeakyReLU(inplace=True)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x = self.adaptive_layer1(x)
        # out = self.adaptive_layerpre(x)
        # out = self.relu(out)
        out = self.adaptive_layer(x)

        # out = self.relu(out)
        return out


if __name__ == '__main__':

    test = Adaptive_selector()
    inpt = torch.ones((4,512, 144))
    output = test(inpt)


    print(output.shape)



