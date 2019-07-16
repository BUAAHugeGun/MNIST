import torch
import math
import torch.nn as nn


class model(nn.Module):
    def __init__(self, in_channels=1, classes=10, use_bn=True):
        super(model, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.use_bn = use_bn
        self.build()
        self.init()

    def conv_layer(self, in_channels, out_channels, kernel, stride=1, padding=0, use_bn=True):
        layers = []
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride,
                                padding=padding, ))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def pool_layer(self, in_channels, kernel, stride, padding=0, mode="Max"):
        if mode != "Max" and mode != "Avg":
            assert 0
        if mode == "Max":
            return nn.Sequential(nn.MaxPool2d(kernel_size=kernel, stride=stride))
        else:
            return nn.Sequential(nn.AvgPool2d(kernel_size=kernel, stride=stride, padding=padding))

    def fc_layer(self, in_channels, out_channels, dropout=0.5):
        layers = []
        layers.append(nn.Linear(in_channels, out_channels))
        if dropout != 0:
            layers.append(nn.Dropout2d(p=dropout))
        return nn.Sequential(*layers)

    def build(self):
        self.conv1 = self.conv_layer(self.in_channels, 2, 3, 1, 1)
        self.mp1 = self.pool_layer(2, kernel=2, stride=2, mode="Max")
        self.conv21 = self.conv_layer(2, 4, 3, 1, 1)
        self.conv22 = self.conv_layer(4, 4, 3, 1, 1)
        self.mp2 = self.pool_layer(4, kernel=2, stride=2, mode="Max")
        self.conv31 = self.conv_layer(4, 8, 3, 1, 1)
        self.conv32 = self.conv_layer(8, 16, 3, 1, 1)
        self.fc = self.fc_layer(784, self.classes, dropout=0)

    def init(self, scale_factor=1.0, mode="FAN_IN"):
        if mode != "FAN_IN" and mode != "FAN_OUT":
            assert 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if mode == "FAN_IN":
                    n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                    m.weight.data.normal_(0, math.sqrt(scale_factor / n))
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(scale_factor / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.mp1(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.mp2(x)
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.fc(x.view(x.shape[0],-1))
        return x


if __name__ == "__main__":
    x = torch.randn([4, 1, 28, 28])
    test = model()
    print(test(x))
