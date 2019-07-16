import math
import torch
from torch import nn
import torch.nn.functional as F


class loss(nn.Module):
    def __init__(self):
        super(loss, self).__init__()
        self.func = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.func.forward(output, target)


if __name__ == "__main__":
    test = loss()
    output = torch.tensor([[0.1, 0.2, 0.1, 0.9, 0.1, 0., 0., 0., 0., 0.]])
    output.requires_grad = True
    # output=(output+1)/2
    target = torch.tensor([9])
    print(test(output, target))
