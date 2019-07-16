import torch
from torch.utils.data import Dataset as dataset
from torchvision import datasets
from torchvision import transforms


class Dataset(dataset):
    def __init__(self, train=True):
        super(Dataset, self).__init__()
        if train == True:
            self.set = datasets.MNIST(root="./data", train=train, transform=transforms.ToTensor(), download=True)
        else:
            self.set = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download=True)

    def __len__(self):
        return self.set.__len__()

    def __getitem__(self, x):
        return self.set.__getitem__(x)


if __name__ == "__main__":
    test = Dataset(train=True)
    print((test.__getitem__(0))[0].shape)
