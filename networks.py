"""
Experiment to see if we can create a loc2vec as detailed in the blogpost.
bloglink: https://www.sentiance.com/2018/05/03/venue-mapping/
"""
from collections import OrderedDict
from torch import nn
from torchvision import models

class Loc2Vec(nn.Module):
    """
    A pretrainned model with a linear layer replaced at the end.
    Tested with Resnet50 and Resnet18
    """

    def __init__(self):
        super(Loc2Vec, self).__init__()
        self.model = models.resnet50(pretrained=True)
        # Experimented with
        # for parameter in self.model.parameters():
        #     parameter.requires_grad = False
        self.model.avgpool = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
        # Resnet 18
        num_ftrs = 4608 # arrived at by looking at errors!

        # Resnet 50
        num_ftrs = 18432 # arrived at by looking at errors!
        self.model.fc = nn.Linear(num_ftrs, 16)

    def forward(self, x):
        x = self.model(x)
        return x

    def unfreeze():
        for params in self.model.parameters():
            params.require_grad = True

    def freeze():
        for params in self.model.parameters():
            params.require_grad = False

class Loc2VecDNet(nn.Module):
    """
    A pretrainned model with a classifier added to the end.
    This model works with Densenet, but not with different input size
    compared to the imagenet.

    Did not get encouraging results with freezing the layers.
    No experimentation done with whole network as image size
    could not be reduced.
    """

    def __init__(self):
        super(Loc2VecDNet, self).__init__()
        self.model = models.densenet121(pretrained=True)

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        num_ftrs = self.model.classifier.in_features

        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(num_ftrs, 1024)),
            ('drop1', nn.Dropout(p=0.5)),
            ('relu1', nn.PReLU()),
            ('fc2', nn.Linear(1024, 256)),
            # ('drop2', nn.Dropout(p=0.5)),
            # ('relu2', nn.PReLU()),
            # ('fc3', nn.Linear(512, 256)),
            ('drop3', nn.Dropout(p=0.5)),
            ('relu3', nn.PReLU()),
            ('fc4', nn.Linear(256, 16)),
        ]))
        self.model.classifier = classifier

    def forward(self, x):
        x = self.model(x)
        return x