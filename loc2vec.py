"""
Experiment to see if we can create a loc2vec as detailed in the blogpost.
bloglink: https://www.sentiance.com/2018/05/03/venue-mapping/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
from torchvision import transforms

# For Mixed precision training
from apex import amp

# Set up the network and training parameters
from trainer import fit
# Strategies for selecting triplets within a minibatch
from utils import HardestNegativeTripletSelector
from utils import RandomNegativeTripletSelector, SemihardNegativeTripletSelector

from datasets import GeoTileDataset
from networks import Loc2Vec
from losses import OnlineTripletLoss
from config import IMG_SIZE, LOG_INTERVAL, N_EPOCHS, BATCH_SIZE, MARGIN, TILE_FILE

def main():
    cuda = torch.cuda.is_available()


    anchor_transform = transforms.Compose([
                transforms.RandomAffine(degrees=90, translate=(0.25, 0.25)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.CenterCrop(128),
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

    train_transforms = transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

    #  Let's use 12 while developing as it reduces the start time.
    dset_train = GeoTileDataset(TILE_FILE,
                                transform=train_transforms,
                                center_transform=anchor_transform)

    pd_files = dset_train.get_file_df()
    weights = pd_files.frequency
    train_sampler = WeightedRandomSampler(weights , len(dset_train))
    # Should numworkers be 1?
    kwargs = {'num_workers': 8, 'pin_memory': True} if cuda else {}
    online_train_loader = DataLoader(dset_train, batch_size=BATCH_SIZE,
                                     sampler=train_sampler,
                                     **kwargs)

    model = Loc2Vec()
    if cuda:
        model.cuda()

    loss_fn = OnlineTripletLoss(MARGIN,
                                HardestNegativeTripletSelector(MARGIN),
                                SemihardNegativeTripletSelector(MARGIN),
                                RandomNegativeTripletSelector(MARGIN))

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, 16, gamma=0.1, last_epoch=-1)

    # Mixed precision training
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)

    fit(online_train_loader, online_train_loader, model, loss_fn, optimizer, scheduler,
        N_EPOCHS, cuda, LOG_INTERVAL)

if __name__ == "__main__":
    main()
