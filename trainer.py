"""
Experiment to see if we can create a loc2vec as detailed in the blogpost.
bloglink: https://www.sentiance.com/2018/05/03/venue-mapping/
"""
import torch
import numpy as np
from config import IMG_SIZE, CHECKPOINT_FILE_PREFIX

# For Mixed precision training
from apex import amp

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples:
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()
        filename = CHECKPOINT_FILE_PREFIX + str(epoch) + ".pth"

        # Train stage
        train_loss, non_zero_triplets = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        message += '\t{}: {:.0f}'.format('Average nonzero triplets', non_zero_triplets)

        # No Test/Validation stage as this is unsupervised learning
        # Technically should add it, but skip it for now!

        print(message)
        print(filename)
        torch.save(model, filename)



def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval):
    non_zero_triplets = []
    temp_non_zero_triplets = []
    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # Because of the way we generate data, each sample would actually
        #  generate 20 images. So a batch of them would have batchsize *20.
        # In our case, the data would be of shape [bs, 20, 3, IMG_SIZE, IMG_SIZE]
        # we want it to be [bs*20, 3, IMG_SIZE, IMG_SIZE]
        # similar modification for target too
        data = data.view(-1, 3, IMG_SIZE, IMG_SIZE)
        target = target.view(-1)
        if cuda:
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        outputs = model(data)
        loss_outputs = loss_fn(outputs, target)

        loss = loss_outputs[0]
        losses.append(loss.item())
        total_loss += loss.item()

        # For Mixed precision training
        # loss.backward()
        # optimizer.step()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        # End mixed precision training changes

        non_zero_triplets.append(loss_outputs[1])
        temp_non_zero_triplets.append(loss_outputs[1])

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * (data.shape[0] / 20), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            message += '\t{}: {:.0f}'.format('Average nonzero triplets',
                    np.mean(temp_non_zero_triplets))
            message += ' ' + loss_outputs[2]
            # Reset it so that we can know intermediate
            # progress
            temp_non_zero_triplets = []
            # print(loss_outputs[2])
            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, np.mean(non_zero_triplets)



