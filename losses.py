"""
Experiment to see if we can create a loc2vec as detailed in the blogpost.
bloglink: https://www.sentiance.com/2018/05/03/venue-mapping/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector1, triplet_selector2, triplet_selector3):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector1 = triplet_selector1
        self.triplet_selector2 = triplet_selector2
        self.triplet_selector3 = triplet_selector3

    def forward(self, embeddings, target):
        triplets = self.triplet_selector1.get_triplets(embeddings, target)
        if len(triplets) == 1:
            triplets = self.triplet_selector2.get_triplets(embeddings, target)
        if len(triplets) == 1:
            triplets = self.triplet_selector3.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = F.pairwise_distance(embeddings[triplets[:, 0]], embeddings[triplets[:, 1]])
        an_distances = F.pairwise_distance(embeddings[triplets[:, 0]], embeddings[triplets[:, 2]])
        # instead of taking AN distance, we check if PN distance is lower; if so, we use that.
        # This is logically equivalent to flipping the roles of the anchor and positive
        pn_distances = F.pairwise_distance(embeddings[triplets[:, 1]], embeddings[triplets[:, 2]])
        min_neg_dist = torch.min(an_distances, pn_distances)
        losses = F.relu(ap_distances - min_neg_dist + self.margin)

        # add the below for some amount of logging!
        # This prints the loss for the last batch alone.
        # if we need average per log session, we need to send actual values
        # and average it in train loop.
        np_losses = losses.cpu().data.numpy()
        np_ap_dist = ap_distances.cpu().data.numpy()
        np_an_dist = an_distances.cpu().data.numpy()
        np_min_dist = min_neg_dist.cpu().data.numpy()
        loss_summary_msg = "{:3.4f}\t {:3.4f}\t \
            {:3.2f}\t {:3.2f}\t {:3.2f}".format(np.max(np_losses),
                                                np.mean(np_losses),
                                                (np.mean(np_ap_dist)*1000),
                                                (np.mean(np_min_dist)*1000),
                                                (np.max(np_min_dist)*1000))
        # print(loss_summary_msg)
        return losses.mean(), len(triplets), loss_summary_msg

