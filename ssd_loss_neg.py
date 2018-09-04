from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torchcv.utils import one_hot_embedding
import time

class SSDLossNegWeights(nn.Module):
    def __init__(self, num_classes, neg_ind):
        super(SSDLossNegWeights, self).__init__()
        self.num_classes = num_classes
        self.neg_ind = neg_ind
        weights = [1.]* num_classes
        weights[neg_ind] = weights[neg_ind] / 2.6 # Wyn Mew
        self.classes_weights = torch.FloatTensor(weights).cuda()

    def _hard_negative_mining(self, cls_loss, pos):
        '''Return negative indices that is 3x the number as postive indices.

        Args:
          cls_loss: (tensor) cross entroy loss between cls_preds and cls_targets, sized [N,#anchors].
          pos: (tensor) positive class mask, sized [N,#anchors].

        Return:
          (tensor) negative indices, sized [N,#anchors].
        '''
        cls_loss = cls_loss * (pos.float() - 1)

        _, idx = cls_loss.sort(1)  # sort by negative losses
        _, rank = idx.sort(1)      # [N,#anchors]

        num_neg = 3*pos.long().sum(1)  # [N,]
        neg = rank < num_neg[:,None]   # [N,#anchors]
        return neg

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [N, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [N, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [N, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [N, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + CrossEntropyLoss(cls_preds, cls_targets).
        '''
        pos = cls_targets > 0  # [N,#anchors]
        batch_size = pos.size(0)

        # zero out neg idx loc loss
        for i in range(batch_size):
            if cls_targets.cpu().data[i][0] == self.neg_ind:
                loc_preds[i] = loc_targets[i]

        num_pos = pos.data.long().sum()

        #===============================================================
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        #===============================================================
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]


        loc_loss = F.smooth_l1_loss(loc_preds[mask], loc_targets[mask], size_average=False)

        #===============================================================
        # cls_loss = CrossEntropyLoss(cls_preds, cls_targets)
        #===============================================================
        #cls_loss = F.cross_entropy(cls_preds.view(-1,self.num_classes), cls_targets.view(-1), reduce=False)  # [N*#anchors,]
        cls_loss = F.cross_entropy(cls_preds.view(-1, self.num_classes), cls_targets.view(-1),reduce=False
                                   , weight=self.classes_weights)  # [N*#anchors,]

        cls_loss = cls_loss.view(batch_size, -1)
        cls_loss[cls_targets<0] = 0  # set ignored loss to 0
        neg = self._hard_negative_mining(cls_loss, pos)  # [N,#anchors]
        cls_loss = cls_loss[pos|neg].sum()

        if num_pos >0:
            self.loss = (loc_loss+cls_loss)/num_pos
            print('loc_loss: %.3f | cls_loss: %.3f' \
                  % (loc_loss.data[0] / num_pos, cls_loss.data[0] / num_pos), end=' | ')
        else:
            self.loss = (loc_loss + cls_loss)
            print('loc_loss: %.3f | cls_loss: %.3f' \
                  % (loc_loss.data[0] / 1, cls_loss.data[0] / 1), end=' | ')

        return self.loss


class SSDLossNeg(nn.Module):
    def __init__(self, num_classes, neg_ind):
        super(SSDLossNeg, self).__init__()
        self.num_classes = num_classes
        self.neg_ind = neg_ind

    def _hard_negative_mining(self, cls_loss, pos):
        '''Return negative indices that is 3x the number as postive indices.

        Args:
          cls_loss: (tensor) cross entroy loss between cls_preds and cls_targets, sized [N,#anchors].
          pos: (tensor) positive class mask, sized [N,#anchors].

        Return:
          (tensor) negative indices, sized [N,#anchors].
        '''
        cls_loss = cls_loss * (pos.float() - 1)

        _, idx = cls_loss.sort(1)  # sort by negative losses
        _, rank = idx.sort(1)      # [N,#anchors]

        num_neg = 3*pos.long().sum(1)  # [N,]
        neg = rank < num_neg[:,None]   # [N,#anchors]
        return neg

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [N, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [N, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [N, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [N, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + CrossEntropyLoss(cls_preds, cls_targets).
        '''
        pos = cls_targets > 0  # [N,#anchors]
        batch_size = pos.size(0)

        # zero out neg idx loc loss
        for i in range(batch_size):
            if cls_targets.cpu().data[i][0] == self.neg_ind:
                loc_preds[i] = loc_targets[i]

        num_pos = pos.data.long().sum()

        #===============================================================
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        #===============================================================
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]


        loc_loss = F.smooth_l1_loss(loc_preds[mask], loc_targets[mask], size_average=False)

        #===============================================================
        # cls_loss = CrossEntropyLoss(cls_preds, cls_targets)
        #===============================================================
        cls_loss = F.cross_entropy(cls_preds.view(-1,self.num_classes), \
                                   cls_targets.view(-1), reduce=False)  # [N*#anchors,]
        cls_loss = cls_loss.view(batch_size, -1)
        cls_loss[cls_targets<0] = 0  # set ignored loss to 0
        neg = self._hard_negative_mining(cls_loss, pos)  # [N,#anchors]
        cls_loss = cls_loss[pos|neg].sum()

        if num_pos >0:
            self.loss = (loc_loss+cls_loss)/num_pos
            print('loc_loss: %.3f | cls_loss: %.3f' \
                  % (loc_loss.data[0] / num_pos, cls_loss.data[0] / num_pos), end=' | ')
        else:
            self.loss = (loc_loss + cls_loss)
            print('loc_loss: %.3f | cls_loss: %.3f' \
                  % (loc_loss.data[0] / 1, cls_loss.data[0] / 1), end=' | ')

        return self.loss