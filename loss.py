import torch.nn as nn
import torch
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3, semi_hard = True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.semi_hard = semi_hard

    def forward(self, a, p, n):
        if self.semi_hard:
            # semi-hard triplet
            dist_ap = torch.diag(a@p.T)
            # dist_an = torch.diag(a@n.T)
            an = a@n.T
            dist_an = []
            for i in range(a.size(0)):
                if an[i, an[i]<(dist_ap[i]+self.margin)].size(0) != 0:
                    dist_an.append(an[i, an[i]<(dist_ap[i]+self.margin)][0].unsqueeze(0))
                else:
                    dist_an.append(an[i,i].unsqueeze(0))
            # print(dist_an)
            dist_an = torch.cat(dist_an)
            y = torch.ones_like(dist_ap)
            return self.ranking_loss(dist_ap, dist_an, y)# + 0.1*1/torch.diag(p@p.T).mean()
        else:
            dist_ap = torch.diag(a @ p.T)
            dist_an = torch.diag(a@n.T)
            loss = F.relu(dist_an - dist_ap + self.margin)
            return loss.mean()
