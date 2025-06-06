
import torch
import torch.nn as nn 
import math

class MarginalNCE(nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.opt=opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.bool #torch.uint8 if version.parse(torch.__version__)
        self.cosm = math.cos(0.25)
        self.sinm = math.sin(0.25)

    def forward(self, feat_q, feat_k):

        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        cosx = l_pos
        sinx = torch.sqrt(1.0 - torch.pow(cosx, 2))
        l_pos = cosx * self.cosm - sinx * self.sinm

        batch_dim_for_bmm = max(1,int(batchSize / 64))
        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # just fill the diagonal with very small number, which is exp(-10)
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device)).mean()

        return loss