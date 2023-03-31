from hashlib import new
import torch
import torch.nn as nn

class PhaseAggregationLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, args,num_classes=6, feat_dim=64, use_gpu=True):
        super(PhaseAggregationLoss, self).__init__()
        self.num_classes = args.class_number-1
        self.feat_dim = args.center_dimension
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.centers = nn.Parameter(torch.eye(self.num_classes).cuda())
        else:
            self.centers = nn.Parameter(torch.eye(self.num_classes))
    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat=torch.addmm(input=distmat,mat1=x, mat2=self.centers.t(),beta=1,alpha=-2 )

        # new_distmat=torch.ones_like(distmat)
        # for i in range(batch_size):
        #     new_distmat[i]=distmat[i]/amplitude[i]

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = 0.5*dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss,self.centers