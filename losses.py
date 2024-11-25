'''
    定义了两个损失函数：ArcFaceLoss和FocalLoss
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


'''
    FocalLoss，其中：
    gamma：衰减速度，gammas值越大模型越关注难以分类的样本，gamma=0时退化为交叉熵
    eps：防止分母为0的小常数
'''
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

'''
    ArcFaceLoss，其中：
    s：缩放因子，控制特征向量和分类权重向量之间的角度幅度，从而放大或者缩小特征向量的长度。s越大，生成的特征向量越分散
    m：用来增加类间边界的角度间隔，确保同一类的样本在特征空间中更为紧密，而不同类的样本之间有更大的边界。（越大越好？）
    crit：基损失函数，可选FocalLoss(focal)或交叉熵(bce)
    weight：用于不同类别的样本具有不同的重要性时，给每个类别分配不同的权重，从而在计算损失时对不同类别的样本进行加权。
    mean：损失缩减方法，平均(mean)或求和(sum)
    class_weights_norm：归一化方法，batch_norm或layer_norm(global？)
'''
class ArcFaceLoss(nn.modules.Module):
    def __init__(self, s=45.0, m=0.1, crit="bce", weight=None, reduction="mean", class_weights_norm='batch'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.class_weights_norm = class_weights_norm

        if crit == "focal":
            self.crit = FocalLoss(gamma=2)
        elif crit == "bce":
            self.crit = nn.CrossEntropyLoss(reduction="none")

        if s is None:
            self.s = torch.nn.Parameter(torch.tensor([45.], requires_grad=True, device='cuda'))
        else:
            self.s = s

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels):
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        labels2 = torch.zeros_like(cosine)
        labels2.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (labels2 * phi) + ((1.0 - labels2) * cosine)
        s = self.s
        output = output * s
        loss = self.crit(output, labels)

        if self.weight is not None:
            w = self.weight[labels].to(logits.device)
            loss = loss * w
            if self.class_weights_norm == "batch":
                loss = loss.sum() / w.sum()
            if self.class_weights_norm == "global":
                loss = loss.mean()
            else:
                loss = loss.mean()

            return loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
