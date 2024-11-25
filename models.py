'''
    定义了多种深度学习模型架构，用于面部识别任务。
    主要包括InceptionResnet、EfficientNetEncoderHead、SEResNeXt101、GeM、FaceNet系列模型，
    以及ArcMarginProduct、FaceNet2等辅助类。
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from facenet_pytorch import InceptionResnetV1
from efficientnet_pytorch import EfficientNet
import timm


class InceptionResnet(nn.Module):
    def __init__(self, device, pool=None, dropout=0.3, pretrain=True):
        super(InceptionResnet, self).__init__()
        if pretrain:
            self.net = InceptionResnetV1(pretrained='vggface2', dropout_prob=dropout, device=device)
        else:
            self.net = InceptionResnetV1(dropout_prob=dropout, device=device)

        self.out_features = self.net.last_linear.in_features
        if pool == 'gem':
            self.net.avgpool_1a = GeM(p_trainable=True)

    def forward(self, x):
        x_out, x_feature = self.net(x)
        return x_out  # 这里作者原来写错了！！！！
        # return x_feature # 这里作者原来写错了！！！！



class EfficientNetEncoderHead(nn.Module):
    def __init__(self, depth, pretrain=True):
        super(EfficientNetEncoderHead, self).__init__()
        self.depth = depth
        if pretrain:
            self.net = EfficientNet.from_pretrained(f'efficientnet-b{self.depth}')
        else:
            self.net = EfficientNet.from_name(f'efficientnet-b{self.depth}')
        self.out_features = self.net._fc.in_features

    def forward(self, x):
        return self.net.extract_features(x)


class SEResNeXt101(nn.Module):
    def __init__(self, pretrain=True):
        super(SEResNeXt101, self).__init__()
        self.net = timm.create_model('gluon_seresnext101_32x4d', pretrained=pretrain)
        self.out_features = self.net.fc.in_features

    def forward(self, x):
        return self.net.forward_features(x)


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=True):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)


class FaceNet(nn.Module):
    def __init__(self, model_name=None, pool=None, dropout=0.0, pretrain=True, embedding_size=512, device=None):
        super(FaceNet, self).__init__()
        self.model_name = model_name
        if model_name == 'resnet':
            self.model = SEResNeXt101(pretrain)
        elif model_name == 'effnet':
            self.model = EfficientNetEncoderHead(depth=5, pretrain=pretrain)
        else:
            self.model = InceptionResnet(device, pool=pool, dropout=dropout, pretrain=pretrain)
        if pool == "gem":
            self.global_pool = GeM(p_trainable=True)
        else:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.neck = nn.Sequential(
            nn.Linear(self.model.out_features, embedding_size, bias=True),
            nn.BatchNorm1d(embedding_size, eps=0.001),
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.model(x)
        x = self.global_pool(x)
        x = self.dropout(x)
        x = x[:, :, 0, 0]
        embeddings = self.neck(x)
        return embeddings


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine


class FaceNet2(nn.Module):
    def __init__(self, num_classes, model_name=None, pool=None, dropout=0.0, embedding_size=512, device='cuda',
                 pretrain=True):
        super(FaceNet2, self).__init__()
        self.model_name = model_name

        # model
        if model_name == 'resnet':
            self.model = SEResNeXt101(pretrain)
        elif model_name == 'effnet':
            self.model = EfficientNetEncoderHead(depth=3, pretrain=pretrain)
        else:
            self.model = InceptionResnet(device, pool=pool, dropout=dropout, pretrain=pretrain)

        # global pooling
        if (pool == "gem"):
            # Generalizing Pooling
            self.global_pool = GeM(p_trainable=True)
        else:
            # global average pooling
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        #neck
        self.neck = nn.Sequential(
            nn.Linear(self.model.out_features, embedding_size, bias=True),
            nn.BatchNorm1d(embedding_size, eps=0.001),
        )
        self.dropout = nn.Dropout(p=dropout)
        self.head = ArcMarginProduct(embedding_size, num_classes)

    def forward(self, x):
        # backbone
        if self.model_name == None:
            embeddings = self.model(x)

            logits = self.head(embeddings)
            return {'logits': logits, 'embeddings': embeddings}

        else:
            x = self.model(x)
            x = self.global_pool(x) # global pool
            x = self.dropout(x)
            x = x[:,:,0,0] # change the output from cnn to a vector first
            embeddings = self.neck(x) # vector with num_classes
            logits = self.head(embeddings)
            return {'logits': logits, 'embeddings': embeddings}


# class FaceNet2_embeddings(nn.Module):
#     def __init__(self, model_name=None, pool=None, dropout=0.0, device='cuda', pretrain=True):
#         super(FaceNet2_embeddings, self).__init__()
#         self.model_name = model_name
#         if model_name == 'resnet':
#             self.model = SEResNeXt101(pretrain)
#         elif model_name == 'effnet':
#             self.model = EfficientNetEncoderHead(depth=5, pretrain=pretrain)
#         else:
#             self.model = InceptionResnet(device, pool=pool, dropout=dropout, pretrain=pretrain)
#         self.global_pool = None
#         self.neck = None
#         self.dropout = None
#
#     def forward(self, x):
#         x = self.model(x)
#         return x
