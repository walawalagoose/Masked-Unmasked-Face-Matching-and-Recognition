'''
    定义了多种深度学习模型架构，用于面部识别任务。
    主要包括InceptionResnet、EfficientNetEncoderHead、SEResNeXt101、GeM、FaceNet系列模型，
    以及ArcMarginProduct、FaceNet2等辅助类。
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
# from facenet_pytorch import InceptionResnetV1
from facenet_pytorch_local import InceptionResnetV1, CBAM
# from facenet_pytorch_local import InceptionResnetV1_CBAM
# from efficientnet_pytorch import EfficientNet
import timm
# from torchvision.models.resnet import BasicBlock
import dlib


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


# class EfficientNetEncoderHead(nn.Module):
#     def __init__(self, depth, pretrain=True):
#         super(EfficientNetEncoderHead, self).__init__()
#         self.depth = depth
#         if pretrain:
#             self.net = EfficientNet.from_pretrained(f'efficientnet-b{self.depth}')
#         else:
#             self.net = EfficientNet.from_name(f'efficientnet-b{self.depth}')
#         self.out_features = self.net._fc.in_features
#
#     def forward(self, x):
#         return self.net.extract_features(x)


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


# class FaceNet(nn.Module):
#     def __init__(self, model_name=None, pool=None, dropout=0.0, pretrain=True, embedding_size=512, device=None):
#         super(FaceNet, self).__init__()
#         self.model_name = model_name
#         if model_name == 'resnet':
#             self.model = SEResNeXt101(pretrain)
#         elif model_name == 'effnet':
#             self.model = EfficientNetEncoderHead(depth=5, pretrain=pretrain)
#         else:
#             self.model = InceptionResnet(device, pool=pool, dropout=dropout, pretrain=pretrain)
#         if pool == "gem":
#             self.global_pool = GeM(p_trainable=True)
#         else:
#             self.global_pool = nn.AdaptiveAvgPool2d(1)
#         self.neck = nn.Sequential(
#             nn.Linear(self.model.out_features, embedding_size, bias=True),
#             nn.BatchNorm1d(embedding_size, eps=0.001),
#         )
#         self.dropout = nn.Dropout(p=dropout)
#
#     def forward(self, x):
#         x = self.model(x)
#         x = self.global_pool(x)
#         x = self.dropout(x)
#         x = x[:, :, 0, 0]
#         embeddings = self.neck(x)
#         return embeddings


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
        # elif model_name == 'effnet':
        #     self.model = EfficientNetEncoderHead(depth=3, pretrain=pretrain)
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


class FaceNet2_CBAM(nn.Module):
    def __init__(self, num_classes, pretrained_backbone=True, dropout=0.3, embedding_size=512, device='cuda'):
        super(FaceNet2, self).__init__()
        self.device = device

        self.backbone = InceptionResnetV1(pretrained=pretrained_backbone)
        self.backbone.to(device)

        self.cbam = CBAM(channels=512)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.neck = nn.Sequential(
            nn.Linear(512, embedding_size, bias=True),
            nn.BatchNorm1d(embedding_size, eps=0.001),
        )

        self.dropout = nn.Dropout(p=dropout)
        self.head = ArcMarginProduct(embedding_size, num_classes)

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('D:\\Coding_projects\\python_pj\\Masked-Unmasked-Face-Matching-and-Recognition\\1\\shape_predictor_68_face_landmarks.dat')

    def mask_mouth_region(self, feature_map, shape, image_size, margin=10):
        B, C, H, W = feature_map.size()
        device = feature_map.device
        masked_feature = feature_map.clone()

        for i in range(B):
            # 获取嘴巴区域的关键点（假设使用68个点中的48-67为嘴巴）
            mouth_points = np.array([(shape[j][0], shape[j][1]) for j in range(48, 68)])
            x_min = np.min(mouth_points[:, 0]) - margin
            x_max = np.max(mouth_points[:, 0]) + margin
            y_min = np.min(mouth_points[:, 1]) - margin
            y_max = np.max(mouth_points[:, 1]) + margin

            scale_x = W / image_size[1]
            scale_y = H / image_size[0]
            x_min = int(max(0, x_min * scale_x))
            x_max = int(min(W, x_max * scale_x))
            y_min = int(max(0, y_min * scale_y))
            y_max = int(min(H, y_max * scale_y))

            masked_feature[i, :, y_min:y_max, x_min:x_max] = 0  # 或其他抑制方法

        return masked_feature

    def forward(self, x, keypoints=None, image_size=(224, 224)):
        """
        x: 输入图像 [B, C, H, W]
        keypoints: 预处理的嘴巴关键点列表，每个元素为 [(x1, y1), (x2, y2), ..., (xn, yn)] 或 None
        image_size: 输入图像的尺寸
        """
        B, C, H, W = x.size()
        device = x.device

        # 如果 keypoints 没有预处理，进行关键点检测
        if keypoints is None:
            # 获取CPU上的图像用于dlib处理
            x_cpu = x.cpu().detach().numpy()
            gray = np.mean(x_cpu, axis=1).astype(np.uint8)  # 转为灰度图

            # 面部检测和关键点检测
            keypoints = []
            for i in range(B):
                dets = self.detector(gray[i], 1)
                if len(dets) > 0:
                    shape = self.predictor(gray[i], dets[0])
                    mouth_points = [(shape.part(j).x, shape.part(j).y) for j in range(48, 68)]
                    keypoints.append(mouth_points)
                else:
                    keypoints.append(None)

        # 进行特征提取
        features = self.backbone(x).unsqueeze(-1).unsqueeze(-1)  # 假设 backbone 输出 [B, C]
        features = self.global_pool(features)  # [B, C, 1, 1]

        # 扩展到 [B, C, H, W]，这里假设 H=W=1
        # 如果 backbone 的输出是 [B, C, H, W]，则无需调整
        # 在此示例中，我们假设需要扩展为 [B, C, H, W]
        # 实际情况请根据 backbone 的输出调整
        # features = features.expand(-1, -1, H, W)  # 示例

        # 创建掩码
        masked_features = self.mask_mouth_region(features, keypoints, image_size)

        # 应用 CBAM
        masked_features = self.cbam(masked_features)

        # 全局池化
        x = self.global_pool(masked_features)  # [B, C, 1, 1]
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # [B, C]
        embeddings = self.neck(x)  # [B, embedding_size]
        logits = self.head(embeddings)  # [B, num_classes]
        return {'logits': logits, 'embeddings': embeddings}