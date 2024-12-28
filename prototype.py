"""
    待定
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from skimage.feature import hog
from skimage import color


def get_hog_features(images):
    hog_features = []
    for img in images:
        img_gray = color.rgb2gray(img.permute(1, 2, 0).cpu().numpy())  # 转换为灰度图
        features = hog(img_gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        hog_features.append(features)
    return np.array(hog_features)

def calculate_prototypes(model, data_loader, device):
    """
    计算每个类别（每个人）的原型向量

    参数:
    model (torch.nn.Module): 已训练的模型
    data_loader (torch.utils.data.DataLoader): 提供数据的数据加载器
    device (str): 计算设备

    返回:
    dict: 每个类别的原型向量
    """
    model.eval()  # 设置模型为评估模式，以确保参数不更新，且禁用dropout等
    embeddings_dict = {}  # 用于存储每个类别的特征嵌入
    labels_dict = {}  # 用于存储每个类别的标签

    with torch.no_grad():  # 禁用梯度计算，节省内存和计算资源
        for inputs, labels in data_loader:  # 遍历数据加载器中的所有数据
            inputs = inputs.to(device)  # 将输入数据移动到指定的设备（例如GPU）
            labels = labels.to(device)  # 将标签移动到指定的设备

            outputs = model(inputs)  # 前向传播，获取模型输出
            embeddings = outputs['embeddings'].cpu().numpy()  # 获取特征嵌入并转为numpy数组
            labels = labels.cpu().numpy()  # 将标签转为numpy数组

            for embedding, label in zip(embeddings, labels):  # 遍历每个特征嵌入和对应的标签
                if label not in embeddings_dict:
                    embeddings_dict[label] = []  # 如果该标签不在字典中，初始化一个空列表
                embeddings_dict[label].append(embedding)  # 将特征嵌入添加到对应标签的列表中

    prototypes = {}  # 用于存储每个类别的原型向量
    for label, embeddings in embeddings_dict.items():
        prototypes[label] = np.mean(embeddings, axis=0)  # 计算每个类别的特征嵌入的均值，作为原型向量

    return prototypes  # 返回包含每个类别原型向量的字典


def calculate_prototypes_hog(data_loader, device):
    """
    计算每个类别（每个人）的HOG特征原型向量

    参数:
    data_loader (torch.utils.data.DataLoader): 提供数据的数据加载器
    device (str): 计算设备

    返回:
    dict: 每个类别的原型向量
    """
    embeddings_dict = {}  # 用于存储每个类别的特征嵌入
    labels_dict = {}  # 用于存储每个类别的标签

    with torch.no_grad():  # 禁用梯度计算，节省内存和计算资源
        for inputs, labels in data_loader:  # 遍历数据加载器中的所有数据
            inputs = inputs.to(device)  # 将输入数据移动到指定的设备（例如GPU）
            labels = labels.cpu().numpy()  # 将标签转为numpy数组

            hog_features = get_hog_features(inputs)  # 提取HOG特征

            for embedding, label in zip(hog_features, labels):  # 遍历每个特征嵌入和对应的标签
                if label not in embeddings_dict:
                    embeddings_dict[label] = []  # 如果该标签不在字典中，初始化一个空列表
                embeddings_dict[label].append(embedding)  # 将特征嵌入添加到对应标签的列表中

    prototypes = {}  # 用于存储每个类别的原型向量
    for label, embeddings in embeddings_dict.items():
        prototypes[label] = np.mean(embeddings, axis=0)  # 计算每个类别的特征嵌入的均值，作为原型向量

    return prototypes  # 返回包含每个类别原型向量的字典


def calculate_distances(embeddings, prototypes, type='None'):
    """
    计算每个嵌入向量与所有原型之间的距离
    """
    if type == 'cos':
        # 将原型向量转化为矩阵
        prototype_matrix = np.stack(list(prototypes.values()))

        # 计算每个嵌入向量与原型向量之间的余弦相似度
        dot_product = np.dot(embeddings, prototype_matrix.T)
        norm_embeddings = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm_prototypes = np.linalg.norm(prototype_matrix, axis=1, keepdims=True)

        cosine_similarities = dot_product / (norm_embeddings * norm_prototypes.T)

        return cosine_similarities
    else: # 欧氏距离
        distances = np.zeros((embeddings.shape[0], len(prototypes)))

        for i, embedding in enumerate(embeddings):
            for j, (label, prototype) in enumerate(prototypes.items()):
                distances[i, j] = np.linalg.norm(embedding - prototype)

        return distances


def calculate_distances_hog(embeddings, prototypes, type='None'):
    """
    计算每个嵌入向量与所有原型之间的距离
    """
    distances = np.zeros((embeddings.shape[0], len(prototypes)))

    for i, embedding in enumerate(embeddings):
        for j, (label, prototype) in enumerate(prototypes.items()):
            if type == 'cos':
                distances[i, j] = 1 - np.dot(embedding, prototype) / (
                            np.linalg.norm(embedding) * np.linalg.norm(prototype))
            else:
                distances[i, j] = np.linalg.norm(embedding - prototype)

    return distances


def get_sorted_labels(distances, prototypes, type='None'):
    """
    对每个嵌入向量与所有原型的距离排序，并返回对应的标签列表
    """
    if type == 'cos':
        sorted_labels = np.argsort(-distances, axis=1) # 余弦相似度，反向排序！
    else:
        sorted_labels = np.argsort(distances, axis=1)
    prototype_labels = list(prototypes.keys())

    sorted_labels = np.vectorize(lambda x: prototype_labels[x])(sorted_labels)

    return sorted_labels


# 定义新的联合损失类
class CombinedLoss(nn.Module):
    def __init__(self, base_loss, assignment_loss, contrastive_loss, orthogonal_loss, alpha=0.5, beta=0.5, gamma=0.5):
        super(CombinedLoss, self).__init__()
        self.base_loss = base_loss
        self.assignment_loss = assignment_loss
        self.contrastive_loss = contrastive_loss
        self.orthogonal_loss = orthogonal_loss
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, logits, embeddings, labels):
        prototypes = self.calculate_prototypes_training(embeddings, labels)
        self.assignment_loss.prototypes = prototypes
        self.orthogonal_loss.prototypes = prototypes

        loss1 = self.base_loss(logits, labels)
        loss2 = self.assignment_loss(embeddings, labels)
        loss3 = self.contrastive_loss(embeddings, labels)
        loss4 = self.orthogonal_loss()
        total_loss = loss1 + self.alpha * loss2 + self.beta * loss3 + self.gamma * loss4
        return total_loss, loss1, loss2, loss3, loss4

    def calculate_prototypes_training(self, embeddings, labels):
        embeddings_dict = {}
        for embedding, label in zip(embeddings, labels):
            label = label.item()
            if label not in embeddings_dict:
                embeddings_dict[label] = []
            embeddings_dict[label].append(embedding.detach().cpu().numpy())

        prototypes = {label: np.mean(embeddings, axis=0) for label, embeddings in embeddings_dict.items()}
        return prototypes


# 定义分配损失类
class AssignmentLoss(nn.Module):
    def __init__(self, prototypes=None):
        super(AssignmentLoss, self).__init__()
        self.prototypes = prototypes

    def forward(self, embeddings, labels):
        cosine_similarities = []
        for embedding, label in zip(embeddings, labels):
            prototype = self.prototypes[label.item()]
            prototype = torch.tensor(prototype).to(embedding.device)
            cosine_similarity = F.cosine_similarity(embedding.unsqueeze(0), prototype.unsqueeze(0))
            cosine_similarities.append(cosine_similarity)
        cosine_similarities = torch.cat(cosine_similarities)
        avg_cosine_similarity = torch.mean(cosine_similarities)
        loss = 1 - avg_cosine_similarity
        return loss


# 定义对比损失类
class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()

    def forward(self, embeddings, labels):
        batch_size = embeddings.size(0)
        # 计算所有向量之间的余弦相似度
        cosine_similarities = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
        total_loss = 0.0
        total_pairs = 0  # 用于统计总对数量

        for label in labels.unique():
            # 创建掩码，只保留当前标签类别
            same_label_mask = labels.unsqueeze(0) == label
            same_label_mask = same_label_mask & same_label_mask.T  # 广播掩码使其形状匹配
            # 提取当前标签类别的相似度
            s_ij = cosine_similarities[same_label_mask].view(-1)
            if len(s_ij) == 0:
                continue  # 跳过没有相同标签的情况

            # 计算指数
            exp_s_ij = torch.exp(s_ij)
            # 计算softmax
            sum_exp_s_ij = exp_s_ij.sum()
            softmax_s_ij = exp_s_ij / sum_exp_s_ij
            # 取对数
            log_softmax_s_ij = torch.log(softmax_s_ij)
            # 取负数并求和
            loss = -log_softmax_s_ij.sum()
            total_loss += loss

            # 用对数量的平方和进行归一化
            total_pairs = sum([(labels == label).sum().item() ** 2 for label in labels.unique()])

        if total_pairs == 0:
            return torch.tensor(0.0, device=embeddings.device)  # 避免除以零
        return total_loss / total_pairs


# 定义正交损失类
class OrthogonalLoss(nn.Module):
    def __init__(self, prototypes=None):
        super(OrthogonalLoss, self).__init__()
        self.prototypes = prototypes

    def forward(self):
        prototype_list = list(self.prototypes.values())
        num_prototypes = len(prototype_list)

        cosine_similarities = []
        for i in range(num_prototypes):
            for j in range(i + 1, num_prototypes):
                cosine_similarity = F.cosine_similarity(
                    torch.tensor(prototype_list[i]).unsqueeze(0),
                    torch.tensor(prototype_list[j]).unsqueeze(0)
                )
                cosine_similarities.append(cosine_similarity ** 2)

        orthogonal_loss = torch.stack(cosine_similarities).sum()
        # 归一化 上三角矩阵
        return orthogonal_loss / ((num_prototypes**2 + num_prototypes)/2)

# class LossFunctions:
#
#     def orthogonality_loss(self, prototypes, k):
#         """
#         prototypes (n_b,64)
#         """
#         # 确保传入的 k 是偶数
#         assert k % 2 == 0, "k must be even."
#         D = k // 2
#
#         # 重塑原型向量并标准化
#         prototypes = prototypes.view(k, 64)
#         prototypes = F.normalize(prototypes, p=2, dim=1)
#
#         # 计算余弦相似度矩阵
#         # (n_b,n_b)
#         cosine_similarity_matrix = torch.mm(prototypes, prototypes.t())
#
#         # 获取上三角部分的元素（不包括对角线）
#         upper_triangular_part = torch.triu(cosine_similarity_matrix, diagonal=1)
#
#         # 计算损失：平方和除以标准化系数
#         loss = torch.sum(upper_triangular_part ** 2)
#         normalization_factor = D * (2 * D - 1)
#         loss /= normalization_factor
#
#         return loss
#
#     def contrastive_loss(self, vec, labels, duration):
#         """
#         vec(instance)  (n_b,64)
#         labels (n_b,1)
#         duration (n_b,1)
#         """
#         # Normalize the vectors
#         vec_norm = F.normalize(vec, p=2, dim=1)
#         # (n_b,n_b)
#         sim_matrix = torch.mm(vec_norm, vec_norm.t())
#
#         # Create masks
#         # 1(ci=cj)
#         label_eq = labels == labels.t()
#         # 1(di=dj)
#         duration_eq = duration == duration.t()
#         # and
#         mask = label_eq & duration_eq
#         mask = mask.float()
#
#         # Compute the softmax denominator
#         exp_sim = torch.exp(sim_matrix)
#         total_sum = torch.sum(exp_sim, dim=1, keepdim=True)
#
#         normal_sim = exp_sim / total_sum
#         log_normal_sim = torch.log(normal_sim + 1e-10)
#
#         weighted_log_prob = mask * log_normal_sim
#         loss = -torch.sum(weighted_log_prob)
#
#         return loss
#
#     def assign_loss(self, output, vec):
#         # 标准化向量
#         output_norm = F.normalize(output, p=2, dim=1)
#         vec_norm = F.normalize(vec, p=2, dim=1)
#
#         # 计算余弦相似度
#         cosine_similarity = torch.sum(output_norm * vec_norm, dim=1)
#
#         # 计算损失
#         loss = 1 - torch.mean(cosine_similarity)
#
#         return loss
