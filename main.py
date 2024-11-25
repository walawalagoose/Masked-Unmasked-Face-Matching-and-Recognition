import torch
import torch.optim as optim
import numpy as np
from torchvision import transforms
# 各种分类器
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
# import face_recognition

import config
from dataset import load_celeba_data, load_celeba_data_masked
from models import FaceNet2
from losses import ArcFaceLoss
from train_vel_test import train_model, train_model_prototype
from prototype import calculate_prototypes, calculate_prototypes_hog, calculate_distances, get_sorted_labels
from evaluate import calculate_metrics_prototype, calculate_metrics_knn, calculate_metrics_rf, calculate_metrics_svm
from prototype import CombinedLoss, AssignmentLoss, ContrastiveLoss, OrthogonalLoss, get_hog_features
# import config

# 画图
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

unmasked_dataset_path = config.unmasked_dataset_path
masked_dataset_path = config.masked_dataset_path
preprocessed_dataset_path = config.preprocessed_dataset_path


# model_path配置
# model_path = config.model_path + '\\model_top100_pretrained_unblurred.pth'
# model_path = config.model_path + '\\model_top100_pretrained_ori.pth'
# model_path = config.model_path + '\\model_top100_unpretrained.pth'
# model_path = config.model_path + '\\model_top100_pretrained.pth'
model_path = config.model_path + '\\model_top100_pretrained_ori_prototype.pth'
# model_path = config.model_path + '\\model_top100_pretrained_unblurred.pth'
# model_path = config.model_path + '\\model_georgia_ori_prototype.pth'

# 钩子函数，用于捕获特定层的输出，用于画特征图
def hook_fn(module, input, output):
    global feature_maps
    feature_maps = output

# 注册钩子到特定层
def register_hooks(model):
    # 选择要可视化的卷积层
    model.model.net.conv2d_1a.register_forward_hook(hook_fn)
    model.model.net.conv2d_2a.register_forward_hook(hook_fn)
    model.model.net.conv2d_2b.register_forward_hook(hook_fn)


def visualize_feature_maps(model, data_loader, device):
    model.eval()
    images, labels = next(iter(data_loader))
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)

    # 可视化捕获的特征图
    for i in range(min(len(images), 5)):  # 这里只展示前5张图像的特征图
        plt.figure(figsize=(10, 10))
        plt.suptitle(f"Feature maps for image {i}")
        num_feature_maps = feature_maps.size(1)
        grid_size = int(num_feature_maps ** 0.5)
        for j in range(num_feature_maps):
            plt.subplot(grid_size, grid_size, j + 1)
            plt.imshow(feature_maps[i, j].cpu().numpy(), cmap='viridis')
            plt.axis('off')
        plt.show()


# 散点图
def plot_embeddings(embeddings, labels, title):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter)
    plt.title(title)
    plt.show()


def all_train_prototype():
    # 训练阶段
    # 损失函数
    base_loss = ArcFaceLoss(s=45.0, m=0.1, crit="focal").cuda() # ArcFace with Focal
    assignment_loss = AssignmentLoss().cuda()
    contrastive_loss = ContrastiveLoss().cuda()
    orthogonal_loss = OrthogonalLoss().cuda()

    criterion = CombinedLoss(base_loss, assignment_loss, contrastive_loss, orthogonal_loss, alpha=1, beta=1,
                             gamma=10).cuda()

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 训练模型
    best_model = train_model_prototype(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device=device, early_stopping_patience=5)

    # 保存模型
    torch.save(best_model, model_path)


def all_train():
    # 训练阶段
    # 损失函数和优化器
    criterion = ArcFaceLoss(s=45.0, m=0.1, crit="focal").cuda()  # ArcFace with Focal
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 训练模型
    best_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device=device, early_stopping_patience=5)

    # 保存模型
    torch.save(best_model, model_path)


def get_embeddings_perdataset_hog(model, data_loader, device, ifTest=True):
    if ifTest:
        # # 设置模型为评估模式
        # model.eval()

        embeddings_perdataset = []
        labels_gt_perdataset = []

        with torch.no_grad():
            for images, labels_batch in data_loader:
                images = images.to(device)
                hog_features = get_hog_features(images)
                embeddings_perdataset.append(hog_features)
                labels_gt_perdataset.append(labels_batch.cpu().numpy())

        embeddings_perdataset = np.concatenate(embeddings_perdataset, axis=0)
        labels_gt_perdataset = np.concatenate(labels_gt_perdataset, axis=0)

        return embeddings_perdataset, labels_gt_perdataset
    else:
        # # 训练集
        # model.eval()

        embeddings_perdataset = []
        labels_gt_perdataset = []

        with torch.no_grad():
            for images, labels_batch in data_loader:
                images = images.to(device)
                hog_features = get_hog_features(images)
                embeddings_perdataset.append(hog_features)
                labels_gt_perdataset.append(labels_batch.cpu().numpy())

        embeddings_perdataset = np.concatenate(embeddings_perdataset, axis=0)
        labels_gt_perdataset = np.concatenate(labels_gt_perdataset, axis=0)

        return embeddings_perdataset, labels_gt_perdataset

def get_embeddings_hog(ifTest=True):
    if ifTest:
        # 测试集1：未遮挡数据集
        embeddings_unmasked, labels_gt_unmasked = get_embeddings_perdataset_hog(model, test_loader_unmasked, device)

        # 测试集2：遮挡数据集
        embeddings_masked, labels_gt_masked = get_embeddings_perdataset_hog(model, test_loader_masked, device)

        print("Unmasked Feature vectors shape:", embeddings_unmasked.shape)
        print("Unmasked Labels shape:", labels_gt_unmasked.shape)
        print("Masked Feature vectors shape:", embeddings_masked.shape)
        print("Masked Labels shape:", labels_gt_masked.shape)

        embeddings = {'unmasked': embeddings_unmasked, 'masked': embeddings_masked}
        labels_gt = {'unmasked': labels_gt_unmasked, 'masked': labels_gt_masked}

        return embeddings, labels_gt
    else:
        # 训练集
        embeddings_train, labels_gt_train = get_embeddings_perdataset_hog(model, train_loader, device, ifTest=False)

        print("Train Feature vectors shape:", embeddings_train.shape)
        print("Train Labels shape:", labels_gt_train.shape)

        return embeddings_train, labels_gt_train

def get_embeddings_perdataset(model, data_loader, device, ifTest=True):
    if ifTest:
        # # 设置模型为评估模式
        # model.eval()

        embeddings_perdataset = []
        labels_gt_perdataset = []

        with torch.no_grad():
            for images, labels_batch in data_loader:
                images = images.to(device)
                outputs = model(images)
                embedding_perdataset = outputs['embeddings']
                embeddings_perdataset.append(embedding_perdataset.cpu().numpy())
                labels_gt_perdataset.append(labels_batch.cpu().numpy())

        embeddings_perdataset = np.concatenate(embeddings_perdataset, axis=0)
        labels_gt_perdataset = np.concatenate(labels_gt_perdataset, axis=0)

        return embeddings_perdataset, labels_gt_perdataset
    else:
        # # 训练集
        # model.eval()

        embeddings_perdataset = []
        labels_gt_perdataset = []

        with torch.no_grad():
            for images, labels_batch in data_loader:
                images = images.to(device)
                outputs = model(images)
                embedding_perdataset = outputs['embeddings']
                embeddings_perdataset.append(embedding_perdataset.cpu().numpy())
                labels_gt_perdataset.append(labels_batch.cpu().numpy())

        embeddings_perdataset = np.concatenate(embeddings_perdataset, axis=0)
        labels_gt_perdataset = np.concatenate(labels_gt_perdataset, axis=0)

        return embeddings_perdataset, labels_gt_perdataset

# def get_embeddings_face_recognition(model, data_loader, device, ifTest=True):
#     if ifTest:
#         # # 设置模型为评估模式
#         # model.eval()
#
#         embeddings_perdataset = []
#         labels_gt_perdataset = []
#
#         with torch.no_grad():
#             for images, labels_batch in data_loader:
#                 images = images.to(device)
#
#                 embedding_perdataset = face_recognition.face_encodings(image, face_recognition.face_locations(image, model='cnn',
#                                                                                        number_of_times_to_upsample=2),
#                                                 model='large')
#                 embeddings_perdataset.append(embedding_perdataset.cpu().numpy())
#                 labels_gt_perdataset.append(labels_batch.cpu().numpy())
#
#         embeddings_perdataset = np.concatenate(embeddings_perdataset, axis=0)
#         labels_gt_perdataset = np.concatenate(labels_gt_perdataset, axis=0)
#
#         return embeddings_perdataset, labels_gt_perdataset
#     else:
#         # # 训练集
#         # model.eval()
#
#         embeddings_perdataset = []
#         labels_gt_perdataset = []
#
#         with torch.no_grad():
#             for images, labels_batch in data_loader:
#                 images = images.to(device)
#                 outputs = model(images)
#                 embedding_perdataset = outputs['embeddings']
#                 embeddings_perdataset.append(embedding_perdataset.cpu().numpy())
#                 labels_gt_perdataset.append(labels_batch.cpu().numpy())
#
#         embeddings_perdataset = np.concatenate(embeddings_perdataset, axis=0)
#         labels_gt_perdataset = np.concatenate(labels_gt_perdataset, axis=0)
#
#         return embeddings_perdataset, labels_gt_perdataset


def get_embeddings(ifTest=True):
    if ifTest:
        # 测试集1：未遮挡数据集
        embeddings_unmasked, labels_gt_unmasked = get_embeddings_perdataset(model, test_loader_unmasked, device)

        # 测试集2：遮挡数据集
        embeddings_masked, labels_gt_masked = get_embeddings_perdataset(model, test_loader_masked, device)

        print("Unmasked Feature vectors shape:", embeddings_unmasked.shape)
        print("Unmasked Labels shape:", labels_gt_unmasked.shape)
        print("Masked Feature vectors shape:", embeddings_masked.shape)
        print("Masked Labels shape:", labels_gt_masked.shape)

        embeddings = {'unmasked': embeddings_unmasked, 'masked': embeddings_masked}
        labels_gt = {'unmasked': labels_gt_unmasked, 'masked': labels_gt_masked}

        return embeddings, labels_gt
    else:
        # 训练集
        embeddings_train, labels_gt_train = get_embeddings_perdataset(model, train_loader, device, ifTest=False)

        print("Train Feature vectors shape:", embeddings_train.shape)
        print("Train Labels shape:", labels_gt_train.shape)

        return embeddings_train, labels_gt_train


def all_test_prototype(type='None'):
    # # 计算原型
    # prototypes = calculate_prototypes(model, train_loader, device)

    # for test，输出原型结果
    print("Prototypes computed for each class (person):")
    for label, prototype in prototypes.items():
        print(f"Class {label}: Prototype shape: {prototype.shape}")

    # 计算分类结果
    test_labels_pred = {}
    for key in ['unmasked', 'masked']:
        # 计算每个嵌入向量与所有原型的距离
        distances = calculate_distances(test_embeddings[key], prototypes, type=type)

        # 获取排序后的标签列表
        sorted_labels = get_sorted_labels(distances, prototypes,type=type)

        print(f"Sorted labels for each {key} embedding based on distance to prototypes:")
        print(sorted_labels)

        test_labels_pred[key] = sorted_labels

    # 计算并输出评估指标
    for key in ['unmasked', 'masked']:
        metrics = calculate_metrics_prototype(test_labels_gt[key], test_labels_pred[key])
        print(f"Metrics for {key} test set:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

def all_test_knn():
    for n_neighbors in [1, 3, 5, 7, 9]:
        print(f"Testing n_neighbors={n_neighbors}")
        # 使用KNN算法进行分类
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean')
        knn.fit(train_embeddings, train_labels_gt.flatten())

        test_labels_pred = {}
        test_labels_pred_proba = {}

        for key in ['unmasked', 'masked']:
            test_labels_pred[key] = knn.predict(test_embeddings[key])
            test_labels_pred_proba[key] = knn.predict_proba(test_embeddings[key])

            print(f"Metrics for {key} test set:")
            metrics = calculate_metrics_knn(test_labels_gt[key], test_labels_pred[key],
                                            labels_pred_proba=test_labels_pred_proba[key])
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name}: {metric_value:.4f}")

def all_test_rf():
    for n_estimators in [100, 200, 300]:
        for max_depth in [None, 1, 2, 3]:
            print(f"Testing n_estimators={n_estimators}, max_depth={max_depth}")
            # 使用随机森林算法进行分类
            rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            rf.fit(train_embeddings, train_labels_gt.flatten())

            test_labels_pred = {}
            test_labels_pred_proba = {}

            for key in ['unmasked', 'masked']:
                test_labels_pred[key] = rf.predict(test_embeddings[key])
                test_labels_pred_proba[key] = rf.predict_proba(test_embeddings[key])

                print(f"Metrics for {key} test set:")
                metrics = calculate_metrics_rf(test_labels_gt[key], test_labels_pred[key],
                                               labels_pred_proba=test_labels_pred_proba[key])
                for metric_name, metric_value in metrics.items():
                    print(f"{metric_name}: {metric_value:.4f}")

def all_test_svm():
    # 使用SVM算法进行分类
    svm = OneVsRestClassifier(SVC(probability=True, kernel='linear', random_state=42))
    svm.fit(train_embeddings, train_labels_gt.flatten())

    test_labels_pred = {}
    test_labels_pred_proba = {}

    for key in ['unmasked', 'masked']:
        test_labels_pred[key] = svm.predict(test_embeddings[key])
        test_labels_pred_proba[key] = svm.predict_proba(test_embeddings[key])

        print(f"Metrics for {key} test set:")
        metrics = calculate_metrics_svm(test_labels_gt[key], test_labels_pred[key],
                                        labels_pred_proba=test_labels_pred_proba[key])
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

def all_test():
    return

if __name__ == '__main__':
    # 加载设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 数据转换，包括resize、转化为Tensor和normalization（这个一定要！）
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 数据加载
    # train_loader, val_loader, test_loader_unmasked, num_classes = load_celeba_data("D:\\NUS_proj\\Bonus\\datasets\\Georgia\\Unmasked_processed", batch_size=32, transform=transform, shuffle=True, train_ratio=0.8,test_ratio=0.1)
    # test_loader_masked = load_celeba_data_masked("D:\\NUS_proj\\Bonus\\datasets\\Georgia\\Masked", batch_size=32, transform=transform)  # 使用相同的transform
    train_loader, val_loader, test_loader_unmasked, num_classes = load_celeba_data(
        config.unmasked_dataset_path, batch_size=1, transform=transform, shuffle=True,
        train_ratio=0.8, test_ratio=0.1)
    test_loader_masked = load_celeba_data_masked(config.masked_dataset_path, batch_size=1,
                                                 transform=transform)  # 使用相同的transform

    # 模型定义
    # model = FaceNet2(num_classes=num_classes, dropout=0.3, device=device, pretrain=False).cuda()
    model = FaceNet2(num_classes=num_classes, dropout=0.3, device=device, pretrain=True).cuda()

    # all_train() # 训练
    # all_train_prototype() # 优化

    # # 加载模型
    # model.load_state_dict(torch.load(model_path))

    # 绘制feature map
    register_hooks(model) # 注册钩子函数
    visualize_feature_maps(model, test_loader_unmasked, device) # 可视化特征图

    # # 特征提取，得到特征向量
    train_embeddings, train_labels_gt = get_embeddings(ifTest=False)
    test_embeddings, test_labels_gt = get_embeddings(ifTest=True)

    # # hog版本
    # train_embeddings, train_labels_gt = get_embeddings_hog(ifTest=False)
    # test_embeddings, test_labels_gt = get_embeddings_hog(ifTest=True)

    # face_recognition版本
    # face_recognition.face_encodings(image,face_recognition.face_locations(image, model='cnn', number_of_times_to_upsample=2),
    #                                 model='large')

    # # 绘制散点图
    # plot_embeddings(test_embeddings['unmasked'], test_labels_gt['unmasked'], 'Unmasked Test Embeddings')
    # plot_embeddings(test_embeddings['masked'], test_labels_gt['masked'], 'Masked Test Embeddings')



    # 计算原型
    # prototypes = calculate_prototypes(model, train_loader, device)
    # prototypes = calculate_prototypes_hog(train_loader, device)

    # all_test_prototype()
    # all_test_knn()
    # all_test_rf()
    # all_test_svm()







