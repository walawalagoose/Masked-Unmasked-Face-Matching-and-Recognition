from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import top_k_accuracy_score
from scipy.spatial.distance import cdist
import numpy as np

# from sklearn.metrics import roc_curve, auc, precision_recall_curve # 可随机抽几个类画图，但是没必要
# import matplotlib.pyplot as plt

# def embedding_matching(feature_vectors, labels, input_vector, top_k=20):
#     # 计算输入向量与所有特征向量的距离
#     distances = cdist([input_vector], feature_vectors, metric='euclidean')[0]
#     # 获取距离最小的前K个索引
#     top_k_indices = np.argsort(distances)[:top_k]
#     # 获取对应的标签和距离
#     top_k_labels = labels[top_k_indices]
#     top_k_distances = distances[top_k_indices]
#     return top_k_labels, top_k_distances
#
# # 评估函数
# def evaluate_accuracy(y_true, y_pred, distances):
#     # Accuracy
#     accuracy = accuracy_score(y_true, y_pred)
#     print(f"Accuracy: {accuracy}")
#
#     # Top-5, Top-10, Top-20 Accuracy
#     top_5_accuracy = top_k_accuracy_score(y_true, distances, k=5)
#     top_10_accuracy = top_k_accuracy_score(y_true, distances, k=10)
#     top_20_accuracy = top_k_accuracy_score(y_true, distances, k=20)
#
#     print(f"Top-5 Accuracy: {top_5_accuracy}")
#     print(f"Top-10 Accuracy: {top_10_accuracy}")
#     print(f"Top-20 Accuracy: {top_20_accuracy}")
#
#     return accuracy, top_5_accuracy, top_10_accuracy, top_20_accuracy
#
#
# def evaluate_others(y_true, y_pred):
#     # Precision, Recall, F1 Score
#     precision = precision_score(y_true, y_pred, average='weighted')
#     recall = recall_score(y_true, y_pred, average='weighted')
#     f1 = f1_score(y_true, y_pred, average='weighted')
#
#     print(f"Precision: {precision}")
#     print(f"Recall: {recall}")
#     print(f"F1 Score: {f1}")
#
#     return precision, recall, f1


def calculate_metrics_prototype(labels_gt, labels_pred, top_k=(1, 5, 10, 20)):
    metrics = {}
    y_true = labels_gt.flatten()
    y_pred = labels_pred[:, 0]  # top-1 predictions

    # Precision, Recall, F1 Score for top-1
    metrics['precision'] = precision_score(y_true, y_pred, average='macro')
    metrics['recall'] = recall_score(y_true, y_pred, average='macro')
    metrics['f1_score'] = f1_score(y_true, y_pred, average='macro')

    # Accuracy for top-1
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    # Top-k Accuracy
    for k in top_k:
        top_k_correct = 0
        for true_label, pred_labels in zip(y_true, labels_pred):
            if true_label in pred_labels[:k]:
                top_k_correct += 1
        metrics[f'top_{k}_accuracy'] = top_k_correct / len(y_true)

    return metrics

def calculate_metrics_knn(labels_gt, labels_pred, labels_pred_proba=None, top_k=(1, 5, 10, 20)):
    metrics = {}
    y_true = labels_gt.flatten()
    y_pred = labels_pred

    metrics['precision'] = precision_score(y_true, y_pred, average='macro')
    metrics['recall'] = recall_score(y_true, y_pred, average='macro')
    metrics['f1_score'] = f1_score(y_true, y_pred, average='macro')
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    if labels_pred_proba is not None:
        for k in top_k:
            top_k_predictions = np.argsort(labels_pred_proba, axis=1)[:, -k:]
            top_k_correct = 0
            for true_label, pred_labels in zip(y_true, top_k_predictions):
                if true_label in pred_labels:
                    top_k_correct += 1
            metrics[f'top_{k}_accuracy'] = top_k_correct / len(y_true)

    return metrics


def calculate_metrics_rf(labels_gt, labels_pred, labels_pred_proba=None, top_k=(5, 10, 20)):
    metrics = {}
    y_true = labels_gt.flatten()
    y_pred = labels_pred

    metrics['precision'] = precision_score(y_true, y_pred, average='macro')
    metrics['recall'] = recall_score(y_true, y_pred, average='macro')
    metrics['f1_score'] = f1_score(y_true, y_pred, average='macro')
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    if labels_pred_proba is not None:
        for k in top_k:
            top_k_predictions = np.argsort(labels_pred_proba, axis=1)[:, -k:]
            top_k_correct = 0
            for true_label, pred_labels in zip(y_true, top_k_predictions):
                if true_label in pred_labels:
                    top_k_correct += 1
            metrics[f'top_{k}_accuracy'] = top_k_correct / len(y_true)

    return metrics


def calculate_metrics_svm(labels_gt, labels_pred, labels_pred_proba=None, top_k=(5, 10, 20)):
    metrics = {}
    y_true = labels_gt.flatten()
    y_pred = labels_pred

    metrics['precision'] = precision_score(y_true, y_pred, average='macro')
    metrics['recall'] = recall_score(y_true, y_pred, average='macro')
    metrics['f1_score'] = f1_score(y_true, y_pred, average='macro')
    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    if labels_pred_proba is not None:
        for k in top_k:
            top_k_predictions = np.argsort(labels_pred_proba, axis=1)[:, -k:]
            top_k_correct = 0
            for true_label, pred_labels in zip(y_true, top_k_predictions):
                if true_label in pred_labels:
                    top_k_correct += 1
            metrics[f'top_{k}_accuracy'] = top_k_correct / len(y_true)

    return metrics