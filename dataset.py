import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import cv2
from preprocess import preprocess_and_save_keypoints
import pickle

class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 遍历根目录下的每个子文件夹
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for file in os.listdir(label_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(label_dir, file))
                        self.labels.append(int(label))  # 标签是子文件夹的名字，转换为整数

        # 确定num_classes
        self.num_classes = max(self.labels) + 1  # 假设标签从0开始

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_celeba_data(root_dir, batch_size=32, transform=None, shuffle=True, train_ratio=0.8, test_ratio=0.1):
    dataset = CelebADataset(root_dir, transform=transform)

    # 计算训练集、验证集和测试集的大小
    train_size = int(train_ratio * len(dataset))
    test_size = int(test_ratio * len(dataset))
    val_size = len(dataset) - train_size - test_size

    # 随机划分数据集
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader, test_loader, dataset.num_classes

def load_celeba_data_masked(root_dir, batch_size=32, transform=None, shuffle=True):
    dataset = CelebADataset(root_dir, transform=transform)

    # 创建数据加载器
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return test_loader

# 带关键点信息的数据集类
class FaceDatasetWithKeypoints(Dataset):
    def __init__(self, root_dir, transform=None, keypoints_path=None):
        self.root_dir = root_dir
        self.transform = transform
        self.keypoints_path = keypoints_path

        # 加载图像路径和标签
        self.image_paths = []
        self.labels = []

        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for file in os.listdir(label_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(label_dir, file))
                        self.labels.append(int(label))  # 假设标签是子文件夹名称，转换为整数

        # 确定 num_classes
        self.num_classes = max(self.labels) + 1  # 假设标签从 0 开始

        # 加载关键点信息
        if keypoints_path is not None:
            with open(keypoints_path, 'rb') as f:
                self.keypoints_dict = pickle.load(f)
        else:
            self.keypoints_dict = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        if self.keypoints_dict is not None:
            keypoints = self.keypoints_dict.get(idx, None)  # 获取关键点或 None
        else:
            keypoints = None

        return image, label, keypoints

# 带关键点的loader
def load_celeba_data_with_keypoints(root_dir, batch_size=32, transform=None, shuffle=True, train_ratio=0.8, val_ratio=0.1, predictor_path=None):
    dataset = FaceDatasetWithKeypoints(root_dir, transform=transform, keypoints_path=None)

    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_subset, val_subset, test_subset = random_split(dataset, [train_size, val_size, test_size])

    # 预处理关键点
    if predictor_path is not None:
        print("开始预处理训练集关键点...")
        preprocess_and_save_keypoints(train_subset, predictor_path, 'train_keypoints.pkl', image_size=(224, 224))
        print("开始预处理验证集关键点...")
        preprocess_and_save_keypoints(val_subset, predictor_path, 'val_keypoints.pkl', image_size=(224, 224))
    else:
        print("未提供预测器路径，跳过关键点预处理。")

    # 创建新的数据集对象，加载关键点信息
    train_dataset = FaceDatasetWithKeypoints(root_dir, transform=transform, keypoints_path='train_keypoints.pkl')
    val_dataset = FaceDatasetWithKeypoints(root_dir, transform=transform, keypoints_path='val_keypoints.pkl')
    test_dataset = FaceDatasetWithKeypoints(root_dir, transform=transform, keypoints_path=None)  # 测试集未预处理关键点

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    num_classes = dataset.num_classes

    return train_loader, val_loader, test_loader, num_classes

# 示例：如何使用修改后的数据集和数据加载器
if __name__ == "__main__":
    # transform = transforms.Compose([
    #     transforms.Resize((128, 128)),
    #     transforms.ToTensor()
    # ])
    #
    # root_dir = '/path/to/your/dataset'
    # dataloader = load_celeba_data(root_dir, batch_size=32, transform=transform)
    #
    # for images, labels in dataloader:
    #     print(images.shape, labels.shape)
    #     # 在此处可以添加进一步的处理或训练代码
    print()