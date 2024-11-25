import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import cv2

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