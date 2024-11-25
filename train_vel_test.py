import torch
from tqdm import tqdm
from prototype import calculate_prototypes, calculate_prototypes_hog

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device='cuda', early_stopping_patience=5):
    """
        训练模型函数，添加early stopping机制

        参数:
        model (torch.nn.Module): 要训练的模型
        train_loader (torch.utils.data.DataLoader): 提供训练数据的数据加载器
        val_loader (torch.utils.data.DataLoader): 提供验证数据的数据加载器
        criterion (torch.nn.Module): 损失函数
        optimizer (torch.optim.Optimizer): 优化器
        num_epochs (int): 训练的周期数，默认为 25
        device (str): 计算设备，默认为 'cuda'
        early_stopping_patience (int): 如果验证集损失在该次数的epoch中没有改善，则停止训练

        功能:
        迭代数据加载器中的数据，进行前向传播、损失计算、反向传播和参数更新。
        打印每个周期的平均损失，并在验证集损失没有改善时停止训练。
        返回训练过程中最优的模型参数。
        """
    model.to(device)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_params = model.state_dict()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # 使用tqdm显示进度条
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            logits = outputs['logits']
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # 更新进度条描述信息
            progress_bar.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        tqdm.write(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}')

        # 验证模型
        val_loss = val_model(model, val_loader, criterion, device)
        tqdm.write(f'Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}')

        # 检查early stopping条件
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_params = model.state_dict() # 保存最优模型参数
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f'Early stopping triggered after {epoch + 1} epochs.')
            break

    # # 在训练集里放原型
    # best_model = model.load_state_dict(best_model_params)
    #
    # # 训练完成后计算原型
    # prototypes = calculate_prototypes(best_model, train_loader, device)
    #
    # return best_model_params, prototypes

    # 在训练完成后放原型
    return best_model_params


def train_model_prototype(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device='cuda', early_stopping_patience=5):
    """
        训练模型函数，添加early stopping机制

        参数:
        model (torch.nn.Module): 要训练的模型
        train_loader (torch.utils.data.DataLoader): 提供训练数据的数据加载器
        val_loader (torch.utils.data.DataLoader): 提供验证数据的数据加载器
        criterion (torch.nn.Module): 损失函数
        optimizer (torch.optim.Optimizer): 优化器
        num_epochs (int): 训练的周期数，默认为 25
        device (str): 计算设备，默认为 'cuda'
        early_stopping_patience (int): 如果验证集损失在该次数的epoch中没有改善，则停止训练

        功能:
        迭代数据加载器中的数据，进行前向传播、损失计算、反向传播和参数更新。
        打印每个周期的平均损失，并在验证集损失没有改善时停止训练。
        返回训练过程中最优的模型参数。
        """
    model.to(device)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_params = model.state_dict()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # 加的
        running_base_loss = 0.0
        running_assignment_loss = 0.0
        running_contrastive_loss = 0.0
        running_orthogonal_loss = 0.0

        # 使用tqdm显示进度条
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            logits = outputs['logits']
            embeddings = outputs['embeddings']

            total_loss, base_loss, assignment_loss, contrastive_loss, orthogonal_loss = criterion(logits, embeddings,
                                                                                                  labels)
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * inputs.size(0)
            running_base_loss += base_loss.item() * inputs.size(0)
            running_assignment_loss += assignment_loss.item() * inputs.size(0)
            running_contrastive_loss += contrastive_loss.item() * inputs.size(0)
            running_orthogonal_loss += orthogonal_loss.item() * inputs.size(0)

            # 更新进度条描述信息
            progress_bar.set_postfix(loss=total_loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_base_loss = running_base_loss / len(train_loader.dataset)
        epoch_assignment_loss = running_assignment_loss / len(train_loader.dataset)
        epoch_contrastive_loss = running_contrastive_loss / len(train_loader.dataset)
        epoch_orthogonal_loss = running_orthogonal_loss / len(train_loader.dataset)

        tqdm.write(f'Training - Total Loss: {epoch_loss:.4f}, Base Loss: {epoch_base_loss:.4f}, '
          f'Assignment Loss: {epoch_assignment_loss:.4f}, Contrastive Loss: {epoch_contrastive_loss:.4f}, '
          f'Orthogonal Loss: {epoch_orthogonal_loss:.4f}')
        tqdm.write(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}')

        # 验证模型
        val_loss = val_model_prototype(model, val_loader, criterion, device)
        tqdm.write(f'Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}')

        # 检查early stopping条件
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_params = model.state_dict() # 保存最优模型参数
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f'Early stopping triggered after {epoch + 1} epochs.')
            break

    # # 在训练集里放原型
    # best_model = model.load_state_dict(best_model_params)
    #
    # # 训练完成后计算原型
    # prototypes = calculate_prototypes(best_model, train_loader, device)
    #
    # return best_model_params, prototypes

    # 在训练完成后放原型
    return best_model_params


def val_model(model, dataloader, criterion, device='cuda'):
    """
    验证模型函数

    参数:
    model (torch.nn.Module): 要测试的模型
    dataloader (torch.utils.data.DataLoader): 提供测试数据的数据加载器
    criterion (torch.nn.Module): 损失函数
    device (str): 计算设备，默认为 'cuda'

    功能:
    在没有梯度计算的情况下，迭代数据加载器中的数据，进行前向传播和损失计算。
    返回验证集的平均损失。
    """
    model.eval()
    running_loss = 0.0
    # 使用tqdm显示进度条
    progress_bar = tqdm(dataloader, desc='Validating')
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            logits = outputs['logits']
            loss = criterion(logits, labels)

            running_loss += loss.item() * inputs.size(0)

            # 更新进度条描述信息
            progress_bar.set_postfix(loss=loss.item())

    val_loss = running_loss / len(dataloader.dataset)
    return val_loss


def val_model_prototype(model, dataloader, criterion, device='cuda'):
    """
    验证模型函数

    参数:
    model (torch.nn.Module): 要测试的模型
    dataloader (torch.utils.data.DataLoader): 提供测试数据的数据加载器
    criterion (torch.nn.Module): 损失函数
    device (str): 计算设备，默认为 'cuda'

    功能:
    在没有梯度计算的情况下，迭代数据加载器中的数据，进行前向传播和损失计算。
    返回验证集的平均损失。
    """
    model.eval()
    running_loss = 0.0
    # 加的
    running_base_loss = 0.0
    running_assignment_loss = 0.0
    running_contrastive_loss = 0.0
    running_orthogonal_loss = 0.0

    # 使用tqdm显示进度条
    progress_bar = tqdm(dataloader, desc='Validating')
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            logits = outputs['logits']
            embeddings = outputs['embeddings']

            total_loss, base_loss, assignment_loss, contrastive_loss, orthogonal_loss = criterion(logits, embeddings,
                                                                                                  labels)

            running_loss += total_loss.item() * inputs.size(0)
            running_base_loss += base_loss.item() * inputs.size(0)
            running_assignment_loss += assignment_loss.item() * inputs.size(0)
            running_contrastive_loss += contrastive_loss.item() * inputs.size(0)
            running_orthogonal_loss += orthogonal_loss.item() * inputs.size(0)

            # 更新进度条描述信息
            progress_bar.set_postfix(loss=total_loss.item())

        val_loss = running_loss / len(dataloader.dataset)
        val_base_loss = running_base_loss / len(dataloader.dataset)
        val_assignment_loss = running_assignment_loss / len(dataloader.dataset)
        val_contrastive_loss = running_contrastive_loss / len(dataloader.dataset)
        val_orthogonal_loss = running_orthogonal_loss / len(dataloader.dataset)

        tqdm.write(f'Validation - Total Loss: {val_loss:.4f}, Base Loss: {val_base_loss:.4f}, '
              f'Assignment Loss: {val_assignment_loss:.4f}, Contrastive Loss: {val_contrastive_loss:.4f}, '
              f'Orthogonal Loss: {val_orthogonal_loss:.4f}')

    return val_loss