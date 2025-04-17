import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torch.nn as nn
import torch.optim as optim
from utils import download_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv


# 设置中文字体支持，避免乱码警告
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 初始化设备、打印显卡信息
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("当前使用的 GPU 是：", torch.cuda.get_device_name(0))

# 下载并准备数据
download_dataset()
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),  # 数据增强
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])
train_data = ImageFolder(root='data/cats_and_dogs_filtered/train', transform=transform)
val_data = ImageFolder(root='data/cats_and_dogs_filtered/validation',
                       transform=transforms.Compose([
                           transforms.Resize((128, 128)),
                           transforms.ToTensor()
                       ]))

# 加载成 DataLoader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True) # 训练集打乱 (shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

# 使用 ResNet18 模型
model = resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# 损失函数 + 优化器定义
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 创建训练日志 CSV
csv_file = open("train_log_resnet.csv", mode="w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Epoch", "Loss", "Val Accuracy"])


epoch_losses = []
epoch_accuracies = []

# 训练循环模型
for epoch in range(10):
    model.train() # 进入训练模式
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    epoch_losses.append(epoch_loss)

    # 在验证集上测试准确率
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = 100 * correct / total
    epoch_accuracies.append(val_accuracy)

    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
    csv_writer.writerow([epoch + 1, f"{epoch_loss:.4f}", f"{val_accuracy:.2f}"])

# 清空 GPU 缓存
torch.cuda.empty_cache()

# 可视化训练过程
epochs = range(1, len(epoch_losses) + 1)
plt.figure(figsize=(10, 5))
plt.plot(epochs, epoch_losses, 'b-o', label='训练 Loss')
plt.plot(epochs, epoch_accuracies, 'orange', marker='x', label='验证 Accuracy (%)')
plt.title("训练过程曲线（ResNet18 刷分版）")
plt.xlabel("Epoch")
plt.ylabel("值")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_curve_resnet.png")
plt.show()

# 输出最终准确率
print(f"最终验证准确率：{epoch_accuracies[-1]:.2f}%")
csv_file.close()


# 保存模型参数
torch.save(model.state_dict(), "cat_dog_model_resnet18.pth")
print("模型已保存为 cat_dog_model_resnet18.pth")
