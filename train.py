import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from model import SimpleCNN
import torch.nn as nn
import torch.optim as optim
from utils import download_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("当前使用的 GPU 是：", torch.cuda.get_device_name(0))

# 下载并准备数据
download_dataset()
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
train_data = ImageFolder(root='data/cats_and_dogs_filtered/train', transform=transform)
val_data = ImageFolder(root='data/cats_and_dogs_filtered/validation', transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

# 定义模型
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 记录每个 epoch 的 loss 和 accuracy
epoch_losses = []
epoch_accuracies = []

# 训练模型
for epoch in range(10):
    model.train()
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

torch.cuda.empty_cache()

# 可视化训练过程
epochs = range(1, len(epoch_losses) + 1)
plt.figure(figsize=(10, 5))
plt.plot(epochs, epoch_losses, 'b-o', label='训练 Loss')
plt.plot(epochs, epoch_accuracies, 'orange', marker='x', label='验证 Accuracy (%)')
plt.title("训练过程曲线")
plt.xlabel("Epoch")
plt.ylabel("值")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("training_curve.png")
plt.show()

# 最终验证集评估
print(f"最终验证准确率：{epoch_accuracies[-1]:.2f}%")

# 保存模型参数
torch.save(model.state_dict(), "cat_dog_model.pth")
print("模型已保存为 cat_dog_model.pth")
